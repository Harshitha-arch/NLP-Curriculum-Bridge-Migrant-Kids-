"""Generate bridge-pack drafts for every Not Covered and Partially Covered SCERT topic.

Usage:
    uv run python -m bridge.run [--provider anthropic] [--model claude-haiku-5-5]
                                [--limit N] [--topics ID,ID] [--force] [--dry-run]

Each topic gets processed/bridge/<scert_topic_id>.md + .json with status "draft": an unreviewed
pack that a teacher must check in the app before it is used. The run is resumable: topics that
already have a draft are skipped unless --force, and approved packs are never overwritten.
Rate limits, server errors and network errors are retried with backoff; other errors are
listed at the end and the exit code is 1.

The API key comes from the provider's environment variable or from .streamlit/secrets.toml.
It is never printed or written anywhere.

--dry-run writes each prompt to cache/bridge_prompts/<scert_topic_id>.txt and calls no API.

After every run, processed/bridge/index.csv lists all packs with their status and automated
format checks (missing or empty sections, lesson length, key words, questions, cut-off).
"""

from __future__ import annotations

import argparse
import os
import sys
import tomllib
from pathlib import Path

import pandas as pd
import yaml

from bridge import llm
from bridge.data import (
    APPROVED,
    filter_topics,
    generation_record,
    get_sources,
    load_gap_matrix,
    load_matches,
    load_pack,
    load_t_partial,
    load_topics,
    read_meta,
    save_draft,
)
from bridge.prompts import PROMPT_VERSION, build_prompt, pack_checks

GAP_LABELS = ["Not Covered", "Partially Covered"]
INDEX_COLUMNS = [
    "topic_id",
    "unit",
    "title",
    "label",
    "score",
    "confidence",
    "status",
    "provider",
    "model",
    "prompt_version",
    "ncert_sources",
    "missing_sections",
    "empty_sections",
    "finish_reason",
    "cut_off",
    "input_tokens",
    "output_tokens",
    "lesson_words",
    "key_word_count",
    "question_count",
    "generated_at",
]


def api_key(env_var: str, secrets_path: Path) -> str:
    """The key from the environment, else from secrets.toml; "" if neither has it."""
    if os.environ.get(env_var):
        return os.environ[env_var]
    if secrets_path.exists():
        with open(secrets_path, "rb") as f:
            return str(tomllib.load(f).get(env_var, ""))
    return ""


def build_index(out_dir: Path) -> pd.DataFrame:
    """One row per saved pack (draft or approved) with its record and automated checks."""
    rows = []
    for path in sorted(out_dir.glob("*.json")):
        topic_id = path.stem
        loaded = load_pack(out_dir, topic_id)
        if loaded is None:
            continue
        body, meta = loaded
        usage = meta.get("usage") or {}
        finish = meta.get("finish_reason")
        rows.append(
            {
                "topic_id": topic_id,
                "unit": meta.get("unit"),
                "title": meta.get("title"),
                "label": meta.get("label"),
                "score": meta.get("score"),
                "confidence": meta.get("confidence"),
                "status": meta.get("status"),
                "provider": meta.get("provider"),
                "model": meta.get("model"),
                "prompt_version": meta.get("prompt_version"),
                "ncert_sources": " | ".join(meta.get("ncert_sources") or []),
                "finish_reason": finish,
                "cut_off": str(finish).upper() in {"MAX_TOKENS", "LENGTH"},
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "generated_at": meta.get("generated_at"),
                **pack_checks(body),
            }
        )
    return pd.DataFrame(rows, columns=INDEX_COLUMNS)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="configs/bridge.yaml")
    ap.add_argument("--provider", default=None, help="gemini, openai or anthropic")
    ap.add_argument("--model", default=None, help="default: the provider's model in the config")
    ap.add_argument("--limit", type=int, default=None, help="generate at most N topics")
    ap.add_argument("--topics", default=None, help="comma-separated SCERT topic ids")
    ap.add_argument("--force", action="store_true", help="regenerate existing drafts")
    ap.add_argument("--dry-run", action="store_true", help="write prompts only, no API calls")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    paths = {k: Path(v) for k, v in cfg["paths"].items()}
    provider = args.provider or cfg["default_provider"]
    if provider not in cfg["providers"]:
        ap.error(f"--provider must be one of {sorted(cfg['providers'])}")
    pcfg, gcfg, scfg = cfg["providers"][provider], cfg["generation"], cfg["sources"]
    model = args.model or pcfg["model"]
    out_dir = paths["output_dir"]

    gap = load_gap_matrix(paths["gap_matrix"])
    topics = load_topics(paths["topics"])
    matches = load_matches(paths["topic_matches"])
    t_partial = load_t_partial(paths["thresholds"])

    rows = filter_topics(gap, GAP_LABELS)
    if args.topics:
        wanted = [t.strip() for t in args.topics.split(",") if t.strip()]
        unknown = sorted(set(wanted) - set(rows["scert_topic_id"]))
        if unknown:
            ap.error(f"not Not Covered / Partially Covered topics: {', '.join(unknown)}")
        rows = rows[rows["scert_topic_id"].isin(wanted)]

    todo, approved, existing = [], 0, 0
    for tid in rows["scert_topic_id"]:
        status = (read_meta(out_dir, tid) or {}).get("status")
        if status == APPROVED:
            approved += 1
        elif status and not args.force and not args.dry_run:
            existing += 1
        else:
            todo.append(tid)
    if args.limit is not None:
        todo = todo[: args.limit]
    print(
        f"{len(rows)} gap topics: {approved} approved (never overwritten), {existing} drafts "
        f"already there (use --force to redo), {len(todo)} to {'write' if args.dry_run else 'do'}"
        f"; provider {provider}, model {model}, prompt {PROMPT_VERSION}, t_partial {t_partial}"
    )

    key = ""
    if not args.dry_run and todo:
        key = api_key(pcfg["env"], paths["secrets"])
        if not key:
            print(f"No API key: set {pcfg['env']} or add it to {paths['secrets']}.")
            return 2
        print("API key found.")

    failures: list[tuple[str, str]] = []
    for i, tid in enumerate(todo, 1):
        src = get_sources(
            tid,
            gap,
            topics,
            matches,
            n_ncert=scfg["ncert_matches"],
            max_scert_chars=scfg["max_scert_chars"],
            max_ncert_chars=scfg["max_ncert_chars"],
            t_partial=t_partial,
        )
        system, user = build_prompt(src)
        head = f"[{i}/{len(todo)}] {tid} {src.title} ({src.label})"
        if src.label_mismatch:
            print(f"{head} WARNING: label disagrees with t_partial; re-run classification.run")
        if args.dry_run:
            paths["prompt_cache"].mkdir(parents=True, exist_ok=True)
            out = paths["prompt_cache"] / f"{tid}.txt"
            out.write_text(f"SYSTEM:\n{system}\n\nUSER:\n{user}\n", encoding="utf-8")
            print(f"{head} prompt -> {out}", flush=True)
            continue
        try:
            gen = llm.generate(
                provider,
                key,
                model,
                system,
                user,
                temperature=gcfg["temperature"],
                max_tokens=gcfg["max_tokens"],
                timeout=gcfg["timeout_s"],
                retries=gcfg["retries"],
            )
            meta = generation_record(
                src, provider, gen, PROMPT_VERSION, gcfg["temperature"], cfg["paths"]["thresholds"]
            )
            save_draft(out_dir, tid, gen.text, meta)
        except (llm.LLMError, FileExistsError) as e:
            failures.append((tid, str(e)))
            print(f"{head} FAILED: {e}", flush=True)
            continue
        checks = pack_checks(gen.text)
        u = gen.usage
        notes = [f"missing: {checks['missing_sections']}"] if checks["missing_sections"] else []
        if str(gen.finish_reason).upper() in {"MAX_TOKENS", "LENGTH"}:
            notes.append("CUT OFF")
        print(
            f"{head} ok, {u.get('input_tokens')}/{u.get('output_tokens')} tokens, "
            f"{checks['lesson_words']} lesson words" + (f" ({'; '.join(notes)})" if notes else ""),
            flush=True,
        )

    if not args.dry_run:
        index = build_index(out_dir)
        if len(index):
            index.to_csv(out_dir / "index.csv", index=False)
            print(f"index: {out_dir / 'index.csv'} ({len(index)} packs)")
    if failures:
        print(f"{len(failures)} failed:")
        for tid, msg in failures:
            print(f"  {tid}: {msg}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
