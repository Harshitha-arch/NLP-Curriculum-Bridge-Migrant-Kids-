"""Load the gap matrix and the textbook passages a bridge lesson is built from."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from classification.classify import load_thresholds

LABELS = ["Not Covered", "Partially Covered", "Covered"]
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
NO_NCERT = "No related NCERT section found"


@dataclass
class Passage:
    """One textbook topic passed to the model."""

    topic_id: str
    chapter: str
    title: str
    text: str
    coverage: float | None = None  # NCERT only: how well this topic covers the SCERT topic


@dataclass
class TopicSources:
    """Everything a bridge lesson for one SCERT topic is built from."""

    topic_id: str
    unit: str
    title: str
    label: str
    score: float
    confidence: str
    scert: Passage
    ncert: list[Passage] = field(default_factory=list)
    ncert_rule: str = ""  # why these NCERT passages (or none) were chosen
    t_partial: float | None = None  # cut-off the rule used
    label_mismatch: bool = False  # score vs t_partial disagrees with the gap-matrix label


def load_gap_matrix(path: str | Path) -> pd.DataFrame:
    """Phase 4 gap matrix, one row per SCERT topic, in book order."""
    return pd.read_csv(path)


def load_topics(path: str | Path) -> dict[str, dict]:
    """topics.jsonl as {topic_id: record}."""
    with open(path, encoding="utf-8") as f:
        return {rec["topic_id"]: rec for rec in map(json.loads, f)}


def load_matches(path: str | Path) -> pd.DataFrame:
    """Ranked NCERT matches per SCERT topic (alignment topic_matches.csv)."""
    return pd.read_csv(path)


def filter_topics(gap: pd.DataFrame, labels: list[str], unit: str | None = None) -> pd.DataFrame:
    """Rows of the gap matrix with one of `labels`, optionally from one unit, in book order."""
    out = gap[gap["label"].isin(labels)]
    if unit:
        out = out[out["scert_unit"] == unit]
    return out


def load_t_partial(path: str | Path) -> float:
    """``t_partial`` of the primary model in thresholds.json (the Not Covered cut-off)."""
    primary = json.loads(Path(path).read_text(encoding="utf-8"))["primary"]
    return load_thresholds(path, primary).t_partial


def truncate(text: str, max_chars: int) -> str:
    """Cut `text` at the last sentence end within `max_chars` (hard cut if there is none)."""
    if len(text) <= max_chars:
        return text
    head = text[:max_chars]
    ends = [m.start() for m in SENTENCE_END.finditer(head)]
    cut = ends[-1] if ends and ends[-1] > max_chars // 2 else max_chars
    return head[:cut].rstrip() + " [...]"


def get_sources(
    topic_id: str,
    gap: pd.DataFrame,
    topics: dict[str, dict],
    matches: pd.DataFrame,
    n_ncert: int = 2,
    max_scert_chars: int = 6000,
    max_ncert_chars: int = 2500,
    t_partial: float | None = None,
) -> TopicSources:
    """Collect the SCERT topic and its closest NCERT topics for one gap-matrix row.

    With ``t_partial``, a topic scoring below it (Not Covered) gets no NCERT passages: its
    closest NCERT sections are not related enough to build on. ``label_mismatch`` flags a
    gap-matrix label that disagrees with this rule (thresholds.json and gap_matrix.csv out of
    sync)."""
    row = gap.loc[gap["scert_topic_id"] == topic_id]
    if row.empty:
        raise KeyError(f"{topic_id} is not in the gap matrix")
    row = row.iloc[0]
    rec = topics[topic_id]
    scert = Passage(
        topic_id=topic_id,
        chapter=rec["chapter"],
        title=rec["topic_title"],
        text=truncate(rec["text"], max_scert_chars),
    )
    score = float(row["score"])
    below = t_partial is not None and score < t_partial
    if below:
        n_ncert, rule = 0, "none: score < t_partial (Not Covered)"
    else:
        rule = f"top {n_ncert} NCERT matches by coverage"
    mismatch = t_partial is not None and below != (row["label"] == "Not Covered")
    ranked = matches[matches["scert_topic_id"] == topic_id].sort_values("rank").head(n_ncert)
    ncert = [
        Passage(
            topic_id=m.ncert_topic_id,
            chapter=m.ncert_chapter,
            title=m.ncert_topic_title,
            text=truncate(topics[m.ncert_topic_id]["text"], max_ncert_chars),
            coverage=float(m.coverage),
        )
        for m in ranked.itertuples()
    ]
    return TopicSources(
        topic_id=topic_id,
        unit=row["scert_unit"],
        title=row["scert_topic_title"],
        label=row["label"],
        score=score,
        confidence=row["confidence"],
        scert=scert,
        ncert=ncert,
        ncert_rule=rule,
        t_partial=t_partial,
        label_mismatch=mismatch,
    )


def pack_stem(topic_id: str) -> str:
    """File name stem for a saved pack, e.g. SCERT_c6_Term_2_45."""
    return topic_id


DRAFT = "draft"
APPROVED = "approved"


def draft_banner(model: str) -> str:
    """First line of every draft .md."""
    return (
        f"> **Unreviewed draft from {model}.** It must not be used with children before a "
        "teacher has reviewed it."
    )


def _write(out_dir: str | Path, topic_id: str, text: str, meta: dict) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = pack_stem(topic_id)
    md = out / f"{stem}.md"
    md.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")
    (out / f"{stem}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return md


def read_meta(out_dir: str | Path, topic_id: str) -> dict | None:
    """The saved record of a pack, or None if there is none."""
    path = Path(out_dir) / f"{pack_stem(topic_id)}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def save_draft(out_dir: str | Path, topic_id: str, text: str, meta: dict) -> Path:
    """Write a generated, unreviewed pack (status "draft"). Never replaces an approved pack."""
    old = read_meta(out_dir, topic_id)
    if old and old.get("status") == APPROVED:
        raise FileExistsError(f"{topic_id} already has an approved pack; it is not overwritten")
    banner = draft_banner(meta.get("model", "an LLM"))
    return _write(out_dir, topic_id, f"{banner}\n\n{text.strip()}", meta | {"status": DRAFT})


def save_pack(out_dir: str | Path, topic_id: str, text: str, meta: dict) -> Path:
    """Write an approved pack (status "approved") as <stem>.md plus <stem>.json metadata;
    replaces a draft of the same topic. Returns the .md path."""
    if not str(meta.get("approved_by", "")).strip():
        raise ValueError("a pack is saved only after a named reviewer approves it")
    return _write(out_dir, topic_id, text, meta | {"status": APPROVED})


def load_pack(out_dir: str | Path, topic_id: str) -> tuple[str, dict] | None:
    """(body, record) of a saved pack, or None. The body starts at the first "## " heading,
    so the draft banner or the approved title block is left out."""
    meta = read_meta(out_dir, topic_id)
    md = Path(out_dir) / f"{pack_stem(topic_id)}.md"
    if meta is None or not md.exists():
        return None
    text = md.read_text(encoding="utf-8")
    start = text.find("\n## ")
    body = text[start + 1 :] if start >= 0 else text
    return body.strip() + "\n", meta


def generation_record(
    src: TopicSources,
    provider: str,
    gen,
    prompt_version: str,
    temperature: float | None,
    thresholds_file: str,
) -> dict:
    """The .json record of one generated pack (before review)."""
    return {
        "topic_id": src.topic_id,
        "unit": src.unit,
        "title": src.title,
        "label": src.label,
        "score": src.score,
        "confidence": src.confidence,
        "provider": provider,
        "model": gen.model,
        "prompt_version": prompt_version,
        "temperature": temperature,
        "finish_reason": gen.finish_reason,
        "usage": gen.usage,
        "scert_source": src.scert.topic_id,
        "ncert_sources": [p.topic_id for p in src.ncert],
        "ncert_rule": src.ncert_rule,
        "t_partial": src.t_partial,
        "thresholds_file": thresholds_file,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
