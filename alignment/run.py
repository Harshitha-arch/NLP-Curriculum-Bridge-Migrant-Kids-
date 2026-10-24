"""Embed content chunks and align SCERT (destination) topics to NCERT (origin) topics.

Usage:
    uv run python -m alignment.run --model bge|e5|tfidf [--config configs/alignment.yaml]

Outputs (in paths.output_dir/<model slug>/):
    topic_coverage.csv          rows SCERT topic_id, cols NCERT topic_id (Phase 3 exit matrix)
    topic_coverage_reverse.csv  rows NCERT, cols SCERT
    topic_centroid_sim.csv      symmetric centroid cosine, all content topics of both boards
    topic_matches.csv           top-k NCERT topics per SCERT topic, with evidence chunks
    chunk_sim.npy               SCERT x NCERT chunk cosine, + chunk_ids_dest/orig.json
    chapter_heatmap.png         SCERT units x NCERT chapters, cell = max topic coverage
    sanity_report.txt           top-1 chapter per unit, computer units, catch-all topics
    run_meta.json               model, prefix, dims, counts, runtime, git commit, timestamp
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from alignment.align import (
    best_coverage_any,
    centroid_sim,
    chunk_sim,
    top_k,
    topic_coverage,
    topic_order,
)
from alignment.data import by_board, content_only, read_jsonl
from alignment.embed import embed

FLOAT = "%.6f"
# NCERT Ch. 1 and Ch. 9 have no section headings: each is one long topic (8 and 16 chunks),
# which max-over-chunks coverage may favour as a catch-all match.
CATCH_ALL_FILES = ("NCERT/fecu101.pdf", "NCERT/fecu109.pdf")


def git_commit() -> str | None:
    """HEAD commit of the repository, with ``-dirty`` if tracked files changed."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--"], capture_output=True
        ).returncode
        return head + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def chapter_matrix(
    C: pd.DataFrame, topics: dict[str, dict]
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Max topic coverage per (SCERT unit, NCERT chapter), in book order, plus the row and
    column order that puts every unit next to its best chapter (a staircase diagonal)."""
    unit = C.index.map(lambda t: topics[t]["chapter"])
    chapter = C.columns.map(lambda t: topics[t]["chapter"])
    M = C.T.groupby(chapter, sort=False).max().T.groupby(unit, sort=False).max()
    cols = list(M.columns)
    best = M.to_numpy().argmax(axis=1)
    rows = sorted(range(len(M)), key=lambda r: (best[r], -M.iloc[r, best[r]]))
    used = list(dict.fromkeys(cols[best[r]] for r in rows))
    col_order = used + [c for c in cols if c not in used]
    return M, [M.index[r] for r in rows], col_order


def plot_heatmap(M: pd.DataFrame, rows: list[str], cols: list[str], path: Path, title: str):
    """Annotated heatmap of the chapter matrix in the given order."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = M.loc[rows, cols].to_numpy()
    fig, ax = plt.subplots(figsize=(1.0 * len(cols) + 4, 0.45 * len(rows) + 3))
    im = ax.imshow(data, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows)), rows, fontsize=8)
    ax.set_xlabel("NCERT chapter (origin)")
    ax.set_ylabel("SCERT unit (destination)")
    ax.set_title(title, fontsize=10)
    mid = (np.nanmin(data) + np.nanmax(data)) / 2
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            colour = "black" if data[i, j] > mid else "white"
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=6, color=colour)
    fig.colorbar(im, ax=ax, label="max topic coverage")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def sanity_report(
    M: pd.DataFrame,
    best_any: pd.Series,
    matches: pd.DataFrame,
    centroid: pd.DataFrame,
    topics: dict[str, dict],
    computer_units: list[str],
) -> str:
    """Plain-text checks to read after a run; nothing here changes the outputs."""
    lines = ["Top-1 NCERT chapter per SCERT unit (max topic coverage):"]
    for unit, row in M.iterrows():
        lines.append(f"  {unit:<32} -> {row.idxmax():<45} {row.max():.3f}")

    lines.append("")
    ranks = best_any.rank(method="min").astype(int)
    n = len(best_any)
    q = best_any.quantile([0.1, 0.5, 0.9])
    lines.append(
        f"best_coverage_any over {n} SCERT topics: min {best_any.min():.3f}, "
        f"p10 {q[0.1]:.3f}, median {q[0.5]:.3f}, p90 {q[0.9]:.3f}, max {best_any.max():.3f}"
    )
    lines.append("Computer units (no NCERT counterpart; expected lowest), rank 1 = lowest:")
    for unit in computer_units:
        ids = [t for t in best_any.index if topics[t]["chapter"] == unit]
        for t in ids:
            title = topics[t]["topic_title"][:40]
            lines.append(f"  {unit:<28} {title:<40} {best_any[t]:.3f}  rank {ranks[t]}/{n}")
    computer = [t for t in best_any.index if topics[t]["chapter"] in computer_units]
    others = best_any.drop(computer)
    lines.append(
        f"  mean computer {best_any[computer].mean():.3f} vs other topics {others.mean():.3f}; "
        f"{int((others < best_any[computer].max()).sum())} non-computer topics score below the "
        "highest computer topic"
    )

    lines.append("")
    top1_cov = matches[matches["rank"] == 1].set_index("scert_topic_id")["ncert_topic_id"]
    top1_cen = centroid.idxmax(axis=1)
    lines.append("SCERT topics whose top-1 NCERT topic is a single-topic chapter (catch-all):")
    for src in CATCH_ALL_FILES:
        ids = [t for t in centroid.columns if topics[t]["source_file"] == src]
        name = topics[ids[0]]["chapter"] if ids else src
        n_cov = int(top1_cov.isin(ids).sum())
        n_cen = int(top1_cen.isin(ids).sum())
        lines.append(f"  {name:<45} coverage {n_cov:>3}   centroid {n_cen:>3}   (of {n})")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Parse arguments, embed, align and write the outputs."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="bge", help="model key in the config: bge, e5, tfidf")
    parser.add_argument("--config", default="configs/alignment.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    model_cfg = cfg["models"][args.model]
    paths = cfg["paths"]
    dest_board, orig_board = cfg["boards"]["destination"], cfg["boards"]["origin"]
    out_dir = Path(paths["output_dir"]) / model_cfg["slug"]
    started = time.perf_counter()

    chunks = content_only(read_jsonl(paths["chunks"]))
    topics = {t["topic_id"]: t for t in content_only(read_jsonl(paths["topics"]))}
    E = embed(chunks, model_cfg, cfg["embedding"], Path(paths["embedding_cache"]))
    board = np.array([c["board"] for c in chunks])
    dest, orig = by_board(chunks, dest_board), by_board(chunks, orig_board)
    E_dest, E_orig = E[board == dest_board], E[board == orig_board]
    dest_tids, orig_tids = [c["topic_id"] for c in dest], [c["topic_id"] for c in orig]

    S = chunk_sim(E_dest, E_orig)
    C, rows, cols = topic_coverage(S, dest_tids, orig_tids)
    C_rev, rev_rows, rev_cols = topic_coverage(S.T, orig_tids, dest_tids)
    best_any, _ = best_coverage_any(S, dest_tids)
    cen, cen_ids = centroid_sim(E, [c["topic_id"] for c in chunks], [c["chars"] for c in chunks])
    assert cen_ids == topic_order([c["topic_id"] for c in chunks])

    C_df = pd.DataFrame(C, index=pd.Index(rows, name="scert_topic_id"), columns=cols)
    rev_df = pd.DataFrame(C_rev, index=pd.Index(rev_rows, name="ncert_topic_id"), columns=rev_cols)
    cen_df = pd.DataFrame(cen, index=pd.Index(cen_ids, name="topic_id"), columns=cen_ids)
    best_s = pd.Series(best_any, index=rows, name="best_coverage_any")
    cen_block = cen_df.loc[rows, cols]

    n_ev = cfg["alignment"]["evidence_chars"]
    records = []
    for m in top_k(C, cfg["alignment"]["top_k"], S, dest_tids, orig_tids):
        d, o = topics[m.dest_topic], topics[m.orig_topic]
        dc, oc = dest[m.evidence_dest], orig[m.evidence_orig]
        records.append(
            {
                "scert_topic_id": m.dest_topic,
                "scert_chapter": d["chapter"],
                "scert_topic_title": d["topic_title"],
                "rank": m.rank,
                "ncert_topic_id": m.orig_topic,
                "ncert_chapter": o["chapter"],
                "ncert_topic_title": o["topic_title"],
                "coverage": m.coverage,
                "centroid_sim": float(cen_block.at[m.dest_topic, m.orig_topic]),
                "best_coverage_any": float(best_s[m.dest_topic]),
                "evidence_scert_chunk_id": dc["chunk_id"],
                "evidence_ncert_chunk_id": oc["chunk_id"],
                "evidence_sim": m.evidence_sim,
                "evidence_scert_text": dc["text"][:n_ev],
                "evidence_ncert_text": oc["text"][:n_ev],
            }
        )
    matches = pd.DataFrame(records)

    out_dir.mkdir(parents=True, exist_ok=True)
    C_df.to_csv(out_dir / "topic_coverage.csv", float_format=FLOAT)
    rev_df.to_csv(out_dir / "topic_coverage_reverse.csv", float_format=FLOAT)
    cen_df.to_csv(out_dir / "topic_centroid_sim.csv", float_format=FLOAT)
    matches.to_csv(out_dir / "topic_matches.csv", index=False, float_format=FLOAT)
    np.save(out_dir / "chunk_sim.npy", S.astype(np.float32))
    for name, ids in (("chunk_ids_dest.json", dest), ("chunk_ids_orig.json", orig)):
        ids = [c["chunk_id"] for c in ids]
        (out_dir / name).write_text(json.dumps(ids, indent=0) + "\n", encoding="utf-8")

    M, unit_order, chapter_order = chapter_matrix(C_df, topics)
    title = f"{model_cfg['name']}: how well NCERT covers each SCERT unit"
    plot_heatmap(M, unit_order, chapter_order, out_dir / "chapter_heatmap.png", title)
    report = sanity_report(M, best_s, matches, cen_block, topics, cfg["gold"]["computer_units"])
    (out_dir / "sanity_report.txt").write_text(report, encoding="utf-8")

    meta = {
        "model_key": args.model,
        "model": model_cfg["name"],
        "prefix": model_cfg.get("prefix", ""),
        "dims": int(E.shape[1]),
        "chunks": {dest_board: len(dest), orig_board: len(orig)},
        "topics": {dest_board: len(rows), orig_board: len(cols)},
        "runtime_s": round(time.perf_counter() - started, 1),
        "git_commit": git_commit(),
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(report)
    print(f"{len(rows)} x {len(cols)} topics, {S.shape[0]} x {S.shape[1]} chunks -> {out_dir}")


if __name__ == "__main__":
    main()
