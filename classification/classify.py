"""Label SCERT topics Covered / Partially Covered / Not Covered from alignment scores.

The score is ``best_coverage_any`` from ``topic_matches.csv`` (see ``docs/ALIGNMENT.md`` §2).
The cut-offs come from ``processed/alignment/thresholds.json``:

    Covered            score >= t_covered
    Partially Covered  t_partial <= score < t_covered
    Not Covered        score < t_partial
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

LABELS = ("Not Covered", "Partially Covered", "Covered")

REQUIRED_COLUMNS = {
    "scert_topic_id",
    "scert_chapter",
    "scert_topic_title",
    "rank",
    "ncert_topic_id",
    "ncert_chapter",
    "ncert_topic_title",
    "coverage",
    "best_coverage_any",
    "evidence_scert_text",
    "evidence_ncert_text",
    "evidence_sim",
}


@dataclass(frozen=True)
class Thresholds:
    """Cut-offs for one model, with the bootstrap intervals from the evaluation."""

    model_name: str
    t_partial: float
    t_covered: float
    t_partial_ci: tuple[float, float]
    t_covered_ci: tuple[float, float]
    label_source: str
    date: str

    @property
    def provisional(self) -> bool:
        """True unless the gold set the cut-offs were fitted on was labelled by people."""
        return self.label_source.strip().lower() != "human"


def load_thresholds(path: Path | str, model: str) -> Thresholds:
    """Read the cut-offs for ``model`` (``bge``, ``e5`` or ``tfidf``) from thresholds.json."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        m = data["thresholds"][model]
    except KeyError:
        raise KeyError(
            f"no thresholds for model {model!r}; available: {sorted(data['thresholds'])}"
        ) from None
    return Thresholds(
        model_name=m["model"],
        t_partial=float(m["t_partial"]),
        t_covered=float(m["t_covered"]),
        t_partial_ci=(float(m["ci95"]["t_partial"][0]), float(m["ci95"]["t_partial"][1])),
        t_covered_ci=(float(m["ci95"]["t_covered"][0]), float(m["ci95"]["t_covered"][1])),
        label_source=str(data["label_source"]),
        date=str(data["date"]),
    )


def label_score(score: float, t: Thresholds) -> str:
    """Gap label for one coverage score."""
    if score >= t.t_covered:
        return "Covered"
    if score >= t.t_partial:
        return "Partially Covered"
    return "Not Covered"


def label_confidence(score: float, t: Thresholds) -> str:
    """``uncertain`` if the score lies inside the 95% interval of either threshold, meaning a
    refit of the thresholds on resampled gold topics could change the label; else ``stable``."""
    lo_p, hi_p = t.t_partial_ci
    lo_c, hi_c = t.t_covered_ci
    if lo_p <= score <= hi_p or lo_c <= score <= hi_c:
        return "uncertain"
    return "stable"


def margin(score: float, t: Thresholds) -> float:
    """Distance from the score to the nearest threshold."""
    return min(abs(score - t.t_partial), abs(score - t.t_covered))


def build_gap_matrix(matches: pd.DataFrame, t: Thresholds, top_n: int = 3) -> pd.DataFrame:
    """One row per SCERT topic: score, label, confidence and the NCERT evidence behind it.

    ``matches`` is ``topic_matches.csv`` (top-k NCERT topics per SCERT topic). Topic order
    follows the file. ``best_coverage_any`` is the same on every row of a topic, so the largest
    value is used.
    """
    missing = REQUIRED_COLUMNS - set(matches.columns)
    if missing:
        raise ValueError(f"topic_matches is missing columns: {sorted(missing)}")
    rows = []
    for topic_id, grp in matches.groupby("scert_topic_id", sort=False):
        grp = grp.sort_values("rank")
        top = grp.iloc[0]
        score = float(grp["best_coverage_any"].max())
        near = grp.head(top_n)
        rows.append(
            {
                "scert_topic_id": topic_id,
                "scert_unit": top["scert_chapter"],
                "scert_topic_title": top["scert_topic_title"],
                "score": score,
                "label": label_score(score, t),
                "confidence": label_confidence(score, t),
                "margin": margin(score, t),
                "provisional": t.provisional,
                "best_ncert_topic_id": top["ncert_topic_id"],
                "best_ncert_chapter": top["ncert_chapter"],
                "best_ncert_topic_title": top["ncert_topic_title"],
                "best_ncert_coverage": float(top["coverage"]),
                "ncert_matches": " | ".join(
                    f"{r.ncert_chapter}: {r.ncert_topic_title} ({r.coverage:.3f})"
                    for r in near.itertuples()
                ),
                "evidence_sim": float(top["evidence_sim"]),
                "evidence_scert_text": top["evidence_scert_text"],
                "evidence_ncert_text": top["evidence_ncert_text"],
            }
        )
    return pd.DataFrame(rows)


def unit_summary(gap_matrix: pd.DataFrame) -> pd.DataFrame:
    """Label counts and mean score per SCERT unit, in book order."""
    counts = (
        gap_matrix.groupby("scert_unit", sort=False)["label"]
        .value_counts()
        .unstack(fill_value=0)
        .reindex(columns=list(LABELS), fill_value=0)
    )
    counts["topics"] = counts.sum(axis=1)
    counts["pct_not_covered"] = (100 * counts["Not Covered"] / counts["topics"]).round(1)
    counts["mean_score"] = gap_matrix.groupby("scert_unit", sort=False)["score"].mean().round(4)
    return counts.reset_index()


def review_queue(gap_matrix: pd.DataFrame) -> pd.DataFrame:
    """Topics whose label could change under threshold uncertainty, closest to a cut-off first."""
    q = gap_matrix[gap_matrix["confidence"] == "uncertain"]
    return q.sort_values("margin").reset_index(drop=True)
