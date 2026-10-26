"""Tests for the Phase 4 gap classification."""

import json

import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from classification.classify import (
    LABELS,
    Thresholds,
    build_gap_matrix,
    label_confidence,
    label_score,
    load_thresholds,
    margin,
    review_queue,
    unit_summary,
)

T = Thresholds(
    model_name="m",
    t_partial=0.70,
    t_covered=0.80,
    t_partial_ci=(0.65, 0.72),
    t_covered_ci=(0.78, 0.85),
    label_source="AI-labelled (LLM), not human-reviewed",
    date="2026-10-24",
)


def matches(rows: list[tuple]) -> pd.DataFrame:
    """topic_matches-like frame from (topic, unit, rank, ncert title, coverage, any) tuples."""
    return pd.DataFrame(
        [
            {
                "scert_topic_id": topic,
                "scert_chapter": unit,
                "scert_topic_title": f"title {topic}",
                "rank": rank,
                "ncert_topic_id": f"N{rank}",
                "ncert_chapter": "Ch",
                "ncert_topic_title": ncert,
                "coverage": cov,
                "best_coverage_any": best,
                "evidence_scert_text": "s",
                "evidence_ncert_text": "n",
                "evidence_sim": 0.9,
            }
            for topic, unit, rank, ncert, cov, best in rows
        ]
    )


def test_label_boundaries_are_inclusive_at_the_lower_edge():
    assert label_score(0.80, T) == "Covered"
    assert label_score(0.7999, T) == "Partially Covered"
    assert label_score(0.70, T) == "Partially Covered"
    assert label_score(0.6999, T) == "Not Covered"


@given(a=st.floats(0, 1), b=st.floats(0, 1))
def test_label_never_decreases_with_score(a, b):
    lo, hi = sorted((a, b))
    assert LABELS.index(label_score(lo, T)) <= LABELS.index(label_score(hi, T))


def test_confidence_is_uncertain_inside_either_interval():
    assert label_confidence(0.66, T) == "uncertain"  # inside the t_partial interval
    assert label_confidence(0.80, T) == "uncertain"  # inside the t_covered interval
    assert label_confidence(0.50, T) == "stable"
    assert label_confidence(0.75, T) == "stable"  # between the intervals
    assert label_confidence(0.90, T) == "stable"


def test_margin_is_distance_to_nearest_threshold():
    assert margin(0.74, T) == pytest.approx(0.04)
    assert margin(0.78, T) == pytest.approx(0.02)


def test_provisional_until_labels_are_human():
    assert T.provisional
    human = Thresholds(**{**T.__dict__, "label_source": "human"})
    assert not human.provisional


def test_load_thresholds(tmp_path):
    path = tmp_path / "thresholds.json"
    path.write_text(
        json.dumps(
            {
                "label_source": "human",
                "date": "2026-10-30",
                "thresholds": {
                    "bge": {
                        "model": "BAAI/bge-large-en-v1.5",
                        "t_partial": 0.7,
                        "t_covered": 0.8,
                        "ci95": {"t_partial": [0.6, 0.71], "t_covered": [0.75, 0.82]},
                    }
                },
            }
        )
    )
    t = load_thresholds(path, "bge")
    assert (t.t_partial, t.t_covered, t.t_covered_ci) == (0.7, 0.8, (0.75, 0.82))
    assert not t.provisional
    with pytest.raises(KeyError, match="available"):
        load_thresholds(path, "e5")


def test_build_gap_matrix_one_row_per_topic_in_file_order():
    m = matches(
        [
            ("B", "Unit 2", 2, "second", 0.5, 0.82),
            ("B", "Unit 2", 1, "first", 0.6, 0.82),
            ("A", "Unit 1", 1, "only", 0.4, 0.60),
        ]
    )
    gm = build_gap_matrix(m, T, top_n=2)
    assert list(gm["scert_topic_id"]) == ["B", "A"]
    b = gm.iloc[0]
    assert (b["label"], b["best_ncert_topic_title"]) == ("Covered", "first")
    assert b["ncert_matches"] == "Ch: first (0.600) | Ch: second (0.500)"
    assert gm.iloc[1]["label"] == "Not Covered"
    assert gm["provisional"].all()


def test_build_gap_matrix_rejects_missing_columns():
    with pytest.raises(ValueError, match="missing columns"):
        build_gap_matrix(pd.DataFrame({"scert_topic_id": ["A"]}), T)


def test_unit_summary_and_review_queue():
    m = matches(
        [
            ("A", "Unit 1", 1, "x", 0.9, 0.90),  # Covered, stable
            ("B", "Unit 1", 1, "x", 0.5, 0.66),  # Not Covered, uncertain
            ("C", "Unit 2", 1, "x", 0.5, 0.30),  # Not Covered, stable
        ]
    )
    gm = build_gap_matrix(m, T)
    s = unit_summary(gm).set_index("scert_unit")
    assert list(s.index) == ["Unit 1", "Unit 2"]
    assert s.loc["Unit 1", "topics"] == 2
    assert s.loc["Unit 2", "pct_not_covered"] == 100.0
    assert list(review_queue(gm)["scert_topic_id"]) == ["B"]
