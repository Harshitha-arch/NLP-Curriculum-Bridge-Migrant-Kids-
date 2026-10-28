"""Tests for classification/calibrate.py (threshold refit from researcher-reviewed labels)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from alignment import evaluate
from classification import calibrate as cal
from classification.classify import (
    label_confidence,
    label_score,
    load_thresholds,
)

CLASSES = list(evaluate.CLASSES)


def random_case(seed: int, n: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Scores that rise with the class, plus noise (and some repeated scores)."""
    rng = np.random.default_rng(seed)
    y = rng.choice(CLASSES, size=n)
    base = np.array([CLASSES.index(c) for c in y]) * 0.08 + 0.55
    s = np.round(base + rng.normal(0, 0.06, n), 3)
    return s, y


@pytest.mark.parametrize("seed", range(8))
def test_fast_fit_matches_reference_fit(seed):
    s, y = random_case(seed)
    ref = evaluate.fit_thresholds(s, y)
    fast = cal.fit_thresholds_fast(s, y)
    assert fast[:2] == pytest.approx(ref[:2], abs=1e-12)
    assert fast[2] == pytest.approx(ref[2], abs=1e-9)


def test_fast_fit_matches_reference_when_a_class_is_missing():
    s, y = random_case(3)
    y = np.where(y == "Covered", "Partially Covered", y)
    ref = evaluate.fit_thresholds(s, y)
    fast = cal.fit_thresholds_fast(s, y)
    assert fast[:2] == pytest.approx(ref[:2], abs=1e-12)
    assert fast[2] == pytest.approx(ref[2], abs=1e-9)


def test_reported_f1_is_the_f1_of_the_reported_thresholds():
    s, y = random_case(11, n=60)
    tp, tc, f = cal.fit_thresholds_fast(s, y)
    assert tp < tc
    assert f == pytest.approx(evaluate.macro_f1(y, evaluate.classify(s, tp, tc)), abs=1e-9)


def test_separable_scores_give_macro_f1_one_and_cuts_between_the_groups():
    s = np.array([0.50, 0.52, 0.65, 0.67, 0.80, 0.82])
    y = np.array(["Not Covered"] * 2 + ["Partially Covered"] * 2 + ["Covered"] * 2)
    tp, tc, f = cal.fit_thresholds_fast(s, y)
    assert f == pytest.approx(1.0)
    assert 0.52 < tp <= 0.65 and 0.67 < tc <= 0.80


def test_bootstrap_matches_reference_with_the_same_seed():
    s, y = random_case(5, n=24)
    ref = evaluate.bootstrap(s, y, 6, 42)
    fast = cal.bootstrap_ci(s, y, 6, 42)
    for k in ref:
        assert fast[k] == pytest.approx(ref[k], abs=1e-9)


def test_leave_one_out_matches_reference():
    s, y = random_case(7, n=20)
    assert cal.loo_macro_f1(s, y) == pytest.approx(evaluate.loo_macro_f1(s, y), abs=1e-9)


def write_inputs(tmp_path, scores: dict[str, float], labels: dict[str, str]):
    """topic_matches.csv (rank-1 rows carry the score) and a labels file."""
    rows = [{"scert_topic_id": t, "rank": 1, "best_coverage_any": v} for t, v in scores.items()] + [
        {"scert_topic_id": t, "rank": 2, "best_coverage_any": v} for t, v in scores.items()
    ]
    matches = tmp_path / "topic_matches.csv"
    pd.DataFrame(rows).to_csv(matches, index=False)
    labels_path = tmp_path / "labels.csv"
    pd.DataFrame({"scert_topic_id": list(labels), "label": list(labels.values())}).to_csv(
        labels_path, index=False
    )
    return labels_path, matches


def test_load_skips_blank_labels_and_keeps_label_order(tmp_path):
    scores = {"a": 0.5, "b": 0.7, "c": 0.8}
    labels_path, matches = write_inputs(
        tmp_path, scores, {"c": "Covered", "a": "Not Covered", "b": ""}
    )
    s, y = cal.load_labelled_scores(labels_path, matches)
    assert list(y) == ["Covered", "Not Covered"]
    assert list(s) == [0.8, 0.5]


def test_load_rejects_unknown_labels_duplicates_and_unscored_topics(tmp_path):
    labels_path, matches = write_inputs(tmp_path, {"a": 0.5}, {"a": "Maybe"})
    with pytest.raises(ValueError, match="unknown labels"):
        cal.load_labelled_scores(labels_path, matches)
    pd.DataFrame({"scert_topic_id": ["a", "a"], "label": ["Covered", "Covered"]}).to_csv(
        labels_path, index=False
    )
    with pytest.raises(ValueError, match="duplicate"):
        cal.load_labelled_scores(labels_path, matches)
    pd.DataFrame({"scert_topic_id": ["zzz"], "label": ["Covered"]}).to_csv(labels_path, index=False)
    with pytest.raises(ValueError, match="not in"):
        cal.load_labelled_scores(labels_path, matches)


def test_main_writes_thresholds_that_classification_can_read_and_is_deterministic(tmp_path):
    s, y = random_case(2, n=40)
    ids = [f"t{i}" for i in range(len(s))]
    labels_path, matches = write_inputs(
        tmp_path, dict(zip(ids, s, strict=True)), dict(zip(ids, y, strict=True))
    )
    align_cfg = tmp_path / "alignment.yaml"
    align_cfg.write_text(
        yaml.safe_dump(
            {
                "paths": {"output_dir": str(tmp_path)},
                "models": {"bge": {"slug": ".", "name": "BAAI/test"}},
            }
        )
    )
    out = tmp_path / "thresholds.json"
    cfg = tmp_path / "classification.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "paths": {"alignment_config": str(align_cfg), "thresholds": str(out)},
                "model": "bge",
                "calibration": {
                    "labels": str(labels_path),
                    "label_source": "Researcher-reviewed (test)",
                    "bootstrap": 20,
                    "seed": 42,
                },
            }
        )
    )
    cal.main(["--config", str(cfg)])
    first = json.loads(out.read_text())
    cal.main(["--config", str(cfg)])
    assert json.loads(out.read_text()) == first

    t = load_thresholds(out, "bge")
    assert t.t_partial < t.t_covered
    assert t.label_source == "Researcher-reviewed (test)" and t.provisional
    assert first["gold_topics"] == 40
    assert sum(first["topic_labels"].values()) == 40
    assert first["thresholds"]["bge"]["macro_f1_loo"] is not None


def test_dry_run_writes_nothing(tmp_path, capsys):
    s, y = random_case(2, n=40)
    ids = [f"t{i}" for i in range(len(s))]
    labels_path, matches = write_inputs(
        tmp_path, dict(zip(ids, s, strict=True)), dict(zip(ids, y, strict=True))
    )
    align_cfg = tmp_path / "alignment.yaml"
    align_cfg.write_text(
        yaml.safe_dump(
            {
                "paths": {"output_dir": str(tmp_path)},
                "models": {"bge": {"slug": ".", "name": "BAAI/test"}},
            }
        )
    )
    out = tmp_path / "thresholds.json"
    cfg = tmp_path / "classification.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "paths": {"alignment_config": str(align_cfg), "thresholds": str(out)},
                "model": "bge",
                "calibration": {
                    "labels": str(labels_path),
                    "label_source": "x",
                    "bootstrap": 5,
                    "seed": 1,
                },
            }
        )
    )
    cal.main(["--config", str(cfg), "--dry-run"])
    assert not out.exists()
    assert "dry run" in capsys.readouterr().out


# --- the committed files must agree with each other (guards against stale thresholds) ------------

REPO = Path(__file__).resolve().parent.parent


def test_committed_thresholds_are_what_the_committed_labels_produce():
    """thresholds.json must be reproducible: refit from eval/human_labels.csv, same result."""
    cfg = yaml.safe_load((REPO / "configs" / "classification.yaml").read_text(encoding="utf-8"))
    c = cfg["calibration"]
    align = yaml.safe_load((REPO / cfg["paths"]["alignment_config"]).read_text(encoding="utf-8"))
    model = cfg["model"]
    m = align["models"][model]
    matches = REPO / align["paths"]["output_dir"] / m["slug"] / "topic_matches.csv"
    s, y = cal.load_labelled_scores(REPO / c["labels"], matches)
    fresh = cal.calibrate(s, y, m["name"], c["bootstrap"], c["seed"])
    committed = json.loads((REPO / cfg["paths"]["thresholds"]).read_text(encoding="utf-8"))
    assert committed["thresholds"][model] == fresh
    assert committed["gold_topics"] == len(y)
    assert committed["topic_labels"] == {k: int((y == k).sum()) for k in CLASSES}


def test_committed_gap_matrix_matches_the_committed_thresholds():
    """Every label and confidence in gap_matrix.csv follows from its score and thresholds.json."""
    cfg = yaml.safe_load((REPO / "configs" / "classification.yaml").read_text(encoding="utf-8"))
    t = load_thresholds(REPO / cfg["paths"]["thresholds"], cfg["model"])
    gm = pd.read_csv(REPO / cfg["paths"]["output_dir"] / "gap_matrix.csv")
    assert len(gm) == 132
    for row in gm.itertuples():
        assert row.label == label_score(row.score, t), row.scert_topic_id
        assert row.confidence == label_confidence(row.score, t), row.scert_topic_id
