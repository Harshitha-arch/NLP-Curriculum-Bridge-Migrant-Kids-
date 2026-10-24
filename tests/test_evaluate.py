"""Tests for the gold-set evaluation (retrieval metrics, topic labels, thresholds, kappa)."""

import numpy as np
import pytest

from alignment.evaluate import (
    CLASSES,
    classify,
    fit_thresholds,
    macro_f1,
    retrieval_metrics,
    topic_label,
)


def test_topic_label_rules():
    assert topic_label(["none", "partial", "same"]) == "Covered"
    assert topic_label(["none", "partial"]) == "Partially Covered"
    assert topic_label(["prerequisite", "none"]) == "Partially Covered"
    assert topic_label(["none", "none"]) == "Not Covered"


def test_retrieval_metrics():
    ranked = {"a": ["x", "y", "z"], "b": ["y", "z", "x"], "c": ["p", "q", "r"], "d": ["x"]}
    relevant = {"a": {"x"}, "b": {"x"}, "c": set(), "d": {"unjudged"}}
    m = retrieval_metrics(ranked, relevant)
    # c has no relevant pair and is skipped; d's relevant topic is never ranked.
    assert m["n_topics"] == 3
    assert m["hit@1"] == pytest.approx(1 / 3)
    assert m["hit@3"] == pytest.approx(2 / 3)
    assert m["mrr@3"] == pytest.approx((1 + 1 / 3) / 3)


def test_classify_boundaries():
    pred = classify(np.array([0.1, 0.5, 0.7, 0.9]), 0.5, 0.8)
    assert list(pred) == [CLASSES[0], CLASSES[1], CLASSES[1], CLASSES[2]]


def test_fit_thresholds_separates_clean_classes():
    scores = np.array([0.50, 0.52, 0.55, 0.70, 0.72, 0.90, 0.95])
    y = np.array([CLASSES[0]] * 3 + [CLASSES[1]] * 2 + [CLASSES[2]] * 2, dtype=object)
    tp, tc, f = fit_thresholds(scores, y)
    assert 0.55 < tp < 0.70 < 0.72 < tc < 0.90
    assert f == pytest.approx(1.0)
    assert macro_f1(y, classify(scores, tp, tc)) == pytest.approx(1.0)


def test_fit_thresholds_is_ordered_and_deterministic():
    rng = np.random.default_rng(1)
    scores = rng.uniform(0.5, 0.9, 30)
    y = rng.choice(np.array(CLASSES, dtype=object), 30)
    first = fit_thresholds(scores, y)
    assert first == fit_thresholds(scores, y)
    assert first[0] < first[1]
