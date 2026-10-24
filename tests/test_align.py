"""Tests for the alignment stage on tiny synthetic inputs (no model downloads)."""

import json

import numpy as np
import pytest

from alignment.align import (
    best_coverage_any,
    centroid_sim,
    chunk_sim,
    top_k,
    topic_coverage,
)
from alignment.data import by_board, content_only, read_jsonl
from alignment.embed import cache_key, embed_tfidf

# Unit vectors along the axes: chunks on the same axis are identical, others orthogonal.
X, Y, Z = np.eye(3, dtype=np.float32)


def unit(*v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_identical_topics_score_one():
    E = np.stack([X, Y])
    S = chunk_sim(E, E)
    C, _, _ = topic_coverage(S, ["d1", "d1"], ["o1", "o1"])
    assert C[0, 0] == pytest.approx(1.0)


def test_coverage_is_directional():
    # Destination topic d has one chunk (X); origin topic o covers X and also Y.
    S = chunk_sim(np.stack([X]), np.stack([X, Y]))
    C, _, _ = topic_coverage(S, ["d"], ["o", "o"])
    C_rev, _, _ = topic_coverage(S.T, ["o", "o"], ["d"])
    assert C[0, 0] == pytest.approx(1.0)  # o covers all of d
    assert C_rev[0, 0] == pytest.approx(0.5)  # d covers only half of o


def test_best_coverage_any_bounds_every_topic_coverage():
    rng = np.random.default_rng(0)
    E_dest, E_orig = rng.normal(size=(6, 4)), rng.normal(size=(5, 4))
    dest_ids = ["a", "a", "b", "c", "c", "c"]
    orig_ids = ["p", "p", "q", "r", "r"]
    S = chunk_sim(E_dest, E_orig)
    C, rows, _ = topic_coverage(S, dest_ids, orig_ids)
    best, best_rows = best_coverage_any(S, dest_ids)
    assert best_rows == rows
    assert np.all(best[:, None] >= C - 1e-6)


def test_ids_follow_first_appearance_order():
    S = np.zeros((3, 3), dtype=np.float32)
    C, rows, cols = topic_coverage(S, ["t2", "t1", "t2"], ["z", "a", "m"])
    assert rows == ["t2", "t1"] and cols == ["z", "a", "m"]
    assert C.shape == (2, 3)


def test_top_k_order_and_evidence():
    # Destination topic d (chunks X, Y). Origin: o1 = (X), o2 = (X, Y), o3 = (Z).
    E_dest = np.stack([X, Y])
    E_orig = np.stack([X, X, Y, Z])
    orig_ids = ["o1", "o2", "o2", "o3"]
    S = chunk_sim(E_dest, E_orig)
    C, _, _ = topic_coverage(S, ["d", "d"], orig_ids)
    matches = top_k(C, 2, S, ["d", "d"], orig_ids)
    assert [m.orig_topic for m in matches] == ["o2", "o1"]
    assert [m.rank for m in matches] == [1, 2]
    assert matches[0].coverage >= matches[1].coverage
    assert matches[1].evidence_orig == 0 and matches[1].evidence_dest == 0
    assert matches[1].evidence_sim == pytest.approx(1.0)


def test_top_k_ties_keep_column_order():
    S = np.ones((1, 3), dtype=np.float32)
    C, _, _ = topic_coverage(S, ["d"], ["a", "b", "c"])
    assert [m.orig_topic for m in top_k(C, 3, S, ["d"], ["a", "b", "c"])] == ["a", "b", "c"]


def test_centroid_sim_symmetric_and_weighted():
    E = np.stack([X, Y, X, unit(1, 1, 0)])
    sim, ids = centroid_sim(E, ["a", "a", "b", "c"], [1, 3, 1, 1])
    assert ids == ["a", "b", "c"]
    assert np.allclose(sim, sim.T)
    assert np.allclose(np.diag(sim), 1.0)
    # "a" is weighted 3:1 towards Y, so it is closer to Y than to X.
    assert sim[0, 1] == pytest.approx(1 / np.sqrt(10))


def test_content_only_filter_keeps_order(tmp_path):
    rows = [
        {"chunk_id": "N_1", "board": "NCERT", "section": "content"},
        {"chunk_id": "N_2", "board": "NCERT", "section": "summary"},
        {"chunk_id": "S_1", "board": "SCERT", "section": "content"},
        {"chunk_id": "N_3", "board": "NCERT", "section": "content"},
        {"chunk_id": "S_2", "board": "SCERT", "section": "exercises"},
    ]
    path = tmp_path / "chunks.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    content = content_only(read_jsonl(path))
    assert [c["chunk_id"] for c in content] == ["N_1", "S_1", "N_3"]
    assert [c["chunk_id"] for c in by_board(content, "NCERT")] == ["N_1", "N_3"]


def test_cache_key_changes_with_text_prefix_and_model():
    items = [("c1", "Magnets attract iron."), ("c2", "Water boils at 100 C.")]
    base = cache_key("bge", "", items)
    assert base == cache_key("bge", "", list(items))
    assert base != cache_key("bge", "", [items[0], ("c2", "Water boils at 100 C!")])
    assert base != cache_key("bge", "", items[::-1])
    assert base != cache_key("bge", "query: ", items)
    assert base != cache_key("e5", "", items)


def test_tfidf_rows_are_unit_length():
    E = embed_tfidf(["magnets attract iron", "iron filings near a magnet", "water boils"])
    assert np.allclose(np.linalg.norm(E, axis=1), 1.0)
