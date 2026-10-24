"""Chunk and topic similarity between a destination and an origin curriculum (pure numpy).

``S[i, j]`` is the cosine between destination chunk ``i`` and origin chunk ``j``. Topic
coverage is directional: ``C[d, o]`` asks how well origin topic ``o`` covers destination
topic ``d``, so ``C`` is not symmetric.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from alignment.embed import l2_normalize


def topic_order(topic_ids: list[str]) -> list[str]:
    """Distinct topic ids in first-appearance order (= jsonl order)."""
    return list(dict.fromkeys(topic_ids))


def topic_rows(topic_ids: list[str]) -> dict[str, np.ndarray]:
    """Chunk row indices of every topic, keyed in first-appearance order."""
    ids = np.asarray(topic_ids)
    return {t: np.flatnonzero(ids == t) for t in topic_order(topic_ids)}


def chunk_sim(E_dest: np.ndarray, E_orig: np.ndarray) -> np.ndarray:
    """Cosine of every destination chunk with every origin chunk (n_dest x n_orig)."""
    return l2_normalize(E_dest) @ l2_normalize(E_orig).T


def topic_coverage(
    S: np.ndarray, dest_topic_ids: list[str], orig_topic_ids: list[str]
) -> tuple[np.ndarray, list[str], list[str]]:
    """``C[d, o]`` = mean over chunks i of topic d of (max over chunks j of topic o of S[i, j]).

    Returns ``C`` with its row (destination) and column (origin) topic ids."""
    dest, orig = topic_rows(dest_topic_ids), topic_rows(orig_topic_ids)
    C = np.empty((len(dest), len(orig)), dtype=np.float32)
    for o, cols in enumerate(orig.values()):
        best = S[:, cols].max(axis=1)  # each destination chunk's best match inside topic o
        for d, rows in enumerate(dest.values()):
            C[d, o] = best[rows].mean()
    return C, list(dest), list(orig)


def best_coverage_any(S: np.ndarray, dest_topic_ids: list[str]) -> tuple[np.ndarray, list[str]]:
    """Per destination topic: mean over its chunks of the max over *all* origin chunks.

    Ignores origin topic boundaries, so it is an upper bound on every ``C[d, o]``."""
    best = S.max(axis=1)
    dest = topic_rows(dest_topic_ids)
    return np.array([best[rows].mean() for rows in dest.values()], dtype=np.float32), list(dest)


def centroid_sim(
    E: np.ndarray, topic_ids: list[str], chars: list[int] | np.ndarray
) -> tuple[np.ndarray, list[str]]:
    """Cosine between topic centroids: chunk embeddings mean-pooled per topic, weighted by
    chunk length, renormalised. Symmetric topic x topic matrix."""
    weights = np.asarray(chars, dtype=np.float64)
    rows = topic_rows(topic_ids)
    centroids = np.stack([(weights[r, None] * E[r]).sum(axis=0) for r in rows.values()])
    centroids = l2_normalize(centroids)
    return centroids @ centroids.T, list(rows)


@dataclass
class Match:
    """One ranked origin topic for a destination topic, with its strongest chunk pair."""

    dest_topic: str
    rank: int
    orig_topic: str
    coverage: float
    evidence_dest: int  # chunk row in S
    evidence_orig: int  # chunk column in S
    evidence_sim: float


def top_k(
    C: np.ndarray,
    k: int,
    S: np.ndarray,
    dest_topic_ids: list[str],
    orig_topic_ids: list[str],
) -> list[Match]:
    """The ``k`` best origin topics per destination topic by ``C`` (ties keep column order),
    each with the (destination chunk, origin chunk) pair of highest ``S`` in that topic pair."""
    dest, orig = topic_rows(dest_topic_ids), topic_rows(orig_topic_ids)
    dest_ids, orig_ids = list(dest), list(orig)
    matches = []
    for d, d_id in enumerate(dest_ids):
        order = np.argsort(-C[d], kind="stable")[:k]
        for rank, o in enumerate(order, start=1):
            rows, cols = dest[d_id], orig[orig_ids[o]]
            block = S[np.ix_(rows, cols)]
            i, j = np.unravel_index(np.argmax(block), block.shape)
            matches.append(
                Match(
                    d_id,
                    rank,
                    orig_ids[o],
                    float(C[d, o]),
                    int(rows[i]),
                    int(cols[j]),
                    float(block[i, j]),
                )
            )
    return matches
