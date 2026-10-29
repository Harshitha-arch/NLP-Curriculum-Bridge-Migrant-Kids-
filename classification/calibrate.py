"""Refit the gap-classification thresholds from researcher-reviewed labels.

Usage:
    uv run python -m classification.calibrate [--config configs/classification.yaml]
                                              [--model bge] [--dry-run]

Reads ``eval/human_labels.csv`` (one Covered / Partially Covered / Not Covered label per SCERT
topic; blank rows are skipped) and the score of each topic (``best_coverage_any`` in
``topic_matches.csv``), then finds the two cut-offs ``t_partial < t_covered`` that maximise
macro-F1. It also reports bootstrap 95% intervals (used for the ``uncertain`` flag in Phase 4)
and a leave-one-topic-out macro-F1.

Writes ``processed/alignment/thresholds.json`` (the file ``classification.run`` reads). The
fit is deterministic: same labels, scores and seed give the same file.

The fit is the same as ``alignment.evaluate.fit_thresholds`` (candidates are the midpoints
between consecutive distinct scores plus one value below and above all scores; the first
best pair in ascending order wins), but vectorised so the bootstrap takes seconds.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from alignment.evaluate import CLASSES, classify, macro_f1

TOL = 1e-12  # same tie tolerance as alignment.evaluate.fit_thresholds


def fit_thresholds_fast(scores: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """``t_partial < t_covered`` maximising macro-F1; returns ``(t_partial, t_covered, f1)``.

    ``y`` holds class names from ``CLASSES``. Same result as
    ``alignment.evaluate.fit_thresholds``, computed from prefix counts instead of one
    classification per candidate pair.
    """
    scores = np.asarray(scores, dtype=float)
    codes = np.array([CLASSES.index(v) for v in y])
    n = len(scores)
    order = np.argsort(scores, kind="stable")
    s_sorted, c_sorted = scores[order], codes[order]

    u = np.unique(scores)
    cuts = np.concatenate([[u[0] - 1e-3], (u[:-1] + u[1:]) / 2, [u[-1] + 1e-3]])
    below = np.searchsorted(s_sorted, cuts, side="left")  # samples with score < cut

    # cum[k, i] = how many of the i lowest-scoring samples are in class k
    cum = np.zeros((3, n + 1), dtype=float)
    for k in range(3):
        cum[k, 1:] = np.cumsum(c_sorted == k)
    support = cum[:, n]

    a, b = np.triu_indices(len(cuts), k=1)  # ascending t_partial, then ascending t_covered
    i, j = below[a], below[b]
    hits = np.stack([cum[0, i], cum[1, j] - cum[1, i], cum[2, n] - cum[2, j]])
    predicted = np.stack([i, j - i, n - j]).astype(float)
    denom = predicted + support[:, None]
    f1 = np.divide(2 * hits, denom, out=np.zeros_like(hits), where=denom > 0)
    macro = f1.mean(axis=0)

    best = int(np.argmax(macro >= macro.max() - TOL))
    return float(cuts[a[best]]), float(cuts[b[best]]), float(macro[best])


def bootstrap_ci(scores: np.ndarray, y: np.ndarray, n_boot: int, seed: int) -> dict:
    """95% percentile intervals for both thresholds and macro-F1, refitting on each resample."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(scores), len(scores))
        rows.append(fit_thresholds_fast(scores[idx], y[idx]))
    ci = np.percentile(np.array(rows), [2.5, 97.5], axis=0)
    return {
        k: [float(ci[0, i]), float(ci[1, i])]
        for i, k in enumerate(("t_partial", "t_covered", "macro_f1"))
    }


def loo_macro_f1(scores: np.ndarray, y: np.ndarray) -> float:
    """Leave-one-topic-out macro-F1: fit without each topic, then label that topic."""
    pred = np.empty(len(scores), dtype=object)
    for i in range(len(scores)):
        keep = np.arange(len(scores)) != i
        tp, tc, _ = fit_thresholds_fast(scores[keep], y[keep])
        pred[i] = classify(scores[i : i + 1], tp, tc)[0]
    return macro_f1(y, pred)


def load_labelled_scores(labels_path: Path, matches_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Scores and labels of the labelled topics (blank labels skipped), in label-file order."""
    lab = pd.read_csv(labels_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    lab["label"] = lab["label"].str.strip()
    lab = lab[lab["label"] != ""]
    bad = set(lab["label"]) - set(CLASSES)
    if bad:
        raise ValueError(f"unknown labels in {labels_path}: {sorted(bad)}")
    if lab["scert_topic_id"].duplicated().any():
        raise ValueError(f"duplicate topics in {labels_path}")
    matches = pd.read_csv(matches_path)
    score = matches[matches["rank"] == 1].set_index("scert_topic_id")["best_coverage_any"]
    missing = sorted(set(lab["scert_topic_id"]) - set(score.index))
    if missing:
        raise ValueError(f"labelled topics not in {matches_path.name}: {missing[:5]}")
    return score.loc[lab["scert_topic_id"]].to_numpy(), lab["label"].to_numpy()


def calibrate(scores: np.ndarray, y: np.ndarray, model_name: str, n_boot: int, seed: int) -> dict:
    """Thresholds, macro-F1, leave-one-out macro-F1 and bootstrap intervals for one model."""
    tp, tc, f = fit_thresholds_fast(scores, y)
    ci = bootstrap_ci(scores, y, n_boot, seed)
    return {
        "model": model_name,
        "t_partial": round(tp, 4),
        "t_covered": round(tc, 4),
        "macro_f1": round(f, 4),
        "macro_f1_loo": round(loo_macro_f1(scores, y), 4),
        "ci95": {k: [round(v, 4) for v in vs] for k, vs in ci.items()},
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="configs/classification.yaml")
    ap.add_argument("--model", default=None, help="bge, e5 or tfidf (default: from config)")
    ap.add_argument("--dry-run", action="store_true", help="print the result, write nothing")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cal = cfg["calibration"]
    model = args.model or cfg["model"]
    align_cfg = yaml.safe_load(Path(cfg["paths"]["alignment_config"]).read_text(encoding="utf-8"))
    m = align_cfg["models"][model]
    matches_path = Path(align_cfg["paths"]["output_dir"]) / m["slug"] / "topic_matches.csv"

    scores, y = load_labelled_scores(Path(cal["labels"]), matches_path)
    result = calibrate(scores, y, m["name"], cal["bootstrap"], cal["seed"])
    counts = {c: int((y == c).sum()) for c in CLASSES}

    out = {
        "primary": model,
        "gold_file": Path(cal["labels"]).as_posix(),
        "gold_topics": len(y),
        "topic_labels": counts,
        "label_source": cal["label_source"],
        "date": datetime.now(UTC).date().isoformat(),
        "method": (
            f"macro-F1 grid fit on best_coverage_any; bootstrap {cal['bootstrap']} resamples, "
            f"seed {cal['seed']}; script classification/calibrate.py"
        ),
        "thresholds": {model: result},
        "rule": {
            "score": "best_coverage_any (processed/alignment/<model>/topic_matches.csv)",
            "Covered": "score >= t_covered",
            "Partially Covered": "t_partial <= score < t_covered",
            "Not Covered": "score < t_partial",
        },
    }
    ci = result["ci95"]
    print(f"{len(y)} labelled topics: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    print(
        f"{model}: t_partial {result['t_partial']} {ci['t_partial']}, "
        f"t_covered {result['t_covered']} {ci['t_covered']}"
    )
    print(f"macro-F1 {result['macro_f1']} (leave-one-out {result['macro_f1_loo']})")
    if args.dry_run:
        print("dry run: nothing written")
        return
    path = Path(cfg["paths"]["thresholds"])
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}; now run: uv run python -m classification.run")


if __name__ == "__main__":
    main()
