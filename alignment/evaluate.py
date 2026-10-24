"""Evaluate the alignment models against the labelled gold set and calibrate thresholds.

Usage (after ``alignment.run`` for every model and a labelled ``eval/gold_pairs.csv``):
    uv run python -m alignment.evaluate [--config configs/alignment.yaml]

Retrieval: relevant = ``same`` or ``partial``. Over SCERT topics with at least one relevant
pair: Hit@1, Hit@3 and MRR@3 of each model's ranking. Unjudged candidates are not relevant.

Topic level: Covered if any pair is ``same``; Partially Covered if the best is ``partial`` or
``prerequisite``; Not Covered if all pairs are ``none``. Thresholds ``t_partial < t_covered`` on
``best_coverage_any`` maximise macro-F1 against these labels, with bootstrap 95% CIs.

Outputs:
    eval/results.md                      models x metrics, thresholds, agreement
    eval/score_by_label_<model>.png      best_coverage_any by topic-level gold label
    processed/alignment/thresholds.json  thresholds for Phase 4
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import cohen_kappa_score, f1_score

RELEVANT = {"same", "partial"}
LABELS = ("same", "partial", "prerequisite", "none")
CLASSES = ("Not Covered", "Partially Covered", "Covered")
MODELS = ("bge", "e5", "tfidf")


def topic_label(labels: list[str]) -> str:
    """Topic-level gold label from the labels of all its candidate pairs."""
    if "same" in labels:
        return "Covered"
    if "partial" in labels or "prerequisite" in labels:
        return "Partially Covered"
    return "Not Covered"


def retrieval_metrics(ranked: dict[str, list[str]], relevant: dict[str, set[str]]) -> dict:
    """Hit@1, Hit@3 and MRR@3 over topics with at least one relevant origin topic."""
    topics = [t for t, rel in relevant.items() if rel]
    hit1 = hit3 = rr = 0.0
    for t in topics:
        top = ranked.get(t, [])[:3]
        ranks = [i for i, o in enumerate(top, start=1) if o in relevant[t]]
        hit1 += bool(ranks and ranks[0] == 1)
        hit3 += bool(ranks)
        rr += 1 / ranks[0] if ranks else 0.0
    n = len(topics)
    return {"n_topics": n, "hit@1": hit1 / n, "hit@3": hit3 / n, "mrr@3": rr / n}


def classify(scores: np.ndarray, t_partial: float, t_covered: float) -> np.ndarray:
    """Covered at or above ``t_covered``, Partially Covered at or above ``t_partial``."""
    out = np.full(len(scores), CLASSES[0], dtype=object)
    out[scores >= t_partial] = CLASSES[1]
    out[scores >= t_covered] = CLASSES[2]
    return out


def macro_f1(y: np.ndarray, pred: np.ndarray) -> float:
    """Macro-F1 over the three coverage classes (absent classes score 0)."""
    return float(f1_score(y, pred, labels=list(CLASSES), average="macro", zero_division=0))


def fit_thresholds(scores: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """``t_partial < t_covered`` maximising macro-F1. Candidates are the midpoints between
    consecutive distinct scores plus one value below and above all scores; ties keep the
    first pair in ascending order, so the result is deterministic."""
    s = np.unique(scores)
    cuts = np.concatenate([[s[0] - 1e-3], (s[:-1] + s[1:]) / 2, [s[-1] + 1e-3]])
    best = (-1.0, cuts[0], cuts[-1])
    for i, tp in enumerate(cuts):
        for tc in cuts[i + 1 :]:
            f = macro_f1(y, classify(scores, tp, tc))
            if f > best[0] + 1e-12:
                best = (f, tp, tc)
    return float(best[1]), float(best[2]), best[0]


def bootstrap(scores: np.ndarray, y: np.ndarray, n: int, seed: int) -> dict:
    """95% percentile CIs for both thresholds and macro-F1, refitting on each resample."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        idx = rng.integers(0, len(scores), len(scores))
        rows.append(fit_thresholds(scores[idx], y[idx]))
    arr = np.array(rows)
    ci = np.percentile(arr, [2.5, 97.5], axis=0)
    return {
        k: [float(ci[0, i]), float(ci[1, i])]
        for i, k in enumerate(("t_partial", "t_covered", "macro_f1"))
    }


def loo_macro_f1(scores: np.ndarray, y: np.ndarray) -> float:
    """Leave-one-topic-out macro-F1: thresholds fit without each topic, then applied to it."""
    pred = np.empty(len(scores), dtype=object)
    for i in range(len(scores)):
        keep = np.arange(len(scores)) != i
        tp, tc, _ = fit_thresholds(scores[keep], y[keep])
        pred[i] = classify(scores[i : i + 1], tp, tc)[0]
    return macro_f1(y, pred)


def plot_scores(
    scores: pd.Series, y: pd.Series, thresholds: tuple[float, float], path: Path, title: str
):
    """Strip + box plot of best_coverage_any per topic-level gold label."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    data = [scores[y == c].to_numpy() for c in CLASSES]
    ax.boxplot(data, widths=0.5, showfliers=False)
    rng = np.random.default_rng(0)
    for i, d in enumerate(data, start=1):
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(d)), d, s=18, alpha=0.75, color="#2a6f97")
    for t, style in zip(thresholds, ("--", "-"), strict=True):
        ax.axhline(t, color="#c1121f", linestyle=style, linewidth=1)
    ax.set_xticks(range(1, 4), [f"{c}\n(n={len(d)})" for c, d in zip(CLASSES, data, strict=True)])
    ax.set_ylabel("best_coverage_any")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Load the labelled gold set and every model's outputs, then write the results."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/alignment.yaml")
    parser.add_argument(
        "--gold", default=None, help="labelled pairs (default: <eval_dir>/gold_pairs.csv)"
    )
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    paths, models = cfg["paths"], cfg["models"]
    align_dir, eval_dir = Path(paths["output_dir"]), Path(paths["eval_dir"])
    gold_path = Path(args.gold) if args.gold else eval_dir / "gold_pairs.csv"
    seed = cfg["gold"]["seed"]

    gold = pd.read_csv(gold_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    gold["label"] = gold["label"].str.strip().str.lower()
    bad = set(gold["label"]) - set(LABELS)
    if bad:
        raise ValueError(f"unknown or missing labels in {gold_path}: {sorted(bad)}")
    relevant = {
        t: set(g.loc[g["label"].isin(RELEVANT), "ncert_topic_id"])
        for t, g in gold.groupby("scert_topic_id", sort=False)
    }
    y_topic = gold.groupby("scert_topic_id", sort=False)["label"].agg(
        lambda s: topic_label(list(s))
    )

    # Retrieval quality of each ranking.
    retrieval = {}
    best_any = {}
    for key in MODELS:
        matches = pd.read_csv(align_dir / models[key]["slug"] / "topic_matches.csv")
        ranked = {
            t: g.sort_values("rank")["ncert_topic_id"].tolist()
            for t, g in matches.groupby("scert_topic_id")
        }
        retrieval[key] = retrieval_metrics(ranked, relevant)
        best_any[key] = matches[matches["rank"] == 1].set_index("scert_topic_id")[
            "best_coverage_any"
        ]
    centroid = pd.read_csv(
        align_dir / models["bge"]["slug"] / "topic_centroid_sim.csv", index_col=0
    )
    ncert = [c for c in centroid.columns if c.startswith(cfg["boards"]["origin"])]
    ranked = {
        t: centroid.loc[t, ncert].sort_values(ascending=False, kind="stable").index.tolist()
        for t in relevant
    }
    retrieval["bge (centroid)"] = retrieval_metrics(ranked, relevant)

    # Thresholds on best_coverage_any for the dense models.
    thresholds = {}
    for key in ("bge", "e5", "tfidf"):
        s = best_any[key].loc[y_topic.index].to_numpy()
        y = y_topic.to_numpy()
        tp, tc, f = fit_thresholds(s, y)
        thresholds[key] = {
            "model": models[key]["name"],
            "t_partial": round(tp, 4),
            "t_covered": round(tc, 4),
            "macro_f1": round(f, 4),
            "macro_f1_loo": round(loo_macro_f1(s, y), 4),
            "ci95": {
                k: [round(v, 4) for v in vs]
                for k, vs in bootstrap(s, y, args.bootstrap, seed).items()
            },
        }
        plot_scores(
            best_any[key].loc[y_topic.index],
            y_topic,
            (tp, tc),
            eval_dir / f"score_by_label_{key}.png",
            f"{models[key]['name']}: best_coverage_any by gold label (lines: thresholds)",
        )

    double = gold[gold["label_2"].str.strip() != ""]
    kappa = (
        float(cohen_kappa_score(double["label"], double["label_2"].str.strip().str.lower()))
        if len(double)
        else None
    )

    meta = {
        "gold_file": gold_path.as_posix(),
        "gold_pairs": len(gold),
        "gold_topics": len(y_topic),
        "topic_labels": y_topic.value_counts().to_dict(),
        "label_source": cfg["gold"].get("label_source", "human"),
        "date": datetime.now(UTC).date().isoformat(),
    }
    out = {
        "primary": "bge",
        **meta,
        "thresholds": thresholds,
        "rule": {
            "score": "best_coverage_any (processed/alignment/<model>/topic_matches.csv)",
            "Covered": "score >= t_covered",
            "Partially Covered": "t_partial <= score < t_covered",
            "Not Covered": "score < t_partial",
        },
    }
    (align_dir / "thresholds.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    (eval_dir / "results.md").write_text(
        results_markdown(retrieval, thresholds, meta, gold, kappa, len(double)), encoding="utf-8"
    )
    print((eval_dir / "results.md").read_text(encoding="utf-8"))


def results_markdown(retrieval, thresholds, meta, gold, kappa, n_double) -> str:
    """The results report."""
    lines = ["# Alignment Evaluation Results", ""]
    if meta["label_source"] != "human":
        lines += [
            f"> **Labels: {meta['label_source']}.** These numbers are provisional until the gold "
            "set is labelled by people (see `eval/LABELLING_GUIDE.md`).",
            "",
        ]
    lines += [
        f"Gold set: `{meta['gold_file']}`, {meta['gold_pairs']} pairs over {meta['gold_topics']} "
        f"SCERT topics ({meta['date']}). Pair labels: "
        + ", ".join(f"{k} {v}" for k, v in gold["label"].value_counts().items())
        + ". Topic labels: "
        + ", ".join(f"{k} {v}" for k, v in meta["topic_labels"].items())
        + ".",
        "",
        "## Retrieval (relevant = same or partial)",
        "",
        "| Ranking | Topics | Hit@1 | Hit@3 | MRR@3 | Hit@3 ≥ 0.80 |",
        "|---|---|---|---|---|---|",
    ]
    for name, m in retrieval.items():
        ok = "yes" if m["hit@3"] >= 0.80 else "no"
        lines.append(
            f"| {name} | {m['n_topics']} | {m['hit@1']:.3f} | {m['hit@3']:.3f} "
            f"| {m['mrr@3']:.3f} | {ok} |"
        )
    lines += [
        "",
        "Rankings are by topic coverage `C[d, o]`, except `bge (centroid)`. Unjudged candidates "
        "count as not relevant.",
        "",
        "## Thresholds on `best_coverage_any` (for Phase 4)",
        "",
        "| Model | t_partial (95% CI) | t_covered (95% CI) | macro-F1 (95% CI) "
        "| macro-F1 leave-one-out |",
        "|---|---|---|---|---|",
    ]
    for key, t in thresholds.items():
        ci = t["ci95"]
        lines.append(
            f"| {key} | {t['t_partial']:.3f} ({ci['t_partial'][0]:.3f}–{ci['t_partial'][1]:.3f}) "
            f"| {t['t_covered']:.3f} ({ci['t_covered'][0]:.3f}–{ci['t_covered'][1]:.3f}) "
            f"| {t['macro_f1']:.3f} ({ci['macro_f1'][0]:.3f}–{ci['macro_f1'][1]:.3f}) "
            f"| {t['macro_f1_loo']:.3f} |"
        )
    lines += [
        "",
        "Rule: Covered if score ≥ t_covered; Partially Covered if t_partial ≤ score < t_covered; "
        "otherwise Not Covered. macro-F1 is in-sample (thresholds fit on the same topics); the "
        "leave-one-out column is the honest estimate. CIs: 1,000 bootstrap resamples of topics, "
        "thresholds refit on each. Plots: `eval/score_by_label_<model>.png`.",
        "",
        "## Annotator agreement",
        "",
        f"Cohen's kappa: {kappa:.3f} on {n_double} double-labelled pairs."
        if kappa is not None
        else "Cohen's kappa: not available (no pairs have `label_2`).",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
