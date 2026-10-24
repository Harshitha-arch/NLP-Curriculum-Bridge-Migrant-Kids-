"""Build the gold-set template that a human labels to evaluate the alignment.

Usage (after ``alignment.run`` for bge, e5 and tfidf):
    uv run python -m alignment.make_gold [--config configs/alignment.yaml]

Sampling (seed ``gold.seed``):
    1. one random SCERT content topic from every unit (20, all 3 computer units included);
    2. the rest stratified by bge ``best_coverage_any`` terciles over all SCERT topics
       (``gold.tercile_draws`` = low / mid / high), drawn from topics not already picked,
       so Not Covered and Partially Covered cases are represented for threshold calibration.

Candidates per topic: union of the top-3 NCERT topics by coverage from bge, e5 and tfidf,
plus the top-3 by bge centroid similarity. Rows are shuffled; scores and model names are
hidden. Labels are left empty: the gold set must be labelled by people.

Outputs (in paths.eval_dir):
    gold_pairs_template.csv  one row per (SCERT topic, NCERT candidate) pair
    gold_topics.csv          the sampled SCERT topics, how each was picked, its tercile
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from alignment.data import by_board, content_only, read_jsonl

TERCILES = ["low", "mid", "high"]
COLUMNS = [
    "pair_id",
    "scert_topic_id",
    "scert_title",
    "scert_text",
    "ncert_topic_id",
    "ncert_title",
    "ncert_text",
    "label",
    "label_2",
    "double_label",
    "notes",
    "cov_tercile",
]


def terciles(scores: pd.Series) -> pd.Series:
    """low / mid / high third of each topic's score."""
    return pd.qcut(scores, 3, labels=TERCILES).astype(str)


def sample_topics(
    units: pd.Series, tercile: pd.Series, draws: list[int], rng: np.random.Generator
) -> pd.DataFrame:
    """One topic per unit (book order), then ``draws`` more per tercile from the rest."""
    picked: list[tuple[str, str]] = []
    for _, ids in units.groupby(units, sort=False):
        picked.append((rng.choice(ids.index.to_numpy()), "unit"))
    taken = {t for t, _ in picked}
    for name, n in zip(TERCILES, draws, strict=True):
        pool = [t for t in tercile.index if tercile[t] == name and t not in taken]
        if n > len(pool):
            raise ValueError(f"only {len(pool)} {name}-tercile topics left, need {n}")
        picked += [(t, f"tercile_{name}") for t in rng.choice(pool, n, replace=False)]
    out = pd.DataFrame(picked, columns=["scert_topic_id", "picked_by"])
    out["unit"] = out["scert_topic_id"].map(units)
    out["cov_tercile"] = out["scert_topic_id"].map(tercile)
    return out


def candidates(
    topic_id: str, rankings: list[pd.DataFrame], centroid: pd.DataFrame, n: int
) -> list[str]:
    """Union of the top-``n`` NCERT topics in every ranking, in first-seen order."""
    found: list[str] = []
    for matches in rankings:
        rows = matches[(matches["scert_topic_id"] == topic_id) & (matches["rank"] <= n)]
        found += rows.sort_values("rank")["ncert_topic_id"].tolist()
    found += centroid.loc[topic_id].sort_values(ascending=False, kind="stable").index[:n].tolist()
    return list(dict.fromkeys(found))


def main() -> None:
    """Sample topics, pool candidates and write the unlabelled template."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/alignment.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    paths, gold, models = cfg["paths"], cfg["gold"], cfg["models"]
    align_dir, eval_dir = Path(paths["output_dir"]), Path(paths["eval_dir"])
    rng = np.random.default_rng(gold["seed"])

    topics = content_only(read_jsonl(paths["topics"]))
    by_id = {t["topic_id"]: t for t in topics}
    scert = by_board(topics, cfg["boards"]["destination"])
    ncert_ids = [t["topic_id"] for t in by_board(topics, cfg["boards"]["origin"])]

    rankings = {
        key: pd.read_csv(align_dir / models[key]["slug"] / "topic_matches.csv")
        for key in ("bge", "e5", "tfidf")
    }
    bge = rankings["bge"]
    best_any = bge[bge["rank"] == 1].set_index("scert_topic_id")["best_coverage_any"]
    best_any = best_any.loc[[t["topic_id"] for t in scert]]
    centroid = pd.read_csv(
        align_dir / models["bge"]["slug"] / "topic_centroid_sim.csv", index_col=0
    ).loc[best_any.index, ncert_ids]

    units = pd.Series({t["topic_id"]: t["chapter"] for t in scert})
    missing = set(gold["computer_units"]) - set(units)
    if missing:
        raise ValueError(f"computer units not found among SCERT units: {sorted(missing)}")
    sample = sample_topics(units, terciles(best_any), gold["tercile_draws"], rng)

    n, cut = gold["candidates_per_model"], gold["max_text_chars"]
    pairs = []
    for row in sample.itertuples():
        s = by_id[row.scert_topic_id]
        found = candidates(row.scert_topic_id, list(rankings.values()), centroid, n)
        for o_id in found:
            o = by_id[o_id]
            pairs.append(
                {
                    "scert_topic_id": s["topic_id"],
                    "scert_title": f"{s['chapter']}: {s['topic_title']}",
                    "scert_text": s["text"][:cut],
                    "ncert_topic_id": o_id,
                    "ncert_title": f"{o['chapter']}: {o['topic_title']}",
                    "ncert_text": o["text"][:cut],
                    "label": "",
                    "label_2": "",
                    "double_label": "no",
                    "notes": "",
                    "cov_tercile": row.cov_tercile,
                }
            )
        sample.loc[row.Index, "n_candidates"] = len(found)

    template = pd.DataFrame(pairs).sample(frac=1, random_state=rng).reset_index(drop=True)
    template.insert(0, "pair_id", [f"P{i:03d}" for i in range(1, len(template) + 1)])
    double = rng.choice(len(template), gold["double_label"], replace=False)
    template.loc[double, "double_label"] = "yes"

    eval_dir.mkdir(parents=True, exist_ok=True)
    # utf-8-sig so that Excel shows the text correctly; pandas reads it back unchanged.
    out = eval_dir / "gold_pairs_template.csv"
    template[COLUMNS].to_csv(out, index=False, encoding="utf-8-sig")
    sample["n_candidates"] = sample["n_candidates"].astype(int)
    sample.to_csv(eval_dir / "gold_topics.csv", index=False, encoding="utf-8")
    counts = sample["picked_by"].value_counts().to_dict()
    print(
        f"{len(sample)} SCERT topics ({counts}) from {sample['unit'].nunique()} units, "
        f"{len(template)} pairs, {gold['double_label']} marked for a second annotator "
        f"-> {out}"
    )


if __name__ == "__main__":
    main()
