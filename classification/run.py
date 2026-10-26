"""Build the curriculum gap matrix from the alignment outputs.

Usage:
    uv run python -m classification.run [--model bge|e5|tfidf]
                                        [--config configs/classification.yaml]

Outputs (in paths.output_dir, default processed/gap/):
    gap_matrix.csv     one row per SCERT topic: score, label, confidence, NCERT evidence
    unit_summary.csv   label counts and mean score per SCERT unit
    review_queue.csv   topics whose label is uncertain, closest to a threshold first
    run_meta.json      model, thresholds, label source, label counts, timestamp
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml

from classification.classify import (
    LABELS,
    build_gap_matrix,
    load_thresholds,
    review_queue,
    unit_summary,
)

FLOAT = "%.4f"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="configs/classification.yaml")
    ap.add_argument("--model", default=None, help="bge, e5 or tfidf (default: from config)")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    model = args.model or cfg["model"]
    align_cfg = yaml.safe_load(Path(cfg["paths"]["alignment_config"]).read_text(encoding="utf-8"))
    slug = align_cfg["models"][model]["slug"]
    matches_path = Path(align_cfg["paths"]["output_dir"]) / slug / "topic_matches.csv"
    out_dir = Path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    t = load_thresholds(cfg["paths"]["thresholds"], model)
    gm = build_gap_matrix(pd.read_csv(matches_path), t, top_n=cfg["top_n_matches"])
    summary, queue = unit_summary(gm), review_queue(gm)

    gm.to_csv(out_dir / "gap_matrix.csv", index=False, float_format=FLOAT)
    summary.to_csv(out_dir / "unit_summary.csv", index=False)
    queue.to_csv(out_dir / "review_queue.csv", index=False, float_format=FLOAT)

    counts = {label: int((gm["label"] == label).sum()) for label in LABELS}
    meta = {
        "model": model,
        "model_name": t.model_name,
        "t_partial": t.t_partial,
        "t_covered": t.t_covered,
        "label_source": t.label_source,
        "provisional": t.provisional,
        "thresholds_date": t.date,
        "topics": len(gm),
        "label_counts": counts,
        "uncertain": len(queue),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(f"{len(gm)} SCERT topics, model {model} ({t.model_name})")
    print(f"thresholds: partial >= {t.t_partial:.3f}, covered >= {t.t_covered:.3f}")
    for label in LABELS:
        print(f"  {label:<18} {counts[label]:>4}")
    print(f"  uncertain (inside threshold CI): {len(queue)}")
    if t.provisional:
        print(f"WARNING: provisional labels ({t.label_source}); re-run after human labelling.")
    print(f"wrote {out_dir}/")


if __name__ == "__main__":
    main()
