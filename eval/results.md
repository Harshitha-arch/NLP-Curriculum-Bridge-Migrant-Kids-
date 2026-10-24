# Alignment Evaluation Results

> **Labels: AI-labelled (LLM), not human-reviewed.** These numbers are provisional until the gold set is labelled by people (see `eval/LABELLING_GUIDE.md`).

Gold set: `eval/gold_pairs.csv`, 252 pairs over 40 SCERT topics (2026-10-24). Pair labels: none 216, partial 21, prerequisite 13, same 2. Topic labels: Not Covered 20, Partially Covered 18, Covered 2.

## Retrieval (relevant = same or partial)

| Ranking | Topics | Hit@1 | Hit@3 | MRR@3 | Hit@3 ≥ 0.80 |
|---|---|---|---|---|---|
| bge | 15 | 0.733 | 1.000 | 0.856 | yes |
| e5 | 15 | 0.467 | 0.867 | 0.633 | yes |
| tfidf | 15 | 0.733 | 0.933 | 0.822 | yes |
| bge (centroid) | 15 | 0.600 | 1.000 | 0.800 | yes |

Rankings are by topic coverage `C[d, o]`, except `bge (centroid)`. Unjudged candidates count as not relevant.

## Thresholds on `best_coverage_any` (for Phase 4)

| Model | t_partial (95% CI) | t_covered (95% CI) | macro-F1 (95% CI) | macro-F1 leave-one-out |
|---|---|---|---|---|
| bge | 0.692 (0.609–0.694) | 0.770 (0.743–0.812) | 0.890 (0.544–0.964) | 0.737 |
| e5 | 0.887 (0.881–0.905) | 0.939 (0.911–0.939) | 0.742 (0.504–0.923) | 0.614 |
| tfidf | 0.110 (0.093–0.171) | 0.259 (0.241–0.439) | 0.851 (0.567–0.982) | 0.707 |

Rule: Covered if score ≥ t_covered; Partially Covered if t_partial ≤ score < t_covered; otherwise Not Covered. macro-F1 is in-sample (thresholds fit on the same topics); the leave-one-out column is the honest estimate. CIs: 1,000 bootstrap resamples of topics, thresholds refit on each. Plots: `eval/score_by_label_<model>.png`.

## Annotator agreement

Cohen's kappa: not available (no pairs have `label_2`).
