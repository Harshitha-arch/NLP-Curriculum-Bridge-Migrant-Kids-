# Gap Classification: Documentation

Author: Harshitha
Stage: Phase 4, REQ-04 (after embedding and alignment, before bridge content)
Code: `classification/`  ·  Settings: `configs/classification.yaml`  ·  Tests: `tests/test_classify.py`

## 1. What this stage answers

For each of the 132 Tamil Nadu SCERT Class 6 Science topics: **is it Covered, Partially
Covered or Not Covered by NCERT "Curiosity"?** The result is the curriculum gap matrix, the
main deliverable of the project, and the input to bridge content (Phase 5).

## 2. How it works

Input is the alignment output, not the raw text:

- `processed/alignment/<model>/topic_matches.csv`: the top 5 NCERT topics per SCERT topic, with
  `best_coverage_any` and the best evidence chunk pair (see `docs/ALIGNMENT.md` §2, §4).
- `processed/alignment/thresholds.json`: `t_partial` and `t_covered` per model, with 95%
  intervals. The current file was fitted on researcher-reviewed labels (§7); the original AI-labelled
  fit (`docs/ALIGNMENT.md` §6) is kept as `thresholds_ai_labelled.json`.

```
Covered            score >= t_covered
Partially Covered  t_partial <= score < t_covered
Not Covered        score < t_partial          (score = best_coverage_any)
```

Two columns say how far the label can be trusted:

- `confidence`: `uncertain` if the score lies inside the 95% interval of either threshold, so a
  refit of the thresholds on resampled gold topics could change the label; otherwise `stable`.
- `margin`: distance from the score to the nearest threshold.
- `provisional`: `True` unless the labels the thresholds were fitted on were all labelled by people
  (`label_source: human`). It is still `True` today: most labels were AI-assisted (§7).

The label comes from the score alone. The NCERT match and evidence chunks are there so a person
can check it; they do not change the label.

## 3. How to run

```
uv run python -m classification.run                # primary model (bge)
uv run python -m classification.run --model e5     # also: tfidf
uv run pytest tests/test_classify.py
```

To recalibrate, put new labels in `eval/human_labels.csv`, refit the two cut-offs, update
`processed/alignment/thresholds.json` and run the classification again. Only the thresholds
change; no code does.

## 4. Output files (`processed/gap/`)

| File | Content |
|---|---|
| `gap_matrix.csv` | one row per SCERT topic: unit, title, `score`, `label`, `confidence`, `margin`, `provisional`, best NCERT topic and top matches, evidence text from both books |
| `unit_summary.csv` | label counts, `pct_not_covered` and mean score per SCERT unit, in book order |
| `review_queue.csv` | topics with `confidence == uncertain`, closest to a threshold first |
| `run_meta.json` | model, thresholds, label source, label counts, timestamp |

Phase 5 takes the rows of `gap_matrix.csv` whose `label` is not `Covered`.

## 5. Results with the current thresholds

bge, thresholds 0.640 / 0.735, 132 SCERT topics: 26 Covered, 64 Partially Covered, 42 Not
Covered. 72 labels are stable and 60 are uncertain (44 of them Partially Covered). All 12
computer-unit topics, which have no NCERT counterpart, are Not Covered.

These counts are a calibration result, not a finding about the curricula. With the original AI-fitted
thresholds (0.692 / 0.770) the split was 14 / 37 / 81, and the 75 uncertain topics became 60 after
recalibration. The split also depends on the model: e5 and TF-IDF gave very different splits
under the original thresholds, and the three models agreed on only 73 of 132 topics. Report the
Not Covered count as a range, not a single figure.

## 6. Known limitations

1. **Provisional thresholds.** They rest on 131 researcher-reviewed labels, of which only 29 were
   hand-labelled after reading the text; the rest were AI-assisted. On those 29 topics the
   pipeline matches the researcher on 59% and the AI-assisted labels on 66%. Even the best cut-offs
   agree with the researcher on about 62-73% of topics.
2. **Wide intervals.** 60 of 132 topics lie inside a threshold interval (`t_partial` 0.63-0.69,
   `t_covered` 0.72-0.76). Treat these as needing review.
3. **One score.** `best_coverage_any` ignores NCERT topic boundaries and can reward shared
   vocabulary (`docs/ALIGNMENT.md` §7). Read the evidence text for topics that matter.
4. **Prerequisite gaps.** The gold set treats a `prerequisite` pair as Partially Covered, whereas
   the proposal calls a prerequisite gap Not Covered. This stage follows the gold set.
5. **One grade.** "Taught in a different grade" cannot be detected with Class 6 alone.
6. **Many-to-one matching.** Each SCERT chunk takes its single closest NCERT chunk, and the same
   NCERT chunk is often chosen by many SCERT chunks (112 of 251 NCERT content chunks are ever
   chosen; the 10 most chosen account for 75 of 294 SCERT chunks). A broad NCERT chunk can therefore
   raise the score of many SCERT topics, so coverage may be overstated.

7. **Only the single best match is used.** A topic scored Not Covered by its best NCERT chunk
   might be covered by its second or third best, which this stage does not check. Of the 251
   NCERT content chunks, 139 were never the best match for any SCERT chunk, and whether any of
   them is a better match for a Not Covered topic was not tested.
8. **Matching runs one way (SCERT to NCERT).** It answers "does NCERT cover what SCERT teaches?",
   not the reverse. `topic_coverage_reverse.csv` already exists for a reverse analysis.

Limitations 6-8 were identified while reviewing the labels. Testing a one-to-one assignment and
top-3 matches, and a reverse-coverage analysis, are left as future work.

## 7. Calibration history

| Version | Labels used | t_partial | t_covered | Not Covered / Partial / Covered | Uncertain |
|---|---|---|---|---|---|
| v1 (2026-10-24) | 40 SCERT topics (252 pairs), AI-labelled, stricter reading, "prerequisite" = partial | 0.6923 | 0.7696 | 81 / 37 / 14 | 75 |
| v2 (2026-10-26) | 131 topics, researcher's rubric: conceptual similarity = partial, near-identical = covered; 29 hand-labelled after reading, rest AI-assisted and reviewed | 0.640 | 0.735 | 42 / 64 / 26 | 60 |

Rubric used for v2: **Covered** when the two chunks say almost the same thing; **Partially
Covered** when they are conceptually similar but one is broader or leaves out key points; **Not
Covered** when they are about unrelated topics. Labels are in `eval/human_labels.csv` with the
source of each. Hand-labelling more topics is expected to narrow the threshold intervals.
