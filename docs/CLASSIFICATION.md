# Gap Classification: Documentation

Author: Harshitha
Stage: Phase 4, REQ-04 (after embedding and alignment, before bridge content)
Code: `classification/` (`classify.py`, `run.py`, `calibrate.py`)  ·  Settings: `configs/classification.yaml`  ·  Tests: `tests/test_classify.py`, `tests/test_calibrate.py`

## 1. What this stage answers

For each of the 132 Tamil Nadu SCERT Class 6 Science topics: **is it Covered, Partially
Covered or Not Covered by NCERT "Curiosity"?** The result is the curriculum gap matrix, the
main deliverable of the project, and the input to bridge content (Phase 5).

## 2. How it works

Input is the alignment output, not the raw text:

- `processed/alignment/<model>/topic_matches.csv`: the top 5 NCERT topics per SCERT topic, with
  `best_coverage_any` and the best evidence chunk pair (see `docs/ALIGNMENT.md` §2, §4).
- `processed/alignment/thresholds.json`: `t_partial` and `t_covered` per model, with 95%
  intervals. It is written by `classification/calibrate.py` from the researcher-reviewed labels in
  `eval/human_labels.csv` (§3, §7); the original AI-labelled fit (`docs/ALIGNMENT.md` §6) is kept
  as `thresholds_ai_labelled.json`.

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
uv run python -m classification.calibrate          # refit thresholds from eval/human_labels.csv
uv run python -m classification.run                # label every topic (primary model: bge)
uv run python -m classification.run --model e5     # also: tfidf
uv run pytest tests/test_classify.py tests/test_calibrate.py
```

**Recalibrating.** Put new labels in `eval/human_labels.csv` (one label per SCERT topic; blank
rows are skipped), then run `classification.calibrate` followed by `classification.run`. Only
the labels change; no code does. `calibrate` finds the two cut-offs `t_partial < t_covered`
that maximise macro-F1 on `best_coverage_any`, with 95% bootstrap intervals (1,000 resamples,
seed 42, both set in `configs/classification.yaml`) and a leave-one-topic-out macro-F1. The fit is
deterministic: the same labels and scores give a byte-identical `thresholds.json` (apart from its
date) and `gap_matrix.csv`. `--dry-run` prints the result without writing. `calibrate` writes
only the model it fitted (bge by default), so `thresholds.json` has no e5 or TF-IDF cut-offs and
`classification.run --model e5` fails until you fit that model, which replaces the bge entry.

## 4. Output files (`processed/gap/`)

| File | Content |
|---|---|
| `gap_matrix.csv` | one row per SCERT topic: unit, title, `score`, `label`, `confidence`, `margin`, `provisional`, best NCERT topic and top matches, evidence text from both books |
| `unit_summary.csv` | label counts, `pct_not_covered` and mean score per SCERT unit, in book order |
| `review_queue.csv` | topics with `confidence == uncertain`, closest to a threshold first |
| `run_meta.json` | model, thresholds, label source, label counts, timestamp |

Phase 5 takes the rows of `gap_matrix.csv` whose `label` is not `Covered`.

## 5. Results with the current thresholds

bge, thresholds 0.6619 / 0.7267, 132 SCERT topics: 28 Covered, 54 Partially Covered, 50 Not
Covered. 69 labels are stable and 63 are uncertain (34 of them Partially Covered, 16 Not Covered,
13 Covered). All 12 computer-unit topics, which have no NCERT counterpart, are Not Covered.
Against the 131 researcher-reviewed labels the thresholds agree on 96 topics (73%, macro-F1
0.739). That is measured on the same labels they were fitted on; the leave-one-topic-out
macro-F1 is 0.638, which is the better guide to how they would do on new topics.

These counts are a calibration result, not a finding about the curricula. With the original AI-fitted
thresholds (0.692 / 0.770) the split was 14 / 37 / 81, and the 75 uncertain topics became 63 after
recalibration. The split also depends on the model: e5 and TF-IDF gave very different splits
under the original thresholds, and the three models agreed on only 73 of 132 topics. Report the
Not Covered count as a range, not a single figure.

## 6. Known limitations

1. **Provisional thresholds.** They rest on 131 researcher-reviewed labels. 49 were labelled by
   the researcher after reading the text (19 read in order, rows 1-20, and 30 in a blind
   spot-check, one of which, topic 25, rests on 1 of its 6 chunk pairs); the other 82 were
   AI-assisted and then reviewed. Only the blind spot-check is independent of the AI labels: on its
   29 complete topics the pipeline (v1 thresholds) matches the researcher on 59% and the
   AI-assisted labels on 66%. The fitted cut-offs agree with the researcher on 73% of the 131
   labelled topics, but 29 independent labels cannot show that they beat the first set, and the
   leave-one-out macro-F1 is only 0.64.
2. **Wide intervals.** 63 of 132 topics lie inside a threshold interval (`t_partial` 0.63-0.69,
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
| v2, first report (2026-10-26) | 131 topics, researcher's rubric (below); 49 labelled after reading (19 in order, 30 blind spot-check), 82 AI-assisted and reviewed | 0.640 | 0.735 | 42 / 64 / 26 | 60 |
| v2, reproducible refit (2026-10-28) | the same 131 labels, fitted by `classification/calibrate.py` | 0.6619 | 0.7267 | 50 / 54 / 28 | 63 |

**Why there are two v2 rows.** The 0.640 / 0.735 values first reported for v2 were produced outside
the repository, and no code here could regenerate them. Fitted on the committed
`eval/human_labels.csv` they give a macro-F1 of 0.726, below the 0.739 that the best cut-offs
reach, and the label counts recorded in the old `thresholds.json` (53 / 52 / 26) did not match
the file (50 / 53 / 28). On 28 Oct the fit was moved into `classification/calibrate.py` and
the thresholds regenerated from the committed labels; those are the values used now. Ten of the
132 topics changed label (8 from Partially Covered to Not Covered, 2 from Partially Covered to
Covered); all ten have scores between 0.640 and 0.735, close to a threshold. Any figure quoted
from the first v2 report (42 / 64 / 26, 60 uncertain) is superseded.

Rubric used for v2: **Covered** when the two chunks say almost the same thing; **Partially
Covered** when they are conceptually similar but one is broader or leaves out key points; **Not
Covered** when they are about unrelated topics. Labels are in `eval/human_labels.csv` with the
source of each. Hand-labelling more topics is expected to narrow the threshold intervals.

## 8. How the thresholds are calculated

`classification/calibrate.py` does this, in order:

1. **Read the labels.** `eval/human_labels.csv` has one row per SCERT topic: `scert_topic_id`,
   `scert_topic_title`, `label`, `label_source`. Blank labels are skipped (131 of 132 topics are
   labelled; "Types of Motion" is blank). Unknown label names and duplicate topics stop the run
   with an error.
2. **Look up each labelled topic's score.** The score is `best_coverage_any` from the rank-1 row
   of `processed/alignment/bge/topic_matches.csv`: the mean, over the topic's chunks, of each
   chunk's best cosine similarity to any NCERT chunk.
3. **Try every possible pair of cut-offs.** Sort the distinct scores. Candidate cut-offs are the
   midpoints between neighbouring scores, plus one value just below the lowest and one just above
   the highest. For every pair `t_partial < t_covered`, label each topic by the rule in §2.
4. **Score each pair with macro-F1.** For each class (Not Covered, Partially Covered, Covered),
   compute F1 = 2 x hits / (predicted + actual), then average the three. Averaging by class
   stops the large classes from outweighing the small one. The pair with the highest macro-F1
   wins; if two pairs tie, the first in ascending order wins, so the result never changes.
5. **Measure how stable the cut-offs are.** Resample the 131 labelled topics with replacement
   1,000 times (seed 42), refit on each resample, and take the 2.5th and 97.5th percentiles of
   each cut-off. These are the 95% intervals. A topic whose score lies inside either interval
   is flagged `uncertain` in `gap_matrix.csv`.
6. **Estimate how it would do on new topics.** Leave-one-out: remove one topic, refit, label the
   removed topic, repeat for all 131, then compute macro-F1 over those 131 predictions (0.638).
   This is lower than the fitted 0.739, which is measured on the same labels used for fitting.
7. **Write `thresholds.json`.** `classification.run` reads it and labels all 132 topics.

`fit_thresholds_fast` gives exactly the same answer as the earlier loop in
`alignment.evaluate.fit_thresholds` (tests prove it on random data) but counts prefix sums
instead of reclassifying for every pair, so the 1,000 resamples take seconds, not hours.

## 9. Code guide

| File | What it does |
|---|---|
| `classification/classify.py` | Pure functions, no file reading except `load_thresholds`. `label_score` applies the rule in §2; `label_confidence` returns `uncertain` inside either interval; `margin` is the distance to the nearest cut-off; `build_gap_matrix` joins labels to the best NCERT topic and evidence text; `unit_summary` and `review_queue` build the other two tables. `Thresholds.provisional` is `True` unless the label source is `human`. |
| `classification/calibrate.py` | Section 8. Entry point `main`; helpers `fit_thresholds_fast`, `bootstrap_ci`, `loo_macro_f1`, `load_labelled_scores`, `calibrate`. |
| `classification/run.py` | Reads `thresholds.json` and `topic_matches.csv`, calls `build_gap_matrix`, writes the four files in `processed/gap/`. Does no fitting. |
| `configs/classification.yaml` | Paths, model, number of matches listed, and the `calibration` block (labels file, label-source text, bootstrap size, seed). |
| `eval/human_labels.csv` | The 131 labels and where each came from. The only input a person edits. |
| `tests/test_classify.py` | The labelling rule, confidence, margin and table builders. |
| `tests/test_calibrate.py` | Fast fit equals the reference fit; F1 reported is the F1 of the thresholds reported; loader errors; `calibrate` is deterministic and writes a file `classify.load_thresholds` can read; and two checks that the committed `thresholds.json` and `gap_matrix.csv` match the committed labels. |

**Data flow:** `human_labels.csv` + `topic_matches.csv` -> `calibrate` -> `thresholds.json` ->
`run` (with `topic_matches.csv`) -> `processed/gap/*`.

## 10. Worked example

Thresholds 0.6619 / 0.7267, intervals [0.6266, 0.6908] and [0.7203, 0.7564]:

| SCERT topic | Score | Label | Confidence | Why |
|---|---|---|---|---|
| Adatptation in Animals (typo is in the source) | 0.7613 | Covered | stable | above 0.7267 and above the whole upper interval; best match "Plants and Animals in Different Surroundings" |
| Sources of heat | 0.7106 | Partially Covered | stable | between the cut-offs and outside both intervals; best match "Energy from the Sun" |
| Linear and Cubical Expansion | 0.614 | Not Covered | stable | below 0.6619 and below the whole lower interval; the nearest NCERT topic ("Another Mystery") is unrelated |
| The World of Plants ("Plant") | 0.7392 | Covered | uncertain | above 0.7267 but inside the upper interval [0.7203, 0.7564], so a refit on other labelled topics could make it Partially Covered |

A topic scoring exactly 0.6619 is Partially Covered
(`score >= t_partial`).

## 11. Verification

- `uv run pytest tests/test_calibrate.py tests/test_classify.py`: 28 tests pass. The full suite
  (`uv run pytest`) has 86 on this branch.
- `uv run python -m classification.calibrate --dry-run` prints 131 topics (50 / 53 / 28),
  `t_partial 0.6619`, `t_covered 0.7267`, macro-F1 0.7388, leave-one-out 0.6377.
- Running `calibrate` then `run` again changes nothing in the repository except the
  timestamp in `run_meta.json` and the date in `thresholds.json`.
- `uv run ruff check .` and `ruff format --check` are clean.

## 12. Terms

- **Cosine similarity:** how close the directions of two chunk embeddings are (1 = same meaning).
- **F1:** for one class, 2 x hits / (predicted + actual). **Macro-F1:** the average of the three.
- **Bootstrap:** refitting on many resamples of the labelled topics to see how much the cut-offs move.
- **Leave-one-out:** fit without a topic, label it, and repeat for every topic.
- **Provisional:** the thresholds rest on labels that were not all made independently by people.
- **Stable / uncertain:** whether the score is outside or inside a cut-off's 95% interval.
