# Embedding & Alignment: Documentation

Author: Aaryaveer
Stage: Phase 3, REQ-03 (after preprocessing, before gap labelling in Phase 4)
Code: `alignment/`  ·  Settings: `configs/alignment.yaml`  ·  Tests: `tests/test_align.py`

## 1. What this stage answers

The gap analysis compares two Class 6 Science curricula:

| Role | Board | Content topics | Content chunks |
|---|---|---|---|
| Origin (pivot curriculum the home-state books follow) | NCERT "Curiosity" | 82 | 251 |
| Destination | SCERT Tamil Nadu, Standard VI | 132 | 294 |

Main question: **for each SCERT topic, how well does NCERT cover it?** The reverse direction
(each NCERT topic against SCERT) is also computed for the report.

This stage ends at similarity matrices, ranked matches, a human gold set and (after labelling)
recommended thresholds. Phase 4 turns the scores into Covered / Partially Covered / Not Covered.

Only `section == "content"` chunks are used (see `docs/PREPROCESSING.md` §2).

## 2. Why it is built this way

**Directional coverage.** "Does NCERT cover this SCERT topic?" is not the same question as
"does this SCERT topic cover NCERT?". A short SCERT topic can be fully inside a long NCERT
topic while the reverse is not true. So coverage is directional:

> `C[d, o]` = mean over the chunks *i* of SCERT topic *d* of (max over the chunks *j* of NCERT
> topic *o* of cosine(*i*, *j*)).

Every part of the SCERT topic looks for its best match inside the NCERT topic, and the topic
is as covered as its parts are on average. `C` is not symmetric.

**Chunk max-mean instead of whole-topic embeddings.** Topics range from 260 to 17,000
characters. One embedding of a long topic averages many ideas into a vague vector, and the
models only read 512 tokens anyway. Chunks (≤1,200 characters, each starting with the topic
title) keep each idea visible. Whole-topic similarity is still reported as `centroid_sim`
(chunk embeddings averaged per topic, weighted by length) for comparison.

**`best_coverage_any`.** Per SCERT topic, the mean over its chunks of the best match among
*all* NCERT chunks, ignoring NCERT topic boundaries. If a SCERT topic is taught across two NCERT
sections, no single `C[d, o]` is high, but `best_coverage_any` is. It is the main signal for
Not Covered and is always ≥ every `C[d, o]` for that topic.

**Models.**

| Key | Model | Why |
|---|---|---|
| `bge` (primary) | `BAAI/bge-large-en-v1.5` | Both boards are English medium, and the team decided (Slack, 23 Oct) not to use multilingual embedding. bge is a strong English retrieval model. No instruction prefix: this is symmetric passage-to-passage similarity. |
| `e5` | `intfloat/multilingual-e5-large` | The model named in the RTM (REQ-03). Shows what dropping multilingual costs or gains. e5 is trained with prefixes; the model card says to use `"query: "` on both sides for symmetric tasks, so every text gets `"query: "`. |
| `tfidf` | scikit-learn TF-IDF | Lexical baseline (`sublinear_tf`, English stop words), fit on all 545 content chunks, L2-normalised. Shows what the dense models add beyond word overlap. |

All embeddings are unit-length, so cosine is a dot product.

## 3. How to run

```
uv sync
uv run python -m alignment.run --model bge      # primary (default)
uv run python -m alignment.run --model e5
uv run python -m alignment.run --model tfidf
uv run python -m alignment.make_gold            # after all three runs
uv run pytest
```

The first dense run downloads the model (bge ≈1.3 GB, e5 ≈2.2 GB) and embeds on `cuda`, then
`mps`, then `cpu`. Embeddings are cached in `cache/embeddings/<model>/<sha256>.npy`. The hash
covers the model name, the prefix and every `(chunk_id, text)` in order, so any change to the
preprocessing output forces a re-embed; otherwise a re-run takes about a second and gives
byte-identical outputs (apart from `run_meta.json`, which has a timestamp and runtime).

## 4. Output files

`processed/alignment/<model>/` (`bge-large-en-v1.5`, `multilingual-e5-large`, `tfidf`):

| File | Content |
|---|---|
| `topic_coverage.csv` | rows SCERT `topic_id`, columns NCERT `topic_id`, cell `C[d, o]`. **The Phase 3 exit matrix.** |
| `topic_coverage_reverse.csv` | rows NCERT, columns SCERT (how well SCERT covers each NCERT topic) |
| `topic_centroid_sim.csv` | 214 × 214 centroid cosine over all content topics of both boards (symmetric) |
| `topic_matches.csv` | top 5 NCERT topics per SCERT topic: chapters, titles, `coverage`, `centroid_sim`, `best_coverage_any`, and the evidence chunk pair with the highest cosine inside that topic pair (ids, cosine, first 200 characters of each) |
| `chunk_sim.npy`, `chunk_ids_dest.json`, `chunk_ids_orig.json` | raw 294 × 251 chunk cosine matrix and its row / column ids |
| `chapter_heatmap.png` | SCERT units × NCERT chapters, cell = max topic coverage in that block, ordered so each unit sits next to its best chapter |
| `sanity_report.txt` | top-1 NCERT chapter per SCERT unit, the computer units' scores, catch-all check |
| `run_meta.json` | model, prefix, dimensions, counts, runtime, git commit, timestamp |

Rows and columns follow the order of `processed/topics.jsonl`.

`eval/`:

| File | Content |
|---|---|
| `gold_pairs_template.csv` | 252 unlabelled (SCERT topic, NCERT candidate) pairs for human labelling |
| `gold_topics.csv` | the 40 sampled SCERT topics, how each was picked, its tercile, number of candidates |
| `LABELLING_GUIDE.md` | label definitions with one real example of each |

## 5. First results (before evaluation)

Score ranges of `best_coverage_any` over the 132 SCERT topics:

| Model | min | median | max | 3 computer units (mean) | other topics (mean) |
|---|---|---|---|---|---|
| bge | 0.521 | 0.674 | 0.869 | 0.561 | 0.686 |
| e5 | 0.832 | 0.895 | 0.950 | 0.862 | 0.896 |
| tfidf | 0.039 | 0.130 | 0.599 | 0.073 | 0.156 |

- The dense models' cosines sit in a narrow, high band (e5: 0.83–0.95), so a fixed threshold
  such as 0.5 means nothing. Thresholds must be calibrated on the gold set (§6).
- **bge** places 11 of the 12 computer-unit topics among the 17 lowest-scoring SCERT topics.
  The 6 other topics in that band are Uses of Thermal Expansion, Nervous System, The
  Structural Organization (The Cell), Science Today - Robot, Plaster of Paris and Desirable and
  Undesirable changes, several of which have no obvious NCERT Class 6 counterpart. e5 separates
  the computer topics less (47 other topics score below the highest computer topic; 10 for bge).
- Unit-level matches are as expected where both books have the chapter (Magnetism → Exploring
  Magnets, Heat → Temperature, Water → States of Water). SCERT units without an NCERT
  counterpart (Electricity, The Cell, Chemistry in Everyday Life) still get a "best" chapter,
  usually Nature's Treasures, at a lower score.
- NCERT Ch. 1 and Ch. 9 are each one long topic. Under bge they are the top-1 match for 4 and
  8 SCERT topics (coverage) vs 3 and 8 (centroid), so the max-over-chunks measure does not make
  them a strong catch-all.

## 6. Evaluation (pending)

**Gold-set sampling** (`alignment/make_gold.py`, seed 42):

1. One random topic from each of the 20 SCERT units, so every unit is represented, including
   all three computer units (Computer - An Introduction, Hardware and Software, Parts of
   Computer) as negative controls.
2. 20 more topics stratified by bge `best_coverage_any` terciles over all 132 SCERT topics:
   7 low, 7 mid, 6 high, drawn from topics not already picked. A purely random sample would be
   dominated by well-covered topics; the low and mid terciles make sure Not Covered and
   Partially Covered cases exist to calibrate thresholds.
3. Candidates per topic (pooling, so the gold set is not biased toward one model): the union
   of the top-3 NCERT topics by coverage from bge, e5 and tfidf, plus the top-3 by bge
   `centroid_sim`. Rows are shuffled, scores and model names hidden.

Result: 40 topics (15 low, 15 mid, 10 high tercile; 5 computer topics), 252 pairs, 40 marked for
a second annotator. Because of the stratification, the label distribution in the gold set is
**not** the distribution over all SCERT topics, so proportions from it must not be reported as
corpus-wide gap rates.

Metrics, thresholds and agreement are added here after the labelled `eval/gold_pairs.csv`
comes back (`alignment/evaluate.py`).

## 7. Known limitations

1. **Gold set is small.** 40 topics (5 of them computer topics) is enough to compare models
   and calibrate two thresholds roughly; bootstrap confidence intervals will be wide.
2. **Narrow score bands.** Dense cosines differ by a few hundredths between a real match and
   an unrelated topic, so small threshold changes move many topics between labels.
3. **Shared vocabulary.** Word overlap can outrank meaning: the SCERT electric "Cell" matches
   NCERT States of Water, and SCERT "Cell components and their functions" matches NCERT
   "Components of Food" under bge. Evidence chunks in `topic_matches.csv` make these visible.
4. **Unit-level "best chapter" always exists.** The heatmap shows the maximum topic coverage
   per block; a unit with no counterpart still has a brightest cell. Read the value, not just
   the position.
5. **Inherited from preprocessing.** Diagram labels and captions are part of the chunk text,
   NCERT Ch. 1 and Ch. 9 are single long topics, and a few SCERT exercise questions sit inside
   content topics (`docs/PREPROCESSING.md` §6).
