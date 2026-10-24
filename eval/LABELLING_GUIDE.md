# Labelling Guide: Gold Set for SCERT–NCERT Alignment

File to label: `eval/gold_pairs_template.csv` (save your copy as `eval/gold_pairs.csv`).
Stage: Phase 3 evaluation (REQ-03). The labels are the ground truth for the alignment models,
so they must come from a person reading the texts, not from any model output.

## 1. What you are judging

Each row is one pair of topics from the two Class 6 Science books:

- `scert_title`, `scert_text`: a topic from the Tamil Nadu SCERT book (the destination).
- `ncert_title`, `ncert_text`: a topic from the NCERT "Curiosity" book (the origin).

The question for every row is: **does this NCERT topic teach what this SCERT topic teaches?**

Texts are cut at 1,500 characters. They are extracted from PDFs, so expect figure captions,
diagram labels and activity headings mixed into the sentences. Ignore those and judge the
science content.

The rows are shuffled. The same SCERT topic appears in several rows, each with a different
NCERT candidate. Judge each row on its own: a topic can have more than one `same` or none.

## 2. Labels

Write exactly one of these words in `label`:

| Label | Use when |
|---|---|
| `same` | The two topics teach the same concept at a similar depth. A student who learnt the NCERT topic would already know what the SCERT topic teaches. |
| `partial` | The topics overlap, but one side covers clearly more than the other (more sub-concepts, definitions, examples or depth). |
| `prerequisite` | The NCERT topic teaches something a student needs before the SCERT topic, but it does not teach the SCERT topic itself. |
| `none` | No meaningful overlap. Sharing a word ("cell", "components") or a general theme ("science is everywhere") is not overlap. |

How to decide, in order:

1. Is any concept in the SCERT topic also taught in the NCERT topic? If no, go to step 3.
2. Is the coverage roughly equal on both sides? `same`. Does one side clearly teach more? `partial`.
3. Is the NCERT topic needed to understand the SCERT topic? `prerequisite`. Otherwise `none`.

When unsure between two labels, pick the one with less overlap (order: `same`, `partial`,
`prerequisite`, `none`) and explain in `notes`.

## 3. Examples from this corpus

None of these pairs is in the template.

**`same`**: SCERT *Health and Hygiene: Components of Food* (`SCERT_c6_Term_1_36`) and NCERT
*Mindful Eating: What are the Components of Food?* (`NCERT_c6_fecu103_05`). Both define
nutrients and teach carbohydrates, proteins, fats, vitamins and minerals with food sources,
and both cover deficiency diseases such as scurvy and rickets. The sizes and depth are similar.

**`partial`**: SCERT *Water: Water cycle* (`SCERT_c6_Term_3_17`) and NCERT *A Journey through
States of Water: How do Clouds give us Rain?* (`NCERT_c6_fecu108_08`). Both explain how water
vapour rises, cools, condenses into droplets that form clouds, and falls as rain. The SCERT
topic also names the whole cycle with its three stages (evaporation, condensation,
precipitation) and covers transpiration, which this NCERT topic does not.

**`prerequisite`**: SCERT *Heat: Expansion in solids* (`SCERT_c6_Term_2_08`) and NCERT
*Temperature and its Measurement: Temperature* (`NCERT_c6_fecu107_03`). The NCERT topic
defines temperature as how hot or cold a body is and introduces thermometers. The SCERT topic
needs that idea but teaches something else: substances expand when heated, explained with
molecules moving faster and spreading apart.

**`none`**: SCERT *Hardware and Software: System Software* (`SCERT_c6_Term_3_62`) and NCERT
*Temperature and its Measurement: Temperature* (`NCERT_c6_fecu107_03`). The SCERT topic is
about operating systems; the NCERT topic is about temperature and thermometers. Both mention
a "device", but nothing is taught in common.

## 4. Columns

| Column | What to do |
|---|---|
| `label` | Your label (first annotator). Fill in every row. |
| `label_2` | Second annotator only, on rows with `double_label = yes`. Label without looking at `label`. |
| `double_label` | Do not change. The 40 `yes` rows are used to measure agreement (Cohen's kappa). |
| `notes` | Optional. Say why when a row was hard, or if the text looks broken. |
| `cov_tercile` | Sampling information only. Do not change it and do not use it to decide a label. |
| everything else | Do not change. |

Save as UTF-8 CSV with the same columns. Excel: *File > Save As > CSV UTF-8*.
