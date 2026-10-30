# Bridge Draft Check (automated and AI checks)

> **This is not a teacher review.** It is an automated format check of every draft plus an AI
> spot check (an LLM reading 10 drafts against their source passages). No draft has been
> reviewed by a teacher, and none may be used with children until a teacher approves it in the
> app (`docs/BRIDGE.md` §1).

Date: 2026-10-30 · Drafts: `processed/bridge/*.md` · Data: `processed/bridge/index.csv`

## 1. What was generated

| | |
|---|---|
| Gap topics (Phase 4, `processed/gap/run_meta.json`) | 104 (50 Not Covered, 54 Partially Covered) |
| Drafts | 103 (50 Not Covered, 53 Partially Covered), status `draft` |
| Approved packs | 1 (`SCERT_c6_Term_1_03` Mass, prompt v1, approved earlier in the app; not regenerated) |
| Model / prompt | Anthropic `claude-haiku-5-5`, prompt v3, temperature not sent |
| NCERT rule | `t_partial` 0.6619 from `processed/alignment/thresholds.json`: all 50 Not Covered drafts have no NCERT sources, all 53 Partially Covered drafts have two |
| Failures | 0 (4 chunks of 25, after a first run of 3) |
| Tokens | 226,227 in / 261,664 out (about US$0.16 at Haiku 5.5 list prices) |

## 2. Automated checks (all 103 drafts)

Computed by `bridge/run.py` (`bridge.prompts.pack_checks`) and stored as columns of `index.csv`,
so they can be recomputed from the files at any time.

| Check (`index.csv` column) | Rule | Drafts failing |
|---|---|---|
| `missing_sections` | each of the six `## ` headings present as its own line | 0 |
| `empty_sections` | no heading without text under it | 0 |
| `cut_off` | finish reason is not `max_tokens` / `length` | 0 |
| `lesson_words` | Lesson at most 400 words | 0 (3 before regeneration, see below) |
| `key_word_count` | at most 10 key words | 0 |
| `question_count` | at most 5 practice questions | 0 |

| | min | median | max |
|---|---|---|---|
| Lesson words | 122 | 250 | 394 |
| Key words | 3 | 10 | 10 |
| Practice questions | 3 | 5 | 5 |

43 drafts have fewer than 10 key words and 12 have fewer than 5 questions. This is intended
(prompt v3: every count is a maximum): short or list-like passages, such as Force and Motion
(learning goals plus two sentences: 122 words, 3 key words, 3 questions), give short packs instead
of invented content.

**Regenerated.** Three drafts had lessons over 400 words and were regenerated with `--force`:
Atmosphere `SCERT_c6_Term_2_36` (406 → 419 → 367; the longest passage), Skeletal System
`SCERT_c6_Term_2_57` (403 → 313), Food Chain and Food Web `SCERT_c6_Term_3_38` (437 → 360).

## 3. AI spot check (10 drafts)

An LLM compared each draft with the exact passages the model was given (rebuilt with
`bridge.data.get_sources`) and listed every claim not supported by the SCERT passage (or, under
"What you already know", by the NCERT passages). Five Not Covered and five Partially Covered
drafts, one per unit.

| Topic | Label | Unit | Claims not supported by the passage | Other notes |
|---|---|---|---|---|
| Electric Circuits `2_19` | Not Covered | Electricity | Key word "terminal: the end of a cell or battery where the current enters or leaves" (added definition) | Keeps the passage's safety warning |
| Digestive System `2_59` | Not Covered | Human Organ systems | "That is longer than a classroom!" (added comparison); "**Most** absorption happens in the small intestine" (passage: absorption occurs there) | Passage is 7 sentences; note says the mouth and other parts are not named |
| Food Chain and Food Web `3_38` | Not Covered | Our Environment | Key word "Herbivore: an animal that eats **only** plants" (passage: plants or plant products) | Note flags the broken energy sentence |
| Burning and Combustion `2_38` | Not Covered | Air | "We cannot see it" (of oxygen); covered candle "should stop burning" (inferred; the note says so) | Repeats the textbook's own burning-vs-combustion definitions, which are non-standard: a teacher should check them |
| Fertilizers `3_27` | Not Covered | Chemistry in Everyday life | "A chemical change makes a new material" (added definition) | Note says the fertilizer tables are blank, so those questions were dropped |
| Diffusion `1_17` | Partially | Matter Around Us | "Everything is made of tiny bits called particles" (generalisation); key word "Mass: the amount of matter" | Topic text also holds the unit's later sections (compressibility), which the lesson uses |
| Heat and Temperature `2_06` | Partially | Heat | Key word "Heat energy: the energy that heat carries from a hotter thing to a colder thing" (not in this passage) | Note says the garbled temperature figures were left out |
| Composition of water `3_16` | Partially | Water | none | The second NCERT match is about handpicking (weak match), used only for the reminder |
| Storage of Magnets `3_08` | Partially | Magnetism | "Bar magnets are long, straight"; "Horse-shoe magnets are curved, like a U" (descriptions) | Unclear wording: "they can lose their magnetic power ... This is called improper storage"; circular "soft iron is iron that is soft" |
| Plant forms and functions `1_24` | Partially | The World of Plants | none | Note lists what was left out (geotropism, story, Victoria leaf) |

**Result.** No draft states a wrong science fact or teaches content that is not in its passage.
Every unsupported item is a short added explanation, comparison or qualifier ("only", "most"),
mostly in key-word meanings. These are what a teacher should look for in review. None of the 10
was regenerated: regenerating would change wording at random rather than fix a fault, and the
teacher's edit is the intended fix.

Two things a reviewer should know beyond these drafts:
- Some SCERT topic texts contain neighbouring sections or broken fragments (from Phase 2,
  `docs/PREPROCESSING.md` §6). The model usually notices and says so in the note for the teacher.
- Packs repeat the SCERT textbook faithfully, including its weaker passages (e.g. the burning and
  combustion definitions). Checking against the passage does not catch errors in the passage.

## 4. Limits of this check

- 10 of 103 drafts were read; the other 93 only passed the automated format checks.
- The AI spot check is one LLM's reading, not a subject expert's, and it judges support by the
  passage, not age-appropriateness for a particular child.
- Lesson length is counted in words, not reading level.
