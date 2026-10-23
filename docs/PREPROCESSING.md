# Preprocessing: Documentation

Author: Yuvaraj Guliani
Stage: preprocessing (after ingestion/extraction, before embedding)
Code: `preprocessing/`  ·  Settings: `configs/preprocessing.yaml`  ·  Tests: `tests/`

## 1. What we had

Input is 16 English-medium Class 6 Science PDFs, chosen at ingestion:

| Board | Files | Content |
|---|---|---|
| NCERT | `data/NCERT/fecu101.pdf` – `fecu112.pdf` | "Curiosity", 12 chapters, one PDF each |
| NCERT | `data/NCERT/fecu1ps.pdf` | Front matter (cover, foreword, contents), not used |
| SCERT (Tamil Nadu) | `data/SCERT/Term_1.pdf` – `Term_3.pdf` | Standard VI Science, 3 term books, 20 units |

Raw text from these PDFs cannot be embedded as it is:

- It is full of page furniture: page numbers, footers ("Reprint 2026-27"), printer file
  names on every SCERT page, side labels ("Curiosity | Textbook of Science | Grade 6").
- Recurring box titles ("Do you know?", "Activity 4.1", "ICT Corner") add no meaning.
- Lines are broken mid-sentence and words are hyphenated across lines.
- A whole chapter is far too long for one embedding (NCERT Ch. 9 is about 17,000 characters).
- The SCERT books end with an English–Tamil glossary and credits pages.
- NCERT chapters open with Sanskrit/Hindi verses that are not science content.
- Some SCERT diagram labels are stored in a broken font and come out garbled
  ("6NXOO" instead of "Skull").

## 2. What the pipeline produces

`processed/chunks.jsonl` is the input for the embedding stage: **705 chunks** from
**294 topics**, every chunk at most 1,200 characters.

| | NCERT | SCERT | Total |
|---|---|---|---|
| Topics, content | 82 | 132 | 214 |
| Topics, summary / exercises / extension | 35 | 45 | 80 |
| **Chunks, content (embed these)** | **251** | **294** | **545** |
| Chunks, summary / exercises / extension | 67 | 93 | 160 |

For embedding and gap analysis, use only chunks with `section == "content"`. Summaries and
exercises repeat the teaching text and would add noise.

```python
import json

chunks = [json.loads(line) for line in open("processed/chunks.jsonl", encoding="utf-8")]
content = [c for c in chunks if c["section"] == "content"]  # 545 chunks
```

## 3. How it works

```
PDF ──► 1. extract ──► 2. segment ──► 3. clean ──► 4. chunk ──► processed/
        lines + font    topics         clean text   ≤1,200 chars
```

Run everything with `uv run python -m preprocessing.run`. The steps live in four files.

### Step 1. Extract (`preprocessing/extract.py`)

**What:** reads every line of text on every page, together with its font size, whether it
is bold, and its position on the page.

**Why:** the font information is what lets step 2 tell a heading from body text; the
position is needed to fix pages where the PDF stores text out of order.

**How:** PyMuPDF reads the text layer (`native_lines`). If a page has fewer than 50
selectable characters it is a scanned image, so it is rendered at 300 dpi and read with
Tesseract OCR (`ocr_lines`). In these books only 5 NCERT pages needed OCR (full-page
figures). Results are cached in `cache/pages/` so later runs take about a minute.

### Step 2. Split into topics (`preprocessing/segment.py`)

**What:** cuts each book into topics, where a topic is one section of a chapter
(e.g. "Balanced Diet", "3.3 Diffusion"), and tags each with a section type.

**Why:** a section covers one idea, which is the right unit for finding where the two
curricula match or differ. A whole chapter mixes many ideas.

**How:**

- **NCERT:** the chapter title is the large text on page 1 (`chapter_title_ncert`). A new
  topic starts at each bold line clearly larger than the body text (`is_heading`).
- **SCERT:** a unit starts on its opener page, the one with a large "Unit" heading
  (`units_from_openers`). Topics start at numbered sections such as "1. 1 Length" or
  "3.2 Mass, Shape and Volume of Solids", and the number must match the unit.
- Headings that wrap onto two lines are joined (`join_wrapped_headings`).
- Not treated as headings: box titles ("Do you know?", "Activity 9.1"), page numbers,
  table headers, instruction sentences.
- Skipped entirely: covers, contents and preface pages; the SCERT glossary and the
  credits pages after it (`back_matter_start`); the NCERT front-matter file.
- Removed: the Sanskrit/Hindi verses at the start of NCERT chapters 2, 3, 6 and 9, with
  their English translation and attribution (`drop_verses`). The team decided these are
  not needed. The Ayurveda verse in Ch. 6 p.17 is removed, but the chapter's own
  explanation of the 20 properties after it is kept.
- **Section types** (`section` field): `content` for teaching text; `summary` (Summary,
  Keywords, Points to Remember); `exercises` (Let us enhance our learning, Evaluation);
  `extension` (Learning further, ICT Corner).
- Some NCERT pages store end-of-chapter headings after their own text (for example
  "Learning further" in Ch. 6 and 8). Those pages are regrouped by position on the page so
  the text sits under the right heading (`place_end_headings`).
- Topics shorter than 300 characters are merged into the neighbouring topic of the same
  type (`merge_short`).

### Step 3. Clean (`preprocessing/clean.py`, `clean_text`)

**What:** turns a topic's raw lines into one clean paragraph.

**How, in order:**

1. Garbled SCERT diagram labels are decoded (`unshift_line`). Their font stores every
   character 29 code points too low, so "6NXOO" is "Skull" and "$OO ZDWHU" is "All water".
   A line is decoded only if the result is made of real words, so normal capitals such as
   "WATER" are never touched.
2. Unicode is normalised. Typographic quotes, dashes and spaces become plain ones. Symbol-
   font characters become real symbols (×, ÷, =).
3. Any Devanagari is normalised (`normalize_hindi`). After the verse removal the final
   output contains none.
4. Page furniture and box labels are dropped line by line (`keep_line`).
5. Fill-in blanks (`_____`, `.....`) and bullets are removed.
6. Lines are joined into running text and hyphenation is undone ("pic-" + "tures" →
   "pictures").

### Step 4. Chunk (`preprocessing/clean.py`, `split_chunks`)

**What:** splits each topic into chunks of at most 1,200 characters.

**Why:** embedding models have an input limit. Cutting a long topic at 1,200 characters
would throw away most of it (NCERT Ch. 1 would keep 1,152 of 8,287 characters), so long
topics are split instead and nothing is lost.

**How:**

- Chunks end at sentence ends, never mid-sentence.
- Chunks are sized evenly: a 2,500-character topic becomes three chunks of about 830,
  not 1,200 + 1,200 + 100.
- Every chunk starts with its topic title ("Balanced Diet. …"), so each chunk says what it
  is about even on its own.
- Example: NCERT Ch. 1 (8,287 characters) → 8 chunks; Ch. 9 (17,093 characters) →
  16 chunks.

Topics that are empty, too short (under 150 characters) or mostly junk are dropped and
listed with the reason in `processed/dropped_topics.jsonl` (`assess`). Six were dropped:
four ICT Corner boxes that are only screenshots, and two short Evaluation sections in the
SCERT computer units.

## 4. Output files (`processed/`)

| File | One row per | Use |
|---|---|---|
| `chunks.jsonl` | chunk | **Embedding input** |
| `topics.jsonl` | topic | Full, uncut text of each topic (reference, gap analysis) |
| `dropped_topics.jsonl` | dropped topic | What was removed and why |
| `summary.csv` | PDF | Topics found, kept, and chunks per book |
| `verification.csv` | missing line | Every PDF line not in the output, with the reason |

Fields in `chunks.jsonl`:

| Field | Example | Meaning |
|---|---|---|
| `chunk_id` | `SCERT_c6_Term_1_03_2` | topic id + chunk number |
| `topic_id` | `SCERT_c6_Term_1_03` | board, class, file, topic number |
| `board`, `grade`, `term` | `SCERT`, `6`, `1` | `term` is null for NCERT |
| `source_file` | `SCERT/Term_1.pdf` | |
| `chapter`, `topic_title` | `Measurements`, `Mass` | |
| `section` | `content` | content / summary / exercises / extension |
| `page_start`, `page_end` | `11`, `12` | pages in the PDF viewer (1-based) |
| `chunk_index`, `n_chunks` | `2`, `2` | position of this chunk in its topic |
| `ocr`, `lang`, `chars` | `false`, `en`, `1015` | |
| `text` | `Mass. 1000 milligram = 1 gram …` | the text to embed |

## 5. How it was verified

**Automated check of every line** (`uv run python -m preprocessing.verify`). For every
line of text in every PDF, the script checks whether it appears in the output. If it does
not, it must fall under a known reason; anything else is flagged as "unexplained". Result
over all 15 books (25,889 lines with text):

| Outcome | Lines | What it is |
|---|---|---|
| Found in the output | 21,613 | |
| Page furniture | 3,204 | page numbers (979), list markers like "a)" (767), SCERT print file names (684), box labels and side labels (525), NCERT footers (249) |
| Outside the chapters | 865 | covers, contents, SCERT glossary and credits pages |
| Heading | 188 | kept as the topic title, without its section number |
| Verse | 17 | the Sanskrit/Hindi verses and their translations |
| In a dropped topic | 2 | |
| **Unexplained** | **0** | |

The page-furniture lines were also reviewed by category to confirm no real sentences were
being removed.

**Manual comparison with the PDFs.** Pages were rendered and read side by side with the
output, including NCERT Ch. 4 p.2 (Exploring Magnets) and SCERT Term 2 p.76 (Skeletal
System, two columns plus a diagram). All sentences on those pages are present, in reading
order, including both SCERT columns and the "Do you know?" facts. These checks are also
how the problems in section 6 were found and fixed: garbled diagram labels, misplaced
"Learning further" text, the NCERT side label, the SCERT glossary and credits.

**Tests:** 43 automated tests (`uv run pytest`) cover the cleaning rules, topic
splitting, chunking (no chunk over the limit, no text lost), verse removal and file
discovery.

## 6. Known limitations

These were found during verification and left as they are, because fixing them reliably
would risk removing real text.

1. **Diagram and table text is included.** Figure captions ("Fig. 4.1: Some common items
   that have magnets attached to them"), diagram labels ("Bar magnet U-shaped magnet Ring
   magnet") and table cells come out as short fragments inside the topic text. Short
   label-like fragments are about 3% of NCERT and 7% of SCERT content text. They cannot be
   removed by position because both books place full-page background images behind all
   text. Captions are mostly meaningful, so they were kept.
2. **Side boxes can move.** A box printed beside or just above a heading (e.g. "Do magnets
   stick to objects made of certain materials only?") can land at the start of the next
   topic instead of the end of the previous one. It stays in the same chapter.
3. **Four diagram labels stay garbled** ("ammonia", "intake", "muscular sac that", "the
   inferior vena cava" in the SCERT broken font): they could not be safely told apart from
   real capitals.
4. **SCERT computer units.** Their two-column exercise layout puts a few multiple-choice
   questions at the end of the preceding content topic (Term 3, "Paid and Proprietary
   Software").
5. **Uneven topic sizes.** NCERT Ch. 1 and Ch. 9 have no section headings, so each is one
   topic split into 8 and 16 chunks. Some SCERT subsections are short and were merged into
   the next topic.

## 7. Changes from the original proposal

- **Class 3–5 Maths → Class 6 Science** (ingestion decision): the extraction of maths
  symbols and notation was not clean enough for the meaning to survive embedding.
- **Both boards are English medium**, so multilingual embedding is not needed. The
  preprocessing removes the little Devanagari there was (decorative verses) and the
  English–Tamil glossary pages.

## 8. How to run

See `docs/How to Run - Walkthrough.pdf` for screenshots. In short:

```
git pull
pip install uv                                  # one time
winget install UB-Mannheim.TesseractOCR         # one time (Mac: brew install tesseract)
uv sync
uv run python -m preprocessing.run              # writes processed/
uv run python -m preprocessing.verify           # optional: line-by-line check
uv run pytest                                   # optional: 43 tests
```

All settings (chunk size, thresholds, folders) are in `configs/preprocessing.yaml`.
