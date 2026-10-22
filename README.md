# NLP Curriculum Bridge (Migrant Kids)

Corpus: Class 6 Science (English medium).

- `data/NCERT/fecu1*.pdf`: NCERT "Curiosity", one PDF per chapter. `fecu1ps.pdf` is the front matter.
- `data/SCERT/Term_*.pdf`: Tamil Nadu SCERT Standard VI Science, Terms I–III.

`extract_textbooks.py` writes page text, images and manifests to `extracted/`. See the
script's docstring for details.

## Topic preprocessing (`preprocessing/`)

Turns the textbook PDFs into clean, length-capped **topic** text ready for embedding.

```
uv sync
uv run python -m preprocessing.run --config configs/preprocessing.yaml
uv run pytest
```

The first run reads every PDF and caches the lines in `cache/pages/`, which takes a few
minutes. Pages with no selectable text are OCR'd with Tesseract, which must be installed
(Windows: `winget install UB-Mannheim.TesseractOCR`). Science pages only need this for a
few full-page figures.

**Stages**

1. `extract.py`: reads each page's text lines with font size and weight. A page with
   fewer than 50 selectable characters is rendered at 300 dpi and OCR'd.
2. `segment.py`: splits each book into topics.
   - NCERT: the chapter title comes from page 1. Bold subheadings larger than body text
     start sub-topics. Box labels ("Do you know?", "Activity 9.1") are not topics. The
     Sanskrit/Hindi verses printed as quotes (chapters 2, 3, 6, 9) are removed together
     with their English translations.
   - SCERT: units start on their "Unit" opener page. Numbered sections ("3.2 Diffusion")
     start topics.
   - Headings that wrap onto two lines are joined. Covers, contents pages and
     `fecu1ps.pdf` are skipped, as are the English-Tamil glossary and credits pages at
     the back of each SCERT book. Topics under 300 characters are merged into a
     neighbour.
3. `clean.py`:
   - Removes footers, print slugs, page numbers, box labels, fill-in blanks and bullets.
   - Undoes line wraps and hyphenation.
   - Hindi normalisation (NFC, zero-width removal, Devanagari digits → 0-9, `|` → `।`,
     doubled matras collapsed) for any Devanagari that remains; with the verses removed,
     the current Class 6 Science output has none.
   - Caps each topic at 1,200 characters, cutting at a sentence end.
   - Drops topics that are empty, too short, or mostly junk.

**Outputs** (`processed/`)

- `topics.jsonl`: one kept topic per line, with fields `topic_id, board, grade, term,
  source_file, chapter, topic_title, section, page_start, page_end, ocr, lang,
  full_chars, truncated, text`.
  - `text` starts with the topic title and is what should be embedded.
  - `section` is `content` for teaching text. End-of-chapter material is `summary`
    (Summary, Keywords, Points to Remember), `exercises` (Let us enhance our learning,
    Evaluation) or `extension` (Learning further, ICT Corner). Filter on it as needed.
- `dropped_topics.jsonl`: the removed topics, each with a `drop_reason`.
- `summary.csv`: topics found and kept per PDF.
