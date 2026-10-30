# NLP Curriculum Bridge (Migrant Kids)

Corpus: Class 6 Science (English medium).

- `data/NCERT/fecu1*.pdf`: NCERT "Curiosity", one PDF per chapter. `fecu1ps.pdf` is the front matter.
- `data/SCERT/Term_*.pdf`: Tamil Nadu SCERT Standard VI Science, Terms I–III.

`extract_textbooks.py` writes page text, images and manifests to `extracted/`. See the
script's docstring for details.

## Topic preprocessing (`preprocessing/`)

Turns the textbook PDFs into clean text chunks ready for embedding. Full documentation
(what, why, how, verification, limitations): [`docs/PREPROCESSING.md`](docs/PREPROCESSING.md).
Step-by-step screenshots: `docs/How to Run - Walkthrough.pdf`.

```
uv sync
uv run python -m preprocessing.run      # writes processed/
uv run python -m preprocessing.verify   # optional: checks every PDF line against the output
uv run pytest
```

The first run reads every PDF and caches the lines in `cache/pages/`, which takes a few
minutes. Pages with no selectable text are OCR'd with Tesseract, which must be installed
(Windows: `winget install UB-Mannheim.TesseractOCR`).

**Stages**

1. `extract.py`: reads each page's text lines with font size, weight and position; OCR
   for pages without a text layer.
2. `segment.py`: splits each book into topics (NCERT bold subheadings, SCERT numbered
   sections) and tags each as `content`, `summary`, `exercises` or `extension`. Skips
   covers, contents, the SCERT glossary and credits; removes the NCERT Sanskrit/Hindi
   verses with their translations.
3. `clean.py`: removes page furniture and box labels, rejoins broken lines, decodes SCERT
   diagram labels stored in a broken font, and splits each topic into chunks of at most
   1,200 characters at sentence ends, each starting with the topic title.

**Outputs** (`processed/`)

- `chunks.jsonl`: **the embedding input**, one chunk per line (705 chunks). Embed only
  `section == "content"` (545 chunks).
- `topics.jsonl`: one topic per line with its full text (294 topics).
- `dropped_topics.jsonl`: removed topics with the reason.
- `summary.csv`: topics and chunks per PDF.
- `verification.csv`: every PDF line not in the output, with the reason (from `verify`).

## Embedding & alignment (`alignment/`)

Measures how well each SCERT topic is covered by NCERT (directional, chunk-level cosine),
with `bge-large-en-v1.5` as the primary model and `multilingual-e5-large` and TF-IDF for
comparison. Full documentation: [`docs/ALIGNMENT.md`](docs/ALIGNMENT.md).

```
uv run python -m alignment.run --model bge     # also: e5, tfidf -> processed/alignment/<model>/
uv run python -m alignment.make_gold           # eval/gold_pairs_template.csv for human labelling
uv run python -m alignment.evaluate            # metrics + Phase 4 thresholds from eval/gold_pairs.csv
```

## Gap classification (`classification/`)

Turns the alignment scores into the curriculum gap matrix (Covered / Partially Covered / Not
Covered per SCERT topic) using the thresholds in `processed/alignment/thresholds.json`. Full
documentation: [`docs/CLASSIFICATION.md`](docs/CLASSIFICATION.md).

```
uv run python -m classification.calibrate   # refit thresholds from eval/human_labels.csv
uv run python -m classification.run         # writes processed/gap/
```

`calibrate` rewrites `processed/alignment/thresholds.json` from the researcher-reviewed labels
in `eval/human_labels.csv`; it is deterministic, so the same labels give the same thresholds and
the same gap matrix.

## Bridge content app (`bridge/`, `streamlit_app.py`)

A Streamlit app that writes a bridge pack (short lesson, key words, five practice questions with
answers) for a Not Covered or Partially Covered SCERT topic, using the SCERT and closest NCERT
passages and an LLM API key (Gemini, OpenAI or Anthropic). A teacher edits and approves each
pack before it can be downloaded or saved to `processed/bridge/`. Full documentation:
[`docs/BRIDGE.md`](docs/BRIDGE.md).

```
uv run streamlit run streamlit_app.py   # key: sidebar, env var or .streamlit/secrets.toml
```
