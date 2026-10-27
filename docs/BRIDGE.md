# Bridge Content: Documentation

Author: Aaryaveer
Stage: Phase 5, REQ-05 (after gap classification)
Code: `bridge/`, `streamlit_app.py`  ·  Settings: `configs/bridge.yaml`  ·  Tests: `tests/test_bridge.py`

## 1. What this stage does

For a SCERT topic that NCERT does not fully cover, a teacher generates a **bridge pack**: a short
lesson a child who moved from an NCERT state can follow in a Tamil Nadu classroom. The Streamlit
app lets the teacher pick the topic, generate the pack with an LLM, edit it, approve it, and
export it.

Each pack has six sections, in this order:

| Section | Content |
|---|---|
| What you already know | 1-3 sentences linking the topic to NCERT or daily life |
| Lesson | at most 10 minutes read aloud (up to 400 words), one everyday example from the passage |
| Key words | up to 10 words from the SCERT passage, with simple meanings |
| Practice questions | up to 5 questions: recall, then understanding, then one applied to daily life |
| Answers | one answer per question |
| Note for the teacher | what the pack is based on and anything missing or unclear |

Packs can be written in simple English or in Hindi (Devanagari, with the English science word in
brackets, because the class is taught in English).

## 2. How it works

```
gap_matrix.csv ──► pick topic ──► SCERT topic text + 2 closest NCERT topics ──► prompt ──► LLM
                                                                                             │
       processed/bridge/<topic>.<language>.md + .json  ◄── teacher approves ◄── edit ◄──────┘
```

**Inputs** (all from earlier stages, nothing is re-computed):

- `processed/gap/gap_matrix.csv` (Phase 4): label, score and confidence per SCERT topic. The app
  lists Not Covered and Partially Covered topics by default.
- `processed/topics.jsonl` (Phase 2): the full text of the SCERT topic and of the NCERT topics.
- `processed/alignment/bge-large-en-v1.5/topic_matches.csv` (Phase 3): the closest NCERT topics.

**Grounding.** The model gets only the SCERT passage (what the child must learn) and the two closest
NCERT passages (what the child probably knows). The system prompt tells it to use only facts in
the SCERT passage (NCERT facts only under "What you already know") and to name anything missing
in the note for the teacher. Every length and count in the pack format is a maximum, so a short
passage gives a short pack instead of padding. Long passages are cut
at a sentence end (6,000 characters for SCERT, 2,500 per NCERT topic).

**The gap label changes the lesson.** Not Covered: teach from the start, linking only to daily
life. Partially Covered: start from what NCERT covers and spend most of the lesson on what SCERT
adds. Covered: a short revision lesson using SCERT's terms.

**Teacher approval.** Download and save stay disabled until a reviewer is named and confirms they
checked the facts against the passages. The full prompt is shown in the app, so the reviewer can
compare.

**Traceability.** Each saved pack has a `.json` record: topic, label, score, language, provider,
model, prompt version, temperature, token usage, NCERT sources, reviewer, whether the reviewer
edited it, and timestamps.

## 3. LLM providers and API keys

`bridge/llm.py` calls the providers' HTTPS APIs directly with `httpx` (no SDKs):

| Provider | Default model | Key variable |
|---|---|---|
| Google Gemini | `gemini-3.8-flash` | `GEMINI_API_KEY` |
| OpenAI | `gpt-6-luna` | `OPENAI_API_KEY` |
| Anthropic | `claude-haiku-5-5` | `ANTHROPIC_API_KEY` |

Model IDs change often. **List models** in the sidebar shows the text models available to your
key, and picking one uses it. Defaults are in `configs/bridge.yaml`.

The key is typed into the sidebar, or read from the environment variable or from
`.streamlit/secrets.toml` (which is git-ignored):

```toml
GEMINI_API_KEY = "..."
```

The key is sent only in request headers. It never appears in URLs, error messages, saved packs or
logs, and it is not written to disk by the app.

**Temperature is not sent by default** (`generation.temperature: null`). The current default
models ignore or reject it: Gemini 3.6+ ignores custom values, the Claude Haiku 5.5 docs say to
omit it, and OpenAI's reasoning models reject it. If a number is set, it is sent to Gemini and
Anthropic, and the call is retried once without it if the model rejects it; OpenAI never gets one.
Thinking models count their reasoning against `max_tokens` (8,192), and the reasoning tokens are
recorded separately as `usage.thinking_tokens` (Gemini, OpenAI).

## 4. How to run

```
uv sync
uv run streamlit run streamlit_app.py      # opens http://localhost:8501
uv run pytest tests/test_bridge.py
```

## 5. Output files (`processed/bridge/`)

| File | Content |
|---|---|
| `<scert_topic_id>.<language>.md` | the approved pack, with a title line naming the unit, label and reviewer |
| `<scert_topic_id>.<language>.json` | the generation and review record (§2) |

## 6. Verification

- `tests/test_bridge.py` (29 tests) covers source selection and rank order, sentence-end truncation,
  the prompt for each label and language, the request and response format of each provider (with a
  mocked HTTP transport), temperature omitted by default, thinking-token usage, error messages
  that never contain the key, the temperature retry, heading-line checks, saving only with a
  named reviewer, and the app itself (Streamlit `AppTest` with the LLM patched out: generate is
  disabled without a key, save is disabled before approval, edits survive switching topics,
  regenerating replaces the edit and clears the approval, and the key is not in the pack record).
- The app was run in a browser with a stubbed model: generate, edit, approve, download and save
  all work.
- A call with an invalid key returns the provider's own error (for example
  `anthropic API error 401: invalid x-api-key`) without the key.

### Code review (prompt v2)

A review against the current provider docs and the app's state handling found and fixed:

| Severity | Finding | Fix |
|---|---|---|
| High | Regenerating kept the earlier approval, so an unreviewed draft could be saved at once | regenerating clears the approval |
| High | `temperature` 0.4 was sent to models that ignore it (Gemini 3.6+) or say to omit it (Claude Haiku 5.5) | not sent by default (§3) |
| Medium | Prompt v1 required 250-400 words and 8-10 key words even for very short passages, which invites invented content, and mixed "SCERT facts only" with NCERT links | prompt v2: all lengths are maxima, NCERT facts only under "What you already know", exercise questions in passages ignored |
| Low | The missing-sections check matched `## Heading` anywhere, including `###` headings | only a level-2 heading line counts |
| Low | Thinking tokens, which count against the output limit, were not recorded | `usage.thinking_tokens` |

**Not yet done:** prompt v2 has only been checked with tests, not on a real model, and no sample
packs have been generated. Both need an API key in `.streamlit/secrets.toml`.

## 7. Known limitations

1. **Labels are provisional.** The app trusts the Phase 4 label. 60 of 132 labels are uncertain
   (`docs/CLASSIFICATION.md` §6), so a teacher should check the label as well as the lesson.
2. **No automatic fact check.** The proposal's checker agent (back-translation, chrF++, reading
   level) is not built. The teacher's review is the only check, so packs must not be used
   unreviewed.
3. **Hindi is written by the LLM**, not by IndicTrans2 as planned in REQ-06, and has not been
   rated by a Hindi speaker or scored with BLEU/COMET.
4. **Diagrams are missing.** The passages are text only, and figure labels inside them are noisy
   (`docs/PREPROCESSING.md` §6).
5. **One pack at a time.** There is no batch mode; generating the sample packs for the report
   means one click per topic.
