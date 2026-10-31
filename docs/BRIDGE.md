# Bridge Content: Documentation

Author: Aaryaveer
Stage: Phase 5, REQ-05 (after gap classification)
Code: `bridge/`, `streamlit_app.py`  ·  Settings: `configs/bridge.yaml`  ·  Tests: `tests/test_bridge.py`, `tests/test_bridge_run.py`

## 1. What this stage does

For a SCERT topic that NCERT does not fully cover, a **bridge pack** is a short lesson a child
who moved from an NCERT state can follow in a Tamil Nadu classroom. Phase 4 found 104 such topics
(50 Not Covered, 54 Partially Covered of 132; `processed/gap/run_meta.json`).

Packs are made in two steps:

1. **Drafts.** `bridge.run` asks an LLM for a draft of every gap topic and saves it with status
   `draft`. A draft is unreviewed: it starts with a line saying it must not be used with children
   before a teacher has reviewed it.
2. **Approval.** In the Streamlit app a teacher opens a topic (its draft loads into the editor),
   checks it against the textbook passages, edits it, and approves it with their name. Only then
   can it be downloaded, and saving turns it into an `approved` pack.

Each pack has six sections, in this order:

| Section | Content |
|---|---|
| What you already know | 1-3 sentences linking the topic to NCERT or daily life |
| Lesson | at most 10 minutes read aloud (up to 400 words), one everyday example from the passage |
| Key words | up to 10 words from the SCERT passage, with simple meanings |
| Practice questions | up to 5 questions: recall, then understanding, then one applied to daily life |
| Answers | one answer per question |
| Note for the teacher | what the pack is based on and anything missing or unclear |

Packs are written in simple English, the language the Tamil Nadu class is taught in.

## 2. How it works

```
gap_matrix.csv ──► topic ──► SCERT passage + NCERT passages (none if Not Covered) ──► prompt ──► LLM
                                                                                              │
  processed/bridge/<topic>.md + .json   status: draft  ◄──────────────────────────────────────┘
                     │
                     └──► app: teacher checks, edits, names themselves ──► status: approved
```

**Inputs** (all from earlier stages, nothing is re-computed):

- `processed/gap/gap_matrix.csv` (Phase 4): label, score and confidence per SCERT topic. The app
  lists Not Covered and Partially Covered topics by default.
- `processed/topics.jsonl` (Phase 2): the full text of the SCERT topic and of the NCERT topics.
- `processed/alignment/bge-large-en-v1.5/topic_matches.csv` (Phase 3): the closest NCERT topics.

**Which NCERT passages.** A Not Covered topic gets **no NCERT passages**: its closest NCERT
sections are, by definition, not related enough to build on, and showing them invited lessons
that linked to unrelated NCERT content. The rule is the Phase 4 cut-off itself: no NCERT passages
when the coverage score is below `t_partial`, read at run time from
`processed/alignment/thresholds.json` for the primary model (`paths.thresholds` in the config;
no threshold is written in the code). The prompt then says "No related NCERT section was found"
and the app shows "No related NCERT section found". If this rule disagrees with the gap-matrix
label, the two files are out of sync, and the app and `bridge.run` warn to re-run
`classification.run`. Partially Covered and Covered topics get the two closest NCERT passages.

**Grounding.** The model gets only the SCERT passage (what the child must learn) and, if any, the
NCERT passages (what the child probably knows). The system prompt tells it to use only facts in
the SCERT passage (NCERT facts only under "What you already know") and to name anything missing
in the note for the teacher. Every length and count in the pack format is a maximum, so a short
passage gives a short pack instead of padding. Long passages are cut
at a sentence end (6,000 characters for SCERT, 2,500 per NCERT topic).

**The gap label changes the lesson.** Not Covered: teach from the start, linking only to daily
life. Partially Covered: start from what NCERT covers and spend most of the lesson on what SCERT
adds. Covered: a short revision lesson using SCERT's terms.

**Teacher approval.** Download and save stay disabled until a reviewer is named and confirms they
checked the facts against the passages. Regenerating a pack clears an earlier approval. The full
prompt is shown in the app, so the reviewer can compare.

**Traceability.** Each pack has a `.json` record: status (`draft` or `approved`), topic, label,
score, provider, model, prompt version, temperature, token usage, NCERT sources, the NCERT rule
and the `t_partial` it used, and timestamps; approved packs add the reviewer and whether they
edited the draft.

## 3. LLM providers and API keys

`bridge/llm.py` calls the providers' HTTPS APIs directly with `httpx` (no SDKs):

| Provider | Default model | Key variable |
|---|---|---|
| Google Gemini | `gemini-3.8-flash` | `GEMINI_API_KEY` |
| OpenAI | `gpt-6-luna` | `OPENAI_API_KEY` |
| Anthropic | `claude-haiku-5-5` | `ANTHROPIC_API_KEY` |

Model IDs change often. **List models** in the sidebar shows the text models available to your
key, and picking one uses it. Defaults are in `configs/bridge.yaml`.

The key is typed into the sidebar (app only), or read from the environment variable or from
`.streamlit/secrets.toml` (which is git-ignored):

```toml
ANTHROPIC_API_KEY = "..."
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
uv run python -m bridge.run --dry-run --limit 3     # prompts only, to cache/bridge_prompts/
uv run python -m bridge.run                         # drafts for all 104 gap topics
uv run streamlit run streamlit_app.py               # review and approve, http://localhost:8501
uv run pytest tests/test_bridge.py tests/test_bridge_run.py
```

`bridge.run` options: `--provider` (default `default_provider` in the config, `anthropic`),
`--model` (default: the provider's model), `--limit N` (at most N topics still to do),
`--topics ID,ID`, `--force` (redo existing drafts) and `--dry-run` (write the prompts to
`cache/bridge_prompts/`, no key and no API calls).

- **Resumable.** Topics that already have a draft are skipped unless `--force`. Approved packs are
  never overwritten, even with `--force`.
- **Errors.** Rate limits (429), server errors (5xx) and network errors are retried 4 times with
  backoff (2, 4, 8, 16 s, or the server's `Retry-After`, at most 60 s). Other errors are listed at
  the end and the exit code is 1, so the run can simply be repeated.
- **Output.** One progress line per topic, then `processed/bridge/index.csv`.

## 5. Output files (`processed/bridge/`)

| File | Content |
|---|---|
| `<scert_topic_id>.md` | a draft (starting with the "unreviewed draft" line) or an approved pack (title line naming the unit, label and reviewer) |
| `<scert_topic_id>.json` | the generation and review record, with `status` (§2) |
| `index.csv` | one row per pack: topic, unit, title, label, score, confidence, status, provider, model, prompt version, NCERT sources, finish reason, tokens, generated time, and the automated checks: `missing_sections`, `empty_sections`, `cut_off`, `lesson_words`, `key_word_count`, `question_count` |

## 6. Verification and results

**Tests.** `tests/test_bridge.py` (32 tests) and `tests/test_bridge_run.py` (11 tests), with the
LLM mocked and the app run through Streamlit `AppTest` against a temporary output folder and an
empty secrets store:

- sources, truncation, the prompt for each label, and the NCERT rule (no NCERT passages below
  `t_partial`, mismatch warning, and a check that the committed `thresholds.json` and
  `gap_matrix.csv` agree for all 132 topics);
- each provider's request and response format, temperature omitted by default, thinking tokens,
  error messages without the key, the temperature retry, and retries on 429 / 5xx with
  `Retry-After` and backoff (no retry on 4xx, capped waits);
- drafts and approval: `save_draft` banner and status, approved packs never replaced, the app
  loading a saved draft and saving it as approved, edits surviving topic switches, regenerating
  clearing the approval, and the key never in a pack record;
- `bridge.run`: drafts and `index.csv` written, skip and `--force`, `--limit` counting only topics
  still to do, `--dry-run` making no calls and needing no key, key read from `secrets.toml` and
  never in any output, failures listed with exit code 1.

**Drafts generated** (2026-10-30, Anthropic `claude-haiku-5-5`, prompt v3): 103 drafts for the
104 gap topics (the Mass pack was already approved), 0 failures, about 226k input and 262k output
tokens (around US$0.16).

**Checks** (`eval/bridge_draft_check.md`, automated and AI checks, *not* a teacher review):

| Check | Result |
|---|---|
| Missing or empty sections, cut-off outputs, > 10 key words, > 5 questions | 0 of 103 |
| Lessons over 400 words | 3, regenerated; now 0 (lessons 122-394 words, median 250) |
| Not Covered drafts with NCERT sources | 0 of 50 |
| AI spot check of 10 drafts (5 Not Covered, 5 Partially, 10 units) | no wrong science facts and no content outside the passage; 8 drafts have small added explanations or qualifiers (e.g. "eats **only** plants", "longer than a classroom") that a teacher should check |

Short passages give short packs (12 drafts have fewer than 5 questions), as prompt v3 intends.

### Code review (prompt v2)

A review against the current provider docs and the app's state handling found and fixed:

| Severity | Finding | Fix |
|---|---|---|
| High | Regenerating kept the earlier approval, so an unreviewed draft could be saved at once | regenerating clears the approval |
| High | `temperature` 0.4 was sent to models that ignore it (Gemini 3.6+) or say to omit it (Claude Haiku 5.5) | not sent by default (§3) |
| Medium | Prompt v1 required 250-400 words and 8-10 key words even for very short passages, which invites invented content, and mixed "SCERT facts only" with NCERT links | prompt v2: all lengths are maxima, NCERT facts only under "What you already know", exercise questions in passages ignored |
| Low | The missing-sections check matched `## Heading` anywhere, including `###` headings | only a level-2 heading line counts |
| Low | Thinking tokens, which count against the output limit, were not recorded | `usage.thinking_tokens` |

Prompt v3 (Phase 4 review) adds the NCERT rule: no NCERT passages for Not Covered topics.

## 7. Known limitations

1. **Labels are provisional.** The packs trust the Phase 4 label. 63 of 132 labels are uncertain
   (`processed/gap/run_meta.json`), so a teacher should check the label as well as the lesson. A
   topic that moves across `t_partial` after a refit changes whether it gets NCERT passages.
2. **No automatic fact check.** The proposal's checker agent (back-translation, chrF++, reading
   level) is not built. The teacher's review is the only check, so packs must not be used
   unreviewed.
3. **English only.** Packs in the child's home language (REQ-06) were dropped at review: the
   class is taught in English, and machine-translated lessons could not be checked by the team.
4. **Diagrams are missing.** The passages are text only, and figure labels inside them are noisy
   (`docs/PREPROCESSING.md` §6).
5. **Drafts are not reviewed packs.** All drafts were written by one model (Claude Haiku 5.5)
   and only checked automatically and by an AI spot check (§6). None may be used with children
   until a teacher approves it in the app.
