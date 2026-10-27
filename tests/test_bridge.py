"""Tests for Phase 5 bridge content: sources, prompt, LLM client (mocked), saving, app."""

import json
from pathlib import Path

import httpx
import pandas as pd
import pytest

from bridge import llm
from bridge.data import (
    filter_topics,
    get_sources,
    load_gap_matrix,
    load_matches,
    load_topics,
    save_pack,
    truncate,
)
from bridge.prompts import (
    SECTIONS,
    build_prompt,
    missing_sections,
    pack_markdown,
)

ROOT = Path(__file__).resolve().parents[1]
KEY = "sk-test-secret-123"


def topic(tid, board, title, text, chapter="Ch"):
    return {
        "topic_id": tid,
        "board": board,
        "chapter": chapter,
        "topic_title": title,
        "section": "content",
        "text": text,
    }


@pytest.fixture
def corpus():
    topics = {
        "S1": topic("S1", "SCERT", "Cell", "Cells are small. " * 20, "The Cell"),
        "S2": topic("S2", "SCERT", "Magnets", "Magnets attract iron.", "Magnetism"),
        "N1": topic("N1", "NCERT", "Living things", "Living things grow."),
        "N2": topic("N2", "NCERT", "Plants", "Plants make food."),
        "N3": topic("N3", "NCERT", "Animals", "Animals move."),
    }
    gap = pd.DataFrame(
        [
            {
                "scert_topic_id": "S1",
                "scert_unit": "The Cell",
                "scert_topic_title": "Cell",
                "score": 0.61,
                "label": "Not Covered",
                "confidence": "stable",
            },
            {
                "scert_topic_id": "S2",
                "scert_unit": "Magnetism",
                "scert_topic_title": "Magnets",
                "score": 0.86,
                "label": "Covered",
                "confidence": "stable",
            },
        ]
    )
    matches = pd.DataFrame(
        [
            {
                "scert_topic_id": "S1",
                "rank": r,
                "ncert_topic_id": n,
                "ncert_chapter": "Ch",
                "ncert_topic_title": topics[n]["topic_title"],
                "coverage": c,
            }
            for r, n, c in [(2, "N2", 0.55), (1, "N1", 0.60), (3, "N3", 0.50)]
        ]
    )
    return gap, topics, matches


# ---------------------------------------------------------------- data


def test_get_sources_takes_top_ncert_matches_in_rank_order(corpus):
    gap, topics, matches = corpus
    src = get_sources("S1", gap, topics, matches, n_ncert=2)
    assert [p.topic_id for p in src.ncert] == ["N1", "N2"]
    assert src.label == "Not Covered" and src.scert.title == "Cell"
    assert src.ncert[0].coverage == pytest.approx(0.60)


def test_get_sources_unknown_topic(corpus):
    with pytest.raises(KeyError):
        get_sources("nope", *corpus)


def test_truncate_cuts_at_sentence_end():
    text = "One two three. Four five six. Seven eight nine."
    out = truncate(text, 35)
    assert out == "One two three. Four five six. [...]"
    assert truncate(text, 1000) == text


def test_filter_topics_by_label_and_unit(corpus):
    gap, _, _ = corpus
    assert filter_topics(gap, ["Not Covered", "Partially Covered"])["scert_topic_id"].tolist() == [
        "S1"
    ]
    assert filter_topics(gap, ["Covered"], unit="The Cell").empty


def test_real_outputs_fit_together():
    """The committed Phase 3/4 outputs have the columns and ids the app relies on."""
    gap = load_gap_matrix(ROOT / "processed/gap/gap_matrix.csv")
    topics = load_topics(ROOT / "processed/topics.jsonl")
    matches = load_matches(ROOT / "processed/alignment/bge-large-en-v1.5/topic_matches.csv")
    tid = gap.loc[gap["label"] == "Not Covered", "scert_topic_id"].iloc[0]
    src = get_sources(tid, gap, topics, matches)
    assert src.scert.text and len(src.ncert) == 2
    assert all(t in topics for t in gap["scert_topic_id"])


# ---------------------------------------------------------------- prompt


@pytest.mark.parametrize("language", ["English", "Hindi"])
def test_prompt_contains_passages_rules_and_headings(corpus, language):
    src = get_sources("S1", *corpus)
    system, user = build_prompt(src, language)
    assert all(f"## {s}" in system for s in SECTIONS)
    assert "Use only facts that appear in the SCERT passage" in system
    assert "Cells are small." in user and "Living things grow." in user
    assert "teach it from the start" in user  # Not Covered guidance
    assert ("Devanagari" in user) == (language == "Hindi")


def test_prompt_rejects_unknown_language(corpus):
    with pytest.raises(ValueError):
        build_prompt(get_sources("S1", *corpus), "French")


def test_missing_sections():
    full = "\n".join(f"## {s}\ntext" for s in SECTIONS)
    assert missing_sections(full) == []
    assert missing_sections(full.replace("## Answers", "Answers")) == ["Answers"]
    # Only a heading line counts, not "## Answers" mid-line or a level-3 heading.
    assert missing_sections(full.replace("## Answers", "See ## Answers")) == ["Answers"]
    assert missing_sections(full.replace("## Answers", "### Answers")) == ["Answers"]
    # A translation after the English heading is fine (Hindi packs).
    assert missing_sections(full.replace("## Lesson", "## Lesson (पाठ)")) == []


def test_pack_markdown_has_title_and_reviewer(corpus):
    src = get_sources("S1", *corpus)
    md = pack_markdown(src, "## Lesson\nHi", "English", "Ms Rao")
    assert md.startswith("# Cell\n") and "reviewed by Ms Rao" in md and md.endswith("Hi\n")


# ---------------------------------------------------------------- LLM client (mocked HTTP)


def mock_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


RESPONSES = {
    "gemini": {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "thinking", "thought": True}, {"text": "## Lesson\nHello"}]
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
    },
    "openai": {
        "model": "m",
        "choices": [{"message": {"content": "## Lesson\nHello"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    },
    "anthropic": {
        "model": "m",
        "content": [{"type": "text", "text": "## Lesson\nHello"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    },
}


@pytest.mark.parametrize("provider", llm.PROVIDERS)
def test_generate_request_and_parse(provider):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = request.headers
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=RESPONSES[provider])

    gen = llm.generate(
        provider, KEY, "m", "SYS", "USER", temperature=0.4, client=mock_client(handler)
    )
    assert gen.text == "## Lesson\nHello"  # Gemini thought parts are dropped
    assert gen.usage["input_tokens"] == 10 and gen.usage["output_tokens"] == 5
    assert KEY not in seen["url"]
    assert KEY in " ".join(seen["headers"].values())
    body = json.dumps(seen["body"])
    assert "SYS" in body and "USER" in body
    if provider == "openai":
        assert "temperature" not in seen["body"]
        assert seen["body"]["max_completion_tokens"] == 8192
    else:
        assert "temperature" in body


@pytest.mark.parametrize("provider", llm.PROVIDERS)
def test_generate_omits_temperature_by_default(provider):
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=RESPONSES[provider])

    llm.generate(provider, KEY, "m", "s", "u", client=mock_client(handler))
    assert "temperature" not in json.dumps(bodies[0])


def test_thinking_tokens_are_reported():
    gemini = json.loads(json.dumps(RESPONSES["gemini"]))
    gemini["usageMetadata"]["thoughtsTokenCount"] = 900
    openai = json.loads(json.dumps(RESPONSES["openai"]))
    openai["usage"]["completion_tokens_details"] = {"reasoning_tokens": 700}
    assert llm.parse_response("gemini", "m", gemini).usage["thinking_tokens"] == 900
    assert llm.parse_response("openai", "m", openai).usage["thinking_tokens"] == 700


def test_generate_error_hides_key():
    def handler(request):
        return httpx.Response(401, json={"error": {"message": "Invalid API key"}})

    with pytest.raises(llm.LLMError) as e:
        llm.generate("anthropic", KEY, "m", "s", "u", client=mock_client(handler))
    assert "401" in str(e.value) and "Invalid API key" in str(e.value)
    assert KEY not in str(e.value)


def test_generate_retries_without_temperature():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        if "temperature" in bodies[-1]:
            return httpx.Response(400, json={"error": {"message": "temperature is not supported"}})
        return httpx.Response(200, json=RESPONSES["anthropic"])

    gen = llm.generate(
        "anthropic", KEY, "m", "s", "u", temperature=0.4, client=mock_client(handler)
    )
    assert gen.text and len(bodies) == 2 and "temperature" not in bodies[1]


def test_generate_empty_text_is_an_error():
    def handler(request):
        return httpx.Response(
            200, json={"candidates": [{"content": {"parts": []}, "finishReason": "SAFETY"}]}
        )

    with pytest.raises(llm.LLMError, match="SAFETY"):
        llm.generate("gemini", KEY, "m", "s", "u", client=mock_client(handler))


def test_generate_needs_key():
    with pytest.raises(llm.LLMError):
        llm.generate("gemini", "", "m", "s", "u")


def test_list_models_filters_non_text():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "models": [
                    {
                        "name": "models/gemini-x-flash",
                        "supportedGenerationMethods": ["generateContent"],
                    },
                    {
                        "name": "models/gemini-x-flash-tts",
                        "supportedGenerationMethods": ["generateContent"],
                    },
                    {"name": "models/text-embed", "supportedGenerationMethods": ["embedContent"]},
                ]
            },
        )

    assert llm.list_models("gemini", KEY, client=mock_client(handler)) == ["gemini-x-flash"]


# ---------------------------------------------------------------- saving


def test_save_pack_requires_reviewer(tmp_path):
    with pytest.raises(ValueError):
        save_pack(tmp_path, "S1", "English", "text", {"approved_by": " "})
    md = save_pack(tmp_path, "S1", "Hindi", "text", {"approved_by": "Ms Rao", "model": "m"})
    assert md.name == "S1.hindi.md" and md.read_text(encoding="utf-8") == "text\n"
    meta = json.loads((tmp_path / "S1.hindi.json").read_text(encoding="utf-8"))
    assert meta["approved_by"] == "Ms Rao"


# ---------------------------------------------------------------- app (LLM patched out)


def test_app_generate_review_flow(monkeypatch):
    from streamlit.testing.v1 import AppTest

    calls = []

    def fake_generate(provider, api_key, model, system, user, **kw):
        calls.append((provider, api_key, model))
        return llm.Generation(
            "\n".join(f"## {s}\nok" for s in SECTIONS),
            model,
            "STOP",
            {"input_tokens": 1, "output_tokens": 2},
        )

    monkeypatch.setattr(llm, "generate", fake_generate)
    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=60).run()
    assert not at.exception
    generate = next(b for b in at.button if b.label == "Generate bridge pack")
    assert generate.disabled  # no key yet

    at.text_input(key="key::gemini").set_value("k").run()
    next(b for b in at.button if b.label == "Generate bridge pack").click().run()
    assert not at.exception and calls == [("gemini", "k", "gemini-3.8-flash")]
    assert not at.warning  # all sections present
    save = next(b for b in at.button if b.label == "Save to processed/bridge")
    assert save.disabled  # not approved yet


def test_app_keeps_edits_and_resets_approval_on_regenerate(monkeypatch):
    """Edits survive switching topics; a regenerated draft must be approved again."""
    from streamlit.testing.v1 import AppTest

    n = []

    def fake_generate(provider, api_key, model, system, user, **kw):
        n.append(1)
        return llm.Generation("\n".join(f"## {s}\nv{len(n)}" for s in SECTIONS), model, "STOP", {})

    monkeypatch.setattr(llm, "generate", fake_generate)
    gap = load_gap_matrix(ROOT / "processed/gap/gap_matrix.csv")
    a, b = gap.loc[gap["label"] == "Not Covered", "scert_topic_id"].iloc[:2]
    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=60).run()

    def topic():
        return next(s for s in at.selectbox if s.label.startswith("Topic"))

    def generate():
        next(x for x in at.button if x.label == "Generate bridge pack").click().run()

    at.text_input(key="key::gemini").set_value(KEY).run()
    topic().set_value(a).run()
    generate()
    edit, approved = f"edit::{a}::English", f"approved::{a}::English"
    at.text_area(key=edit).set_value("my edit").run()
    topic().set_value(b).run()
    topic().set_value(a).run()
    assert at.text_area(key=edit).value == "my edit"
    assert not at.checkbox(key=approved).value  # approval does not carry over

    at.text_input(key=f"reviewer::{a}::English").set_value("Ms Rao").run()
    at.checkbox(key=approved).check().run()
    generate()
    assert "v2" in at.text_area(key=edit).value  # regenerating replaces the edit
    assert not at.checkbox(key=approved).value
    assert next(x for x in at.button if x.label == "Save to processed/bridge").disabled
    assert not at.exception
    assert KEY not in json.dumps(at.session_state["packs"], default=str)  # never in the record
