"""Tests for the batch draft generator (bridge.run) and LLM retries, with the LLM mocked."""

import json
from pathlib import Path

import httpx
import pandas as pd
import pytest
import yaml

from bridge import llm
from bridge import run as batch
from bridge.data import save_draft, save_pack
from bridge.prompts import SECTIONS

ROOT = Path(__file__).resolve().parents[1]
KEY = "sk-ant-test-secret-987"


@pytest.fixture
def setup(tmp_path, monkeypatch):
    """A config that reads the real Phase 3/4 outputs but writes to tmp_path."""
    cfg = yaml.safe_load((ROOT / "configs/bridge.yaml").read_text(encoding="utf-8"))
    for k, v in cfg["paths"].items():
        cfg["paths"][k] = str(ROOT / v)
    out = tmp_path / "bridge"
    cfg["paths"]["output_dir"] = str(out)
    cfg["paths"]["secrets"] = str(tmp_path / "secrets.toml")
    cfg["paths"]["prompt_cache"] = str(tmp_path / "prompts")
    path = tmp_path / "bridge.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    calls = []

    def fake_generate(provider, api_key, model, system, user, **kw):
        calls.append((provider, api_key, model))
        text = "\n".join(
            f"## {s}\n" + ("one two three" if s == "Lesson" else "x") for s in SECTIONS
        )
        return llm.Generation(text, model, "end_turn", {"input_tokens": 10, "output_tokens": 20})

    monkeypatch.setattr(llm, "generate", fake_generate)
    gap = pd.read_csv(ROOT / "processed/gap/gap_matrix.csv")
    nc = gap.loc[gap["label"] == "Not Covered", "scert_topic_id"].tolist()
    pc = gap.loc[gap["label"] == "Partially Covered", "scert_topic_id"].tolist()
    return {"cfg": str(path), "out": out, "calls": calls, "nc": nc, "pc": pc, "tmp": tmp_path}


def run(setup, *args):
    return batch.main(["--config", setup["cfg"], *args])


def test_writes_drafts_and_index(setup, monkeypatch, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    topics = f"{setup['nc'][0]},{setup['pc'][0]}"
    assert run(setup, "--topics", topics) == 0
    assert [c[2] for c in setup["calls"]] == ["claude-haiku-5-5"] * 2
    md = (setup["out"] / f"{setup['nc'][0]}.md").read_text(encoding="utf-8")
    assert md.startswith("> **Unreviewed draft from claude-haiku-5-5.**")
    meta = json.loads((setup["out"] / f"{setup['nc'][0]}.json").read_text(encoding="utf-8"))
    assert meta["status"] == "draft" and meta["ncert_sources"] == []  # Not Covered: no NCERT
    assert meta["ncert_rule"].startswith("none") and meta["t_partial"] is not None
    index = pd.read_csv(setup["out"] / "index.csv")
    assert list(index.columns) == batch.INDEX_COLUMNS and len(index) == 2
    row = index.set_index("topic_id").loc[setup["nc"][0]]
    assert row["status"] == "draft" and row["lesson_words"] == 3 and row["key_word_count"] == 0
    assert pd.isna(row["missing_sections"])
    out = capsys.readouterr().out
    assert out.count("ok, 10/20 tokens") == 2


def test_skips_existing_drafts_unless_force_and_never_touches_approved(setup, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    draft, done = setup["nc"][0], setup["nc"][1]
    save_draft(setup["out"], draft, "## Lesson\nold", {"model": "m"})
    save_pack(setup["out"], done, "approved text", {"approved_by": "Ms Rao", "model": "m"})
    before = (setup["out"] / f"{done}.md").read_text(encoding="utf-8")

    assert run(setup, "--topics", f"{draft},{done}") == 0
    assert setup["calls"] == []  # draft skipped, approved skipped
    assert run(setup, "--topics", f"{draft},{done}", "--force") == 0
    assert len(setup["calls"]) == 1  # only the draft is redone
    assert "## What you already know" in (setup["out"] / f"{draft}.md").read_text(encoding="utf-8")
    assert (setup["out"] / f"{done}.md").read_text(encoding="utf-8") == before
    meta = json.loads((setup["out"] / f"{done}.json").read_text(encoding="utf-8"))
    assert meta["status"] == "approved"


def test_save_draft_refuses_to_replace_approved(tmp_path):
    save_pack(tmp_path, "S1", "approved", {"approved_by": "Ms Rao"})
    with pytest.raises(FileExistsError):
        save_draft(tmp_path, "S1", "new", {"model": "m"})


def test_limit_counts_only_topics_still_to_do(setup, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    save_draft(setup["out"], setup["nc"][0], "## Lesson\nold", {"model": "m"})
    assert run(setup, "--limit", "2") == 0
    assert len(setup["calls"]) == 2
    old = (setup["out"] / f"{setup['nc'][0]}.md").read_text(encoding="utf-8")
    assert "What you already know" not in old  # the existing draft was skipped, not redone


def test_dry_run_makes_no_calls_and_needs_no_key(setup):
    assert run(setup, "--dry-run", "--limit", "3") == 0
    assert setup["calls"] == []
    prompts = sorted((setup["tmp"] / "prompts").glob("*.txt"))
    assert len(prompts) == 3 and "SCERT passage" in prompts[0].read_text(encoding="utf-8")
    assert not setup["out"].exists()


def test_key_from_secrets_file_never_in_outputs(setup, capsys):
    secrets = setup["tmp"] / "secrets.toml"
    secrets.write_text(f'ANTHROPIC_API_KEY = "{KEY}"\n', encoding="utf-8")
    assert run(setup, "--topics", setup["pc"][0]) == 0
    assert setup["calls"][0][1] == KEY  # read from secrets.toml
    assert KEY not in capsys.readouterr().out
    for f in setup["out"].iterdir():
        assert KEY not in f.read_text(encoding="utf-8"), f.name


def test_missing_key_stops_before_any_call(setup, capsys):
    assert run(setup, "--limit", "1") == 2
    assert setup["calls"] == [] and "No API key" in capsys.readouterr().out


def test_failures_are_listed_and_exit_nonzero(setup, monkeypatch, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    bad = setup["pc"][1]

    # the prompt names the topic by its title, so the mock fails on that title
    title = pd.read_csv(ROOT / "processed/gap/gap_matrix.csv").set_index("scert_topic_id")
    bad_title = title.loc[bad, "scert_topic_title"]

    def flaky_title(provider, api_key, model, system, user, **kw):
        if f"Topic: {bad_title} " in user:
            raise llm.LLMError("anthropic API error 400: bad request", 400)
        return llm.Generation("\n".join(f"## {s}\nx" for s in SECTIONS), model, "end_turn", {})

    monkeypatch.setattr(llm, "generate", flaky_title)
    assert run(setup, "--topics", f"{setup['pc'][0]},{bad}") == 1
    out = capsys.readouterr().out
    assert "1 failed:" in out and f"{bad}: anthropic API error 400" in out
    assert (setup["out"] / f"{setup['pc'][0]}.md").exists()  # kept going


# ---------------------------------------------------------------- retries in llm.generate


def anthropic_ok():
    return httpx.Response(
        200,
        json={
            "model": "m",
            "content": [{"type": "text", "text": "## Lesson\nok"}],
            "stop_reason": "end_turn",
            "usage": {},
        },
    )


def test_retries_429_respecting_retry_after_then_5xx_backoff():
    replies = [
        httpx.Response(429, headers={"retry-after": "7"}, json={"error": {"message": "slow"}}),
        httpx.Response(529, json={"error": {"message": "overloaded"}}),
        anthropic_ok(),
    ]
    waits = []
    client = httpx.Client(transport=httpx.MockTransport(lambda r: replies.pop(0)))
    gen = llm.generate(
        "anthropic", KEY, "m", "s", "u", client=client, retries=4, sleep=waits.append
    )
    assert gen.text == "## Lesson\nok" and waits == [7.0, 4]


def test_no_retry_on_client_errors_or_when_retries_is_zero():
    waits = []

    def bad(request):
        return httpx.Response(400, json={"error": {"message": "bad"}})

    client = httpx.Client(transport=httpx.MockTransport(bad))
    with pytest.raises(llm.LLMError) as e:
        llm.generate("anthropic", KEY, "m", "s", "u", client=client, retries=4, sleep=waits.append)
    assert e.value.status == 400 and waits == [] and KEY not in str(e.value)

    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(503)))
    with pytest.raises(llm.LLMError):
        llm.generate("anthropic", KEY, "m", "s", "u", client=client, sleep=waits.append)
    assert waits == []


def test_gives_up_after_retries_and_caps_the_wait():
    waits = []
    resp = httpx.Response(429, headers={"retry-after": "500"}, json={"error": {"message": "x"}})
    client = httpx.Client(transport=httpx.MockTransport(lambda r: resp))
    with pytest.raises(llm.LLMError) as e:
        llm.generate("anthropic", KEY, "m", "s", "u", client=client, retries=2, sleep=waits.append)
    assert e.value.status == 429 and waits == [60.0, 60.0]
