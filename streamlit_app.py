"""Curriculum Bridge app: generate, review and export bridge packs for SCERT gap topics.

Usage:
    uv run streamlit run streamlit_app.py

The LLM API key is typed into the sidebar or read from GEMINI_API_KEY / OPENAI_API_KEY /
ANTHROPIC_API_KEY (environment or .streamlit/secrets.toml). It is never written to disk.
Approved packs are saved to processed/bridge/ (settings: configs/bridge.yaml).
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import streamlit as st
import yaml

from bridge import llm
from bridge.data import (
    APPROVED,
    DRAFT,
    LABELS,
    NO_NCERT,
    filter_topics,
    generation_record,
    get_sources,
    load_gap_matrix,
    load_matches,
    load_pack,
    load_t_partial,
    load_topics,
    pack_stem,
    save_pack,
)
from bridge.prompts import (
    PROMPT_VERSION,
    build_prompt,
    missing_sections,
    pack_markdown,
)

ROOT = Path(__file__).resolve().parent
CFG = yaml.safe_load((ROOT / "configs" / "bridge.yaml").read_text(encoding="utf-8"))
PATHS = {k: ROOT / v for k, v in CFG["paths"].items()}
if os.environ.get("BRIDGE_OUTPUT_DIR"):  # tests save to a temporary folder
    PATHS["output_dir"] = Path(os.environ["BRIDGE_OUTPUT_DIR"])

st.set_page_config(page_title="Curriculum Bridge", page_icon="📘", layout="wide")


@st.cache_data
def load_data():
    return (
        load_gap_matrix(PATHS["gap_matrix"]),
        load_topics(PATHS["topics"]),
        load_matches(PATHS["topic_matches"]),
        load_t_partial(PATHS["thresholds"]),
    )


def env_key(name: str) -> str:
    """API key from the environment or .streamlit/secrets.toml, if set."""
    if os.environ.get(name):
        return os.environ[name]
    try:
        return str(st.secrets.get(name, ""))
    except Exception:  # no secrets file
        return ""


def pick_model(provider: str) -> None:
    st.session_state[f"model::{provider}"] = st.session_state[f"pick::{provider}"]


def keep_edit(key: str, edit_key: str) -> None:
    """Store edits with the pack, since Streamlit drops widget state when the topic changes."""
    st.session_state["packs"][key]["edited"] = st.session_state[edit_key]


gap, topics, matches, t_partial = load_data()
st.session_state.setdefault("packs", {})

# ---------------------------------------------------------------- sidebar: model settings
with st.sidebar:
    st.header("Model")
    providers = CFG["providers"]
    provider = st.selectbox(
        "Provider",
        list(providers),
        index=list(providers).index(CFG["default_provider"]),
        format_func=lambda p: providers[p]["label"],
    )
    pcfg = providers[provider]
    saved_key = env_key(pcfg["env"])
    typed_key = st.text_input(
        "API key",
        type="password",
        key=f"key::{provider}",
        placeholder=f"Using {pcfg['env']}" if saved_key else "Paste your key",
        help=f"Or set {pcfg['env']} in your environment or .streamlit/secrets.toml.",
    )
    api_key = typed_key.strip() or saved_key
    st.session_state.setdefault(f"model::{provider}", pcfg["model"])
    model = st.text_input("Model", key=f"model::{provider}").strip()
    if st.button("List models", disabled=not api_key, help="Models available to this key"):
        try:
            st.session_state[f"models::{provider}"] = llm.list_models(provider, api_key)
        except llm.LLMError as e:
            st.error(str(e))
    available = st.session_state.get(f"models::{provider}")
    if available:
        st.selectbox(
            "Available models",
            available,
            index=available.index(model) if model in available else None,
            key=f"pick::{provider}",
            on_change=pick_model,
            args=(provider,),
            placeholder="Pick one to use it",
        )
    st.caption("The key is kept only for this browser session and is never saved.")

# ---------------------------------------------------------------- main: pick a topic
st.title("Curriculum Bridge")
st.caption(
    "Bridge lessons for Tamil Nadu SCERT Class 6 Science topics that NCERT *Curiosity* "
    "does not fully cover. Every pack is generated only from the two textbooks and must be "
    "checked by a teacher before it is exported."
)

c1, c2 = st.columns([2, 1])
labels = c1.multiselect("Gap labels", LABELS, default=["Not Covered", "Partially Covered"])
units = list(dict.fromkeys(gap["scert_unit"]))
unit = c2.selectbox("Unit", ["All units", *units])
rows = filter_topics(gap, labels, None if unit == "All units" else unit)
if rows.empty:
    st.info("No topics match these filters.")
    st.stop()

options = rows["scert_topic_id"].tolist()
names = {
    r.scert_topic_id: f"{r.scert_unit} · {r.scert_topic_title} ({r.label}, {r.score:.2f})"
    for r in rows.itertuples()
}
topic_id = st.selectbox(f"Topic ({len(options)})", options, format_func=names.get)
src = get_sources(
    topic_id,
    gap,
    topics,
    matches,
    n_ncert=CFG["sources"]["ncert_matches"],
    max_scert_chars=CFG["sources"]["max_scert_chars"],
    max_ncert_chars=CFG["sources"]["max_ncert_chars"],
    t_partial=t_partial,
)

color = {"Not Covered": "red", "Partially Covered": "orange", "Covered": "green"}[src.label]
closest = f"*{src.ncert[0].chapter} › {src.ncert[0].title}*" if src.ncert else NO_NCERT
st.markdown(
    f":{color}-badge[{src.label}] &nbsp; coverage score **{src.score:.3f}** · "
    f"label confidence **{src.confidence}** · closest NCERT section: {closest}"
)
if src.label_mismatch:
    st.warning(
        f"The gap matrix says {src.label}, but the score {src.score:.4f} is "
        f"{'below' if src.score < t_partial else 'at or above'} t_partial {t_partial} in "
        "thresholds.json. The two files are out of sync: re-run classification.run."
    )

with st.expander("Textbook passages sent to the model"):
    st.markdown(f"**SCERT · {src.scert.chapter} · {src.scert.title}**")
    st.write(src.scert.text)
    for p in src.ncert:
        st.markdown(f"**NCERT · {p.chapter} · {p.title}** (coverage {p.coverage:.3f})")
        st.write(p.text)
    if not src.ncert:
        st.markdown(f"**NCERT:** {NO_NCERT} (score below t_partial {t_partial}).")

# ---------------------------------------------------------------- generate
key = topic_id
edit_key = f"edit::{key}"
system, user = build_prompt(src)

if st.button(
    "Generate bridge pack",
    type="primary",
    disabled=not (api_key and model),
    help=None if api_key else "Add an API key in the sidebar first.",
):
    with st.spinner(f"Writing the pack with {model}..."):
        try:
            gen = llm.generate(
                provider,
                api_key,
                model,
                system,
                user,
                temperature=CFG["generation"]["temperature"],
                max_tokens=CFG["generation"]["max_tokens"],
                timeout=CFG["generation"]["timeout_s"],
            )
        except llm.LLMError as e:
            st.error(str(e))
        else:
            meta = generation_record(
                src,
                provider,
                gen,
                PROMPT_VERSION,
                CFG["generation"]["temperature"],
                CFG["paths"]["thresholds"],
            )
            st.session_state["packs"][key] = {"text": gen.text, "meta": meta | {"status": DRAFT}}
            st.session_state[edit_key] = gen.text
            # A new draft has not been reviewed: clear any earlier approval of this pack.
            st.session_state.pop(f"approved::{key}", None)

# ---------------------------------------------------------------- review and export
if key not in st.session_state["packs"] and (saved := load_pack(PATHS["output_dir"], topic_id)):
    body_on_disk, record = saved
    st.session_state["packs"][key] = {"text": body_on_disk, "meta": record, "from_disk": True}
pack = st.session_state["packs"].get(key)
if pack:
    meta = pack["meta"]
    if pack.get("from_disk"):
        when = meta.get("saved_at") or meta.get("generated_at", "")
        if meta.get("status") == APPROVED:
            st.success(
                f"Loaded the approved pack (reviewed by {meta.get('approved_by')}, {when}). "
                "Saving again replaces it."
            )
        else:
            st.info(
                f"Loaded the saved draft from {meta.get('model')} ({when}). It is unreviewed: "
                "check it against the passages before approving."
            )
    st.session_state.setdefault(edit_key, pack.get("edited", pack["text"]))
    if gaps := missing_sections(st.session_state[edit_key]):
        st.warning("Missing sections: " + ", ".join(gaps) + ". Regenerate or add them.")
    if str(meta.get("finish_reason")).upper() in {"MAX_TOKENS", "LENGTH"}:
        st.warning("The model hit its output limit, so the pack may be cut off.")

    preview, edit = st.tabs(["Preview", "Edit"])
    with edit:
        st.text_area(
            "Edit the pack (Markdown)",
            key=edit_key,
            height=520,
            on_change=keep_edit,
            args=(key, edit_key),
        )
    body = st.session_state[edit_key]
    with preview:
        st.markdown(body)

    st.subheader("Teacher review")
    r1, r2 = st.columns([1, 2])
    reviewer = r1.text_input("Reviewed by", key=f"reviewer::{key}").strip()
    checked = r2.checkbox(
        "I checked every fact against the textbook passages and the language suits the child.",
        key=f"approved::{key}",
    )
    approved = bool(reviewer and checked)
    final = pack_markdown(src, body, reviewer or "-")

    b1, b2, _ = st.columns([1, 1, 2])
    b1.download_button(
        "Download .md",
        final,
        file_name=f"{pack_stem(topic_id)}.md",
        mime="text/markdown",
        disabled=not approved,
    )
    if b2.button("Save to processed/bridge", disabled=not approved):
        approved_meta = meta | {
            "approved_by": reviewer,
            "edited_by_reviewer": body.strip() != pack["text"].strip(),
            "saved_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        saved = save_pack(PATHS["output_dir"], topic_id, final, approved_meta)
        pack["meta"] = approved_meta | {"status": APPROVED}
        st.success(f"Saved {saved} as approved")
    if not approved:
        st.caption("Export unlocks once a reviewer is named and the check box is ticked.")

    tokens = meta.get("usage") or {}
    st.caption(
        f"{meta.get('provider')} · {meta.get('model')} · prompt {meta.get('prompt_version')} · "
        f"tokens in/out: {tokens.get('input_tokens')}/{tokens.get('output_tokens')}"
    )

with st.expander("Prompt sent to the model"):
    st.code(system, language=None, wrap_lines=True)
    st.code(user, language=None, wrap_lines=True)

st.caption(
    "Gap labels come from processed/gap/gap_matrix.csv (Phase 4). They are provisional: "
    "the thresholds were fitted on partly AI-assisted labels (docs/CLASSIFICATION.md)."
)
