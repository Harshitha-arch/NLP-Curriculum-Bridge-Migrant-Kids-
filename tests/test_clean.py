"""Tests for preprocessing.clean (pure string functions)."""

from hypothesis import given
from hypothesis import strategies as st

from preprocessing.clean import (
    assess,
    cap_length,
    clean_text,
    clean_title,
    detect_lang,
    join_lines,
    keep_line,
    looks_like_word,
    normalize_hindi,
)

QUALITY = {"min_kept_chars": 150, "min_alpha_ratio": 0.55, "min_word_quality": 0.7}


# --- Hindi normalisation ---------------------------------------------------------


def test_hindi_digits_become_ascii():
    assert normalize_hindi("कक्षा ३ में १२ बच्चे") == "कक्षा 3 में 12 बच्चे"


def test_pipe_after_devanagari_becomes_danda():
    assert normalize_hindi("यह मेरा घर है |") == "यह मेरा घर है।"
    assert normalize_hindi("श्लोक ||") == "श्लोक॥"


def test_pipe_after_latin_is_untouched():
    assert normalize_hindi("a | b") == "a | b"


def test_zero_width_and_doubled_matra_removed():
    assert normalize_hindi("कि‍ताब") == "किताब"
    assert normalize_hindi("राम") == normalize_hindi("राा‌म")


def test_precomposed_nukta_matches_decomposed():
    assert normalize_hindi("क़") == normalize_hindi("क़")


@given(st.text())
def test_normalize_hindi_is_idempotent(text):
    once = normalize_hindi(text)
    assert normalize_hindi(once) == once


@given(st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126)))
def test_normalize_hindi_leaves_ascii_alone(text):
    assert normalize_hindi(text) == text


# --- line filtering and cleaning ---------------------------------------------------


def test_furniture_and_labels_are_dropped():
    for line in [
        "Reprint 2026-27",
        "3rd_Unit_01_Geometry_Term 1.indd 7",
        "12/11/2024 3:41:26 PM",
        "Let us Do",
        "Exercise 1.1",
        "42",
        "a)",
    ]:
        assert not keep_line(line), line
    assert keep_line("ox")
    assert keep_line("2 × 4 = 8")


def test_join_lines_dehyphenates():
    assert join_lines(["estimate the pic-", "tures to the nearest", "Tens place."]) == (
        "estimate the pictures to the nearest Tens place."
    )
    assert join_lines(["Fifty-", "Five"]) == "Fifty- Five"  # capital: real hyphen kept


def test_clean_text_end_to_end():
    raw = (
        "What’s in a Name?\nLet us Do\nThe cost of a pen is % 32. Find the\n"
        "cost of 6 pens __________\n17\nReprint 2026-27\n5th_Unit_02.indd 17"
    )
    assert clean_text(raw) == "What's in a Name? The cost of a pen is ₹ 32. Find the cost of 6 pens"


def test_running_header_and_ocr_icons_removed():
    assert clean_text("Maths Mela Class 3\nTTT ® Multiply the ones ®") == "Multiply the ones"


def test_tamil_glossary_words_removed_english_kept():
    raw = "Abacus (அபாகஸ்) கணினி\nComputer (கணினி) மண்்டMம்"
    assert clean_text(raw) == "Abacus Computer"


def test_science_furniture_removed():
    raw = "Chapter\n*Pictures are indicative only\nMagnets attract iron."
    assert clean_text(raw) == "Magnets attract iron."


def test_clean_title_keeps_labels():
    assert clean_title("ICT Corner") == "ICT Corner"
    assert clean_text("ICT Corner") == ""


def test_clean_text_maps_symbol_font_glyphs():
    assert clean_text("2  4  8 apples") == "2 × 4 = 8 apples"


@given(st.lists(st.text(alphabet="abc XYZ.,-|_%~12’\n", max_size=40), max_size=8))
def test_clean_text_is_idempotent(lines):
    once = clean_text("\n".join(lines))
    assert clean_text(once) == once


# --- length cap ---------------------------------------------------------------------


def test_cap_prefers_sentence_end():
    text = "A" * 700 + ". " + "B" * 300 + ". " + "C" * 400
    capped, truncated = cap_length(text, 1200)
    assert truncated and capped.endswith(".") and len(capped) == 1003


def test_cap_falls_back_to_word_boundary():
    text = " ".join(["word"] * 400)
    capped, truncated = cap_length(text, 1200)
    assert truncated and len(capped) <= 1200 and capped.endswith("word")


def test_short_text_untouched():
    assert cap_length("short.", 1200) == ("short.", False)


@given(st.text(), st.integers(min_value=1, max_value=2000))
def test_cap_never_exceeds_limit_and_is_a_prefix(text, limit):
    capped, truncated = cap_length(text, limit)
    assert len(capped) <= limit
    assert text.startswith(capped) or text.strip().startswith(capped)
    assert truncated == (len(text) > limit)


# --- language and quality -------------------------------------------------------------


def test_detect_lang():
    assert detect_lang("Count the apples") == "en"
    assert detect_lang("सेब गिनो") == "hi"
    assert detect_lang("Count the apples and write सेब in the box") == "mixed"
    assert detect_lang("123") == "en"


def test_word_heuristic():
    assert all(looks_like_word(t) for t in ["the", "Tens", "₹32", "1st", "किताब", "I", "a", "HTO"])
    assert not any(looks_like_word(t) for t in ["tHe", "xkcd", "q", "3a4b", "lIlIl"])


def test_assess_reasons():
    good = "The cost of a pen is ₹ 32. Find the cost of six pens and estimate it. " * 3
    assert assess(good, QUALITY, ocr=True) is None
    assert assess("", QUALITY, ocr=False) == "empty"
    assert assess("Too short.", QUALITY, ocr=False) == "too_short"
    drill = "Times table: 12 + 34 = 46, 7 x 8 = 56. " * 10
    assert assess(drill, QUALITY, ocr=True) == "low_alpha"
    assert assess(drill, QUALITY, ocr=False) is None  # native arithmetic is real content
    assert assess("tHe xkcd qwrt plmn zzkk Hq tR " * 10, QUALITY, ocr=True) == "noisy_ocr"


def test_ncert_side_label_removed():
    for label in ["Curiosity | Textbook of Science | Grade 6", "Curiosity", "Textbook of Science"]:
        assert not keep_line(label), label
    assert clean_text("Curiosity\nTextbook of Science\nGrade 6\nMagnets attract iron.") == (
        "Magnets attract iron."
    )


# --- chunking -----------------------------------------------------------------------

from preprocessing.clean import split_chunks  # noqa: E402


def test_short_topic_is_one_chunk():
    assert split_chunks("They attract iron.", "Magnets", 1200) == ["Magnets. They attract iron."]
    assert split_chunks("Magnets attract iron.", "Magnets", 1200) == ["Magnets attract iron."]


def test_long_topic_split_at_sentences_evenly():
    body = " ".join(f"Sentence number {i} is about how water evaporates." for i in range(120))
    chunks = split_chunks(body, "Evaporation", 1200)
    assert len(chunks) == 6  # ceil(~6,200 / ~1,187) chunks
    assert all(c.startswith("Evaporation. ") and len(c) <= 1200 for c in chunks)
    assert all(c.endswith(".") for c in chunks)  # cut at sentence ends
    sizes = [len(c) for c in chunks]
    assert max(sizes) - min(sizes) < 200  # no tiny leftover chunk


@given(
    st.lists(
        st.text(alphabet="abcde fgh.?!XYZ", min_size=1, max_size=300), min_size=1, max_size=40
    ),
    st.integers(min_value=60, max_value=1500),
)
def test_chunks_fit_and_lose_nothing(parts, limit):
    body = " ".join(" ".join(parts).split())
    title = "Topic"
    chunks = split_chunks(body, title, limit)
    assert all(len(c) <= limit for c in chunks)
    prefix = f"{title}. "
    first_bare = body.startswith(title)  # a first chunk already starting with the title
    assert all(c.startswith(prefix) for i, c in enumerate(chunks) if i or not first_bare)
    rebuilt = " ".join(
        c if i == 0 and first_bare else c[len(prefix) :] for i, c in enumerate(chunks)
    )
    assert rebuilt.split() == body.split()


def test_later_chunks_get_title_even_if_text_starts_with_it():
    body = "Mass is the measure of matter. " + "The SI unit of mass is kilogram. " * 60
    chunks = split_chunks(body.strip(), "Mass", 1200)
    assert len(chunks) > 1
    assert chunks[0].startswith("Mass is the measure")  # title not repeated
    assert all(c.startswith("Mass. ") for c in chunks[1:])


def test_shifted_font_labels_decoded_only_when_known():
    from preprocessing.clean import build_vocab

    vocab = build_vocab(["All water on earth. The skull and ribs. All water, earth, skull."])
    raw = "6NXOO\n$OO\x03ZDWHU\x03RQ\x03HDUWK\nWATER"
    assert clean_text(raw, vocab) == "Skull All water on earth WATER"
    assert clean_text("6NXOO", None) == "6NXOO"  # no vocabulary: left alone
    assert clean_text("XQNQRZQ", vocab) == "XQNQRZQ"  # decodes to an unknown word: left alone
