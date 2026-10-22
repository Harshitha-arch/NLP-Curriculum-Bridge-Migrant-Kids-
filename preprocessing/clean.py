"""Stage 3: clean topic text, normalise Hindi, cap length, and flag empty/noisy topics.

Everything here is a pure function on strings so it can be unit-tested without PDFs.
"""

from __future__ import annotations

import re
import unicodedata

# --- character-level normalisation ------------------------------------------------

# Typographic spaces, quotes and dashes -> plain equivalents. Private-use code points
# U+F0xx come from the Symbol font in some PDFs; the low byte is the Symbol glyph.
CHAR_MAP = str.maketrans(
    {
        **{c: " " for c in "           \t"},
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        "−": "-",
        "­": "",  # soft hyphen
        "": "+",
        "": "-",
        "": "=",
        "": "×",  # multiplication sign
        "": "÷",  # division sign
        "�": "",  # replacement character: the original glyph is unrecoverable
    }
)
PRIVATE_USE = re.compile("[-]")  # any other unmapped icon/bullet glyph
CONTROL = re.compile("[\x00-\x08\x0b-\x1f\x7f-\x9f]")  # all control chars except \t \n
BULLETS = re.compile("[•●○■▪‣⁃➢✓✔]")

# --- Hindi (Devanagari) normalisation ----------------------------------------------

DEVANAGARI = "ऀ-ॿ"
ZERO_WIDTH = re.compile("[​‌‍⁠﻿]")
DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
# OCR and typists often write the danda (।) as a pipe or "l", and the double danda as "||".
PIPE_DANDA = re.compile(rf"(?<=[{DEVANAGARI}])\s*(\|\||॥)")
SINGLE_DANDA = re.compile(rf"(?<=[{DEVANAGARI}])\s*[|।](?!\|)")
# The same vowel sign / virama / nukta typed twice in a row is always an error.
DOUBLED_MARK = re.compile("([़ा-्ॢॣ])\\1+")


def normalize_hindi(text: str) -> str:
    """Normalise Devanagari text so equal words get equal code points.

    1. Unicode NFC: canonical composition. Note NFC keeps nukta letters such as
       U+0958 (क़) decomposed as क + ़ (they are composition exclusions), so both
       spellings end up identical.
    2. Remove zero-width characters (ZWSP/ZWJ/ZWNJ/BOM). ZWJ/ZWNJ only change how a
       conjunct is drawn, not the word, so dropping them merges visual variants.
    3. Devanagari digits ०-९ -> ASCII 0-9, so numbers match across languages.
    4. Pipe or doubled pipe after Devanagari -> danda (।) / double danda (॥), with
       no space before it.
    5. Collapse a vowel sign, virama or nukta repeated back-to-back.

    Latin text passes through unchanged apart from NFC and step 2.
    """
    text = unicodedata.normalize("NFC", text)
    text = ZERO_WIDTH.sub("", text)
    text = text.translate(DEVANAGARI_DIGITS)
    text = PIPE_DANDA.sub("॥", text)
    text = SINGLE_DANDA.sub("।", text)
    return DOUBLED_MARK.sub(r"\1", text)


# --- line-level noise ----------------------------------------------------------------

FURNITURE = [
    re.compile(r"^reprint\s+\d{4}\s*-\s*\d{2,4}$", re.I),  # NCERT footer
    re.compile(r"indd\b", re.I),  # InDesign slug: "3rd_Unit_01_Geometry_Term 1.indd 7"
    re.compile(r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\s+\d{1,2}:\d{2}"),  # print timestamp
    re.compile(r"^(not to be republished|©.*|isbn.*)$", re.I),
    re.compile(r"^maths? mela(\s+class\s+\d+)?$", re.I),  # NCERT running header
    re.compile(r"^\*?pictures are indicative only\.?$", re.I),  # SCERT image disclaimer
    re.compile(r"^chapter(\s+\d+)?$", re.I),  # NCERT chapter label next to the title
    # NCERT side-margin label, extracted as one line or three
    re.compile(
        r"^(curiosity|textbook of science|grade \d+)(\s*\|?\s*(textbook of science|grade \d+))*$",
        re.I,
    ),
]
# SCERT glossaries pair English terms with Tamil, which the PDFs' legacy font often
# garbles (stray ">" and "M" inside words). Tamil is outside this project's languages
# (English, Hindi), so Tamil-script words are removed and the English terms kept.
TAMIL_WORD = re.compile(r"\(?[^\s]*[஀-௿][^\s]*\)?")
OCR_ICON = re.compile(r"(?<!\S)(T{3,}|[®©]+)(?!\S)")  # pencil/bell icons read as "TTT ®"
# Box labels that repeat in every chapter and say nothing about the topic.
BOX_LABEL = re.compile(
    r"^(let\s*(us|'s)\s+\w+(\s+(at home|out|together|more))?|"  # Let us Do / Observe / Find
    r"try (these|this|it|it out|yourself)|do it yourself|activity|exercise(\s*\d+(\.\d+)?)?|"
    r"practice|teacher'?s note|notes?|do you know|think and (answer|discuss)|answers?|"
    r"worksheet|project( work)?|fun time|i can do|let me do|find out|check\W*\s*check|"
    r"definition|recall|remember|example\s*\d*|solution|travel through|"
    # science books: box titles, often split over two lines ("More to" / "know!")
    r"more to( know)?|know|do you|learning objectives|introduction|ict corner|"
    r"steps?(\s*\d+)?|aim|activity(\s*\d+(\.\d+)?)?|url|caution|know a scientist|"
    r"thinking skills|materials? required)\W*$",
    re.I,
)
FILL_IN = re.compile(r"_{2,}|\.{3,}|…+|(\.\s){3,}")
RUPEE_OCR = re.compile(r"(?<![\w%])[%~](?=\s?\d)")  # Tesseract reads ₹ as % or ~


def clean_title(raw: str) -> str:
    """Normalise a heading for display: the same character fixes as ``clean_text``, but
    without dropping lines, so titles such as "ICT Corner" survive."""
    text = normalize_hindi(raw).translate(CHAR_MAP)
    text = CONTROL.sub("", PRIVATE_USE.sub(" ", BULLETS.sub(" ", text)))
    return re.sub(r"\s+", " ", text).strip()


def keep_line(line: str) -> bool:
    """True if a raw line carries content (not a footer, page number, label or stray glyph).

    A line needs at least two letters, except that a short expression containing "="
    (e.g. "2 × 4 = 8") is kept.
    """
    line = line.strip()
    if not line or any(p.search(line) for p in FURNITURE) or BOX_LABEL.match(line):
        return False
    letters = sum(ch.isalpha() for ch in line)
    return letters >= 2 or ("=" in line and len(line) >= 5)


def join_lines(lines: list[str]) -> str:
    """Join wrapped lines into running text, undoing end-of-line hyphenation.

    "pic-" + "tures" -> "pictures"; any other break becomes a single space.
    """
    out = ""
    for line in lines:
        if out.endswith("-") and line[:1].islower():
            out = out[:-1] + line
        else:
            out = f"{out} {line}" if out else line
    return out


def clean_text(raw: str) -> str:
    """Turn raw extracted text (newline-separated lines) into clean single-paragraph text.

    Order: Unicode/Hindi normalisation -> per-character mapping -> drop noise lines ->
    remove fill-in blanks, bullets, table rules -> join lines -> fix OCR rupee sign ->
    collapse whitespace and repeated punctuation.
    """
    text = normalize_hindi(raw).translate(CHAR_MAP)
    text = CONTROL.sub("", PRIVATE_USE.sub(" ", BULLETS.sub(" ", text)))
    text = TAMIL_WORD.sub(" ", text)
    lines = []
    for line in text.splitlines():
        line = OCR_ICON.sub(" ", FILL_IN.sub(" ", line).replace("|", " "))
        line = re.sub(r"\s+", " ", line).strip(" ,;")
        line = re.sub(r"\s+([,.;:?!।])", r"\1", line)  # as the final tidy-up does, so
        if keep_line(line):  # a second pass judges the same line (idempotence)
            lines.append(line)
    text = RUPEE_OCR.sub("₹", join_lines(lines))
    # Each pass only shortens the text, so this reaches a fixed point; repeating it is
    # what makes clean_text idempotent ("XX." + "..XX" would otherwise form a new "...").
    while True:
        tidy = re.sub(r"\s+([,.;:?!।])", r"\1", text)  # no space before punctuation
        tidy = re.sub(r"([,;:!?])\1+", r"\1", FILL_IN.sub(" ", tidy))
        tidy = re.sub(r"\s{2,}", " ", tidy).strip()
        if tidy == text:
            return text
        text = tidy


# --- length cap ---------------------------------------------------------------------

SENTENCE_END = re.compile(r"[.?!।](?=\s|$)")


def cap_length(text: str, max_chars: int) -> tuple[str, bool]:
    """Cap text at ``max_chars``, cutting at a sentence end where possible.

    Cut point: the last sentence end in [0.6 * max_chars, max_chars]; failing that,
    the last space before max_chars; failing that, max_chars exactly.
    Returns (text, was_truncated). The result is never longer than ``max_chars``.
    """
    if len(text) <= max_chars:
        return text, False
    window = text[: max_chars + 1]
    ends = [m.end() for m in SENTENCE_END.finditer(window) if m.end() <= max_chars]
    if ends and ends[-1] >= 0.6 * max_chars:
        return window[: ends[-1]].strip(), True
    cut = window.rfind(" ", 0, max_chars + 1)
    cut = cut if cut > 0 else max_chars
    return window[:cut].strip(), True


# --- language and quality -----------------------------------------------------------

DEV_LETTER = re.compile(f"[{DEVANAGARI}]")
TOKEN_STRIP = "\"'()[]{}.,;:!?।॥-*/+=×÷<>"
VOWELS = set("aeiouyAEIOUY")
# Measurement units and place-value headers have no vowels but are real tokens here.
ABBREVIATIONS = {"kg", "g", "mg", "l", "ml", "km", "m", "cm", "mm", "hr", "hrs", "h", "min",
                 "s", "sec", "rs", "th", "tth", "hto", "tho"}  # fmt: skip
OPTION_LABEL = re.compile(r"^\(?[a-hA-H]\)[.,]?$")  # (b)  c)


def detect_lang(text: str) -> str:
    """Return "hi" (mostly Devanagari), "mixed" (>5% Devanagari letters) or "en"."""
    letters = sum(ch.isalpha() for ch in text)
    if not letters:
        return "en"
    share = len(DEV_LETTER.findall(text)) / letters
    return "hi" if share > 0.6 else "mixed" if share > 0.05 else "en"


def looks_like_word(token: str) -> bool:
    """Heuristic for OCR garbage: a real token is a number, a Devanagari word, or a
    Latin word that has a vowel and is not mixed-case junk like "tHe" or "IIlI"."""
    if OPTION_LABEL.match(token):
        return True
    token = token.strip(TOKEN_STRIP)
    if not token:
        return True  # pure punctuation, judged elsewhere
    if "-" in token:  # "3-km", "five-year": judge each part
        return all(looks_like_word(part) for part in token.split("-"))
    if re.fullmatch(r"₹?\d+([.,:/]\d+)*%?(st|nd|rd|th)?", token):
        return True
    if DEV_LETTER.search(token) or token.lower() in ABBREVIATIONS:
        return True
    token = re.sub(r"'s$", "", token)  # possessive: "Sam's"
    if not token.isalpha():
        return False
    if len(token) == 1:
        return token in "aAI"
    if not set(token) & VOWELS:
        return False
    return token.islower() or token.isupper() or token.istitle()


def assess(text: str, cfg: dict, ocr: bool) -> str | None:
    """Return why a cleaned topic should be dropped, or None to keep it.

    Checks, in order: empty; shorter than ``min_kept_chars``; for OCR'd topics only,
    letter share of non-space characters below ``min_alpha_ratio`` (native text that is
    mostly digits is a genuine arithmetic topic such as a times table; OCR text that is
    mostly digits and symbols is usually a misread figure); share of word-like tokens
    below ``min_word_quality``.
    """
    if not text.strip():
        return "empty"
    if len(text) < cfg["min_kept_chars"]:
        return "too_short"
    chars = [ch for ch in text if not ch.isspace()]
    if ocr and sum(ch.isalpha() for ch in chars) / len(chars) < cfg["min_alpha_ratio"]:
        return "low_alpha"
    tokens = text.split()
    if sum(looks_like_word(t) for t in tokens) / len(tokens) < cfg["min_word_quality"]:
        return "noisy_ocr"
    return None
