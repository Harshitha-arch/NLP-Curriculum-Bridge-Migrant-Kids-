"""Stage 2: split a book's lines into topics.

What counts as a topic differs by board:

* NCERT (one PDF per chapter): the chapter title is the large text on the first page.
  Bold lines clearly larger than body text start a sub-topic ("Test for proteins").
  Recurring boxes ("Do you know?", "Activity 4.2") are not topics. The Sanskrit/Hindi
  verses NCERT prints as quotes are removed along with their English translations.
* SCERT (one PDF per term, several units): a unit starts on its opener page, which has
  a large standalone "Unit" (or "UNIT-<n>") line; the unit is named by the largest other
  line on that page. Books without openers fall back to the print slug on every page
  ("6th Science_IInd Term English Book_Unit_3.indd"), then to "UNIT-<n>" marks of any
  size. Numbered sections whose number
  matches the unit ("3.2 Mass, Shape and Volume of Solids") start topics. Where OCR
  garbles a number badge into a short token ("EB Systematic Ordering"), a title-cased
  line clearly larger than body text also starts a topic.

Both boards end chapters with fixed sections (NCERT "Summary", "Keywords", "Let us
enhance our learning", "Learning further"; SCERT "Points to Remember", "Evaluation",
"ICT Corner"). These start topics too, tagged with a ``section`` type so they can be
told apart from the teaching content.

Headings that wrap onto two lines are joined before any of this. Pages before the first
chapter/unit (covers, contents, preface) are front matter; the SCERT English-Tamil
glossary and the credits pages after it are back matter. Neither is returned. Topics
shorter than ``min_topic_chars`` are merged into a neighbour of the same section type.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace

from preprocessing.clean import BOX_LABEL, FURNITURE, looks_like_word
from preprocessing.extract import Line

# "2.2 Diffusion", "1. 1 Length" (space only before a one-digit number), "1.1.3 Symmetry"
SECTION_NUMBER = re.compile(r"^(\d{1,2})\.(?:\s?\d\b|\d{2}\b)(?:\.\d{1,2})?\.?\s+(?=[A-Za-z])")
UNIT_MARK = re.compile(r"^unit\s*[-–:]?\s*(\d+)\b", re.I)
UNIT_OPENER = re.compile(r"^unit(\s*[-–:]?\s*(\d+))?$", re.I)  # the whole line is "Unit"/"UNIT-2"
# "6th Science_EM_Unit 1.indd", "6th Science_IInd Term English Book_Unit_3.indd",
# "3rd_Unit_02_Patternsindd" (OCR loses the dot), "4th Maths Unit-1 Term 2 EM.indd"
SLUG = re.compile(
    r"Unit[-_\s]*(\d+)(?:[_\s]+([A-Za-z][A-Za-z_\s&-]*?))?(?:[_\s]+Term|\.?\s*indd)", re.I
)
MEDIUM_TAG = re.compile(r"\s+EM$", re.I)  # "Geometry_EM": English-medium edition marker
END_SECTIONS = [  # (pattern, section type), matched against a whole heading line
    (
        re.compile(r"^(summary|keywords|points to remember)(\s+(summary|keywords))?$", re.I),
        "summary",
    ),
    (re.compile(r"^(let us enhance our learning|evaluation|exercises?)$", re.I), "exercises"),
    (re.compile(r"^(learning further|ict corner)$", re.I), "extension"),
]
NOT_A_TITLE = re.compile(r"^(activity\b|table\b|fig(ure)?\b|[—–(\"'])", re.I)
DEVANAGARI = re.compile("[ऀ-ॿ]")
TAMIL = re.compile("[஀-௿]")
VERSE_ATTRIBUTION_MAX_SIZE = 10.5  # NCERT prints a verse's attribution in small type
VERSE_MAX_TRANSLATION_LINES = 8  # lines between a verse and its attribution, at most
GLOSSARY_MIN_TAMIL_LINES = 10  # a page with this many Tamil lines is an SCERT glossary page
SMALL_WORDS = {"a", "an", "and", "of", "the", "in", "on", "to", "by", "with", "for", "or", "at"}
TRAILING_WORDS = SMALL_WORDS | {"them", "it", "is", "are", "from", "into", "this", "that"}
SHORT_WORDS = {"A", "I", "AN", "IN", "ON", "TO", "MY", "WE", "UP", "NO", "IS", "IT", "OF", "AT"}
LETTER_RUN = re.compile(r"^(?:[A-Z] ){2,}(?=[A-Z][a-z])")  # "E N X T Symmetrical Designs"
OCR_ICON_DEBRIS = {"ate", "&", "®", "©", "@", "|", "§"}  # pencil/bell icons read as text
ACRONYMS = {"ICT", "DNA", "ORS", "LPG", "CNG", "UV", "SI"}  # kept upper case by despace
MIN_OCR_CONF = 40.0  # OCR lines below this mean confidence are treated as noise
MIN_OCR_HEADING_CONF = 60.0  # a heading must be read more reliably than body text
MAX_HEADING_WORDS = 8
NUMBERED_EXTRA_CHARS = 40  # numbered section titles may run longer than other headings
OPENER_SIZE_RATIO = 1.6  # a unit opener's "Unit" line is at least this much larger than body


@dataclass
class Topic:
    """A run of lines under one heading. ``body`` keeps one source line per text line.

    ``section`` is "content" for teaching text, or "summary" / "exercises" / "extension"
    for the fixed end-of-chapter sections.
    """

    chapter: str
    title: str
    page_start: int
    page_end: int
    section: str = "content"
    lines: list[str] = field(default_factory=list)

    @property
    def body(self) -> str:
        """The topic's raw text, newline-separated, ready for ``clean.clean_text``."""
        return "\n".join(self.lines)


@dataclass(frozen=True)
class Unit:
    """The chapter/unit a page belongs to. ``number`` is None when the book has none."""

    number: int | None
    name: str


# --- small text helpers -------------------------------------------------------------


def body_sizes(lines: list[Line]) -> dict[bool, float]:
    """Body text size per source kind (native vs OCR): the size covering most characters.

    The two kinds are measured on different scales (font points vs word heights), so
    each gets its own reference. Keys: True = native, False = OCR.
    """
    weight: dict[tuple[bool, float], int] = defaultdict(int)
    for line in lines:
        weight[(line.conf is None, round(line.size))] += len(line.text)
    best: dict[bool, tuple[int, float]] = {}
    for (native, size), chars in weight.items():
        if chars > best.get(native, (0, 0.0))[0]:
            best[native] = (chars, size)
    return {native: size for native, (_, size) in best.items()}


def is_title_case(text: str) -> bool:
    """True if at least 80% of the non-small words start with a capital letter."""
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z']*", text) if w.lower() not in SMALL_WORDS]
    return bool(words) and sum(w[0].isupper() for w in words) / len(words) >= 0.8


def despace(text: str) -> str:
    """Undo display typography: "G E O M E T R Y" -> "Geometry", "NUMBERS" -> "Numbers"."""
    text = text.strip()
    if re.fullmatch(r"(?:[A-Za-z] )+[A-Za-z]", text):
        return text.replace(" ", "").title()
    if text.isupper() and len(text) > 3:
        return " ".join(w if w in ACRONYMS else w.title() for w in text.split())
    return text


def is_badge_debris(token: str) -> bool:
    """True for what OCR makes of a coloured section-number badge: a short token with no
    lowercase letters ("EB", "(2.2)", "&") that is not a real word or "2D"/"3D"."""
    if re.fullmatch(r"\d[Dd]", token) or token.upper() in SHORT_WORDS:
        return False
    return len(token) <= 4 and not any(ch.islower() for ch in token)


def heading_text(line: Line) -> str:
    """The line's text without section number, OCR badge/icon debris or letter runs."""
    text = SECTION_NUMBER.sub("", line.text.strip())
    if line.conf is not None:
        words = text.split()
        if len(words) > 1 and is_badge_debris(words[0]):
            words = words[1:]
        while len(words) > 1 and (
            words[-1] in OCR_ICON_DEBRIS or not any(ch.isalnum() for ch in words[-1])
        ):
            words = words[:-1]
        text = " ".join(words)
    text = LETTER_RUN.sub("", text)
    return re.sub(r"\s+", " ", despace(text)).strip(" :.-")


def end_section(text: str) -> str | None:
    """The section type if the line is a fixed end-of-chapter heading, else None."""
    text = re.sub(r"\s+", " ", text.strip(" :."))
    for pattern, kind in END_SECTIONS:
        if pattern.match(text):
            return kind
    return None


# --- wrapped headings ---------------------------------------------------------------


def join_wrapped_headings(lines: list[Line], body: dict[bool, float], cfg: dict) -> list[Line]:
    """Join a bold heading line with the bold line(s) that continue it on the same page.

    "3.2 Mass, Shape and Volume of" + "Solids, Liquids and Gases" -> one line. A line
    continues the one before when both are bold, the same size (within 0.5 pt), on the
    same page, the first is a numbered section or larger than body text, the first is
    visibly unfinished (ends in a small word such as "of"/"and", a comma or a hyphen, or
    the second starts in lowercase), and the second does not start a new numbered section.
    """
    out: list[Line] = []
    for line in lines:
        prev = out[-1] if out else None
        prev_text = prev.text.rstrip() if prev else ""
        unfinished = (
            prev_text.endswith((",", "-", "&"))
            or (prev_text.split() or [""])[-1].lower() in TRAILING_WORDS
            or line.text.strip()[:1].islower()
        )
        if (
            prev is not None
            and prev.bold
            and line.bold
            and prev.page == line.page
            and line.text.strip() != prev_text.strip()  # shadow copy, not a continuation
            and abs(prev.size - line.size) <= 0.5
            and unfinished
            and not SECTION_NUMBER.match(line.text.strip())
            and not end_section(line.text)
            and not end_section(prev.text)
            and len(prev.text) + len(line.text) <= cfg["max_heading_chars"] + NUMBERED_EXTRA_CHARS
            and (
                SECTION_NUMBER.match(prev.text.strip())
                or prev.size >= cfg["heading_size_ratio"] * body.get(prev.conf is None, prev.size)
            )
        ):
            sep = "" if prev_text.endswith("-") else " "  # "Non-" + "biodegradable"
            out[-1] = replace(prev, text=f"{prev_text}{sep}{line.text.strip()}")
        else:
            out.append(line)
    return out


# --- headings ----------------------------------------------------------------------


def is_heading(
    line: Line, board: str, body: dict[bool, float], cfg: dict, unit_no: int | None
) -> bool:
    """Decide whether a line starts a new topic (see module docstring for the rules)."""
    text = line.text.strip()
    if end_section(text) and line.bold is not False:
        return True
    title = heading_text(line)
    words = title.split()
    if not words or BOX_LABEL.match(title) or re.match(r"^exercise\b", title, re.I):
        return False
    if any(p.search(text) for p in FURNITURE) or "|" in text or text.endswith((",", ";")):
        return False
    if DEVANAGARI.search(text) or NOT_A_TITLE.match(title) or title[0].islower():
        return False  # Sanskrit epigraph, its attribution, a box label or a fragment
    if not any(sum(ch.isalpha() for ch in w) >= 3 and looks_like_word(w) for w in words):
        return False  # page number, table header ("TTh", "T Th") or OCR junk
    if line.conf is not None and line.conf < MIN_OCR_HEADING_CONF:
        return False
    larger = line.size >= cfg["heading_size_ratio"] * body.get(line.conf is None, line.size)

    if board == "SCERT":
        numbered = SECTION_NUMBER.match(text)
        if numbered and line.bold is not False:
            same_unit = unit_no is None or int(numbered.group(1)) == unit_no
            return same_unit and len(text) <= cfg["max_heading_chars"] + NUMBERED_EXTRA_CHARS
        # OCR misread the number badge: "EB Systematic Ordering".
        first = text.split()[0]
        return (
            line.conf is not None
            and len(text.split()) > 1
            and is_badge_debris(first)
            and larger
            and is_title_case(title)
            and 2 <= len(words) <= MAX_HEADING_WORDS
            and not re.search(r"[\d=]", title)
        )

    if len(text) > cfg["max_heading_chars"] or text.endswith((".", ":")):
        return False
    if len(words) > MAX_HEADING_WORDS or words[-1].lower() in TRAILING_WORDS:
        return False  # an instruction sentence or a wrapped line, not a title
    if line.conf is None:  # native text: weight is known
        return bool(line.bold) and larger
    return larger and is_title_case(title)


# --- chapters / units ----------------------------------------------------------------


def chapter_title_ncert(lines: list[Line], body: dict[bool, float]) -> list[str]:
    """NCERT: the large (>= 1.6 x body) Latin-script lines on the first page, e.g.
    ["House of", "Hundreds - II"]. A title drawn twice (shadow effect) is listed once."""
    parts: list[str] = []
    for line in lines:
        if line.page != 0:
            break
        text = line.text.strip()
        if (
            line.size >= 1.6 * body.get(line.conf is None, line.size)
            and sum(ch.isalpha() for ch in text) >= 2
            and not DEVANAGARI.search(text)
            and not re.fullmatch(r"chapter", text, re.I)
            and text not in parts
        ):
            parts.append(text)
    return parts


def name_from_page(lines: list[Line], page: int) -> str | None:
    """The largest short, word-like, confidently read line on a page, other than the
    "Unit" mark itself: a unit's title on its opening page ("G E O M E T R Y" ->
    "Geometry"). None if nothing qualifies."""
    candidates = [
        line
        for line in lines
        if line.page == page
        and not UNIT_MARK.match(line.text.strip())
        and not UNIT_OPENER.match(line.text.strip())
        and not any(p.search(line.text) for p in FURNITURE)
        and 3 <= sum(ch.isalpha() for ch in line.text)
        and len(line.text) <= 40
        and (line.conf is None or line.conf >= 80)
        and all(looks_like_word(w) for w in despace(line.text).split())
    ]
    if not candidates:
        return None
    name = despace(max(candidates, key=lambda line: line.size).text)
    return re.sub(r"^[\d.\])\s]+", "", name) or None  # "1.1] Iterative ..." -> "Iterative ..."


def units_from_openers(
    lines: list[Line], body: dict[bool, float], min_ratio: float = OPENER_SIZE_RATIO
) -> dict[int, Unit]:
    """SCERT: page -> unit for each unit opener page: a standalone "Unit"/"UNIT-<n>" line
    at least ``min_ratio`` times body size (0 accepts any size).

    The number comes from the mark ("UNIT-2"), else the page's slug, else counts on from
    the previous unit. A page with several marks is a contents page and is skipped.
    """
    slug_numbers = {
        line.page: int(match.group(1)) for line in lines if (match := SLUG.search(line.text))
    }
    by_page: dict[int, list[Line]] = defaultdict(list)
    for line in lines:
        by_page[line.page].append(line)
    units: dict[int, Unit] = {}
    previous = 0
    for page, page_lines in sorted(by_page.items()):
        marks = [
            match
            for line in page_lines
            if (match := UNIT_OPENER.match(line.text.strip()))
            and line.size >= min_ratio * body.get(line.conf is None, line.size)
        ]
        if len(marks) != 1:
            continue  # no unit starts here, or a contents page listing several units
        number = int(marks[0].group(2) or slug_numbers.get(page, previous + 1))
        units[page] = Unit(number, name_from_page(page_lines, page) or f"Unit {number}")
        previous = number
    return units


def units_from_slugs(lines: list[Line]) -> dict[int, Unit]:
    """SCERT fallback: page -> unit, read from print slugs ("..._Unit_02_Numbers_Term 3").

    OCR misspells a slug now and then ("Pattems"), so each unit number is named by the
    spelling seen on most of its pages. Slugs without a name take it from the unit's
    first page (see ``name_from_page``).
    """
    numbers: dict[int, int] = {}
    names: dict[int, Counter[str]] = defaultdict(Counter)
    for line in lines:
        match = SLUG.search(line.text)
        if match:
            number = int(match.group(1))
            numbers[line.page] = number
            name = MEDIUM_TAG.sub("", re.sub(r"[_\s]+", " ", match.group(2) or "").strip())
            if name:
                names[number][name.title()] += 1
    for number in set(numbers.values()) - set(names):
        first_page = min(page for page, n in numbers.items() if n == number)
        names[number][name_from_page(lines, first_page) or f"Unit {number}"] += 1
    return {
        page: Unit(number, names[number].most_common(1)[0][0]) for page, number in numbers.items()
    }


def fill_forward(units: dict[int, Unit], last_page: int) -> dict[int, Unit]:
    """Give every page after the first known unit page the unit of the page before it,
    so pages without their own marker stay in their unit."""
    filled: dict[int, Unit] = {}
    current = None
    for page in range(last_page + 1):
        current = units.get(page, current)
        if current is not None:
            filled[page] = current
    return filled


# --- page clean-up -----------------------------------------------------------------


def drop_verses(lines: list[Line]) -> list[Line]:
    """Remove the Sanskrit/Hindi verses that NCERT prints as decorative quotes, together
    with their English translation and attribution (the team does not need them).

    A verse is a run of lines containing Devanagari (blank icon lines in between are part
    of it). After the run, lines are dropped up to and including the first small-print
    line on the same page (font at most ``VERSE_ATTRIBUTION_MAX_SIZE``): the attribution,
    such as "(Wise saying)", "-Kabir" or "(Rasaratnasamuchchaya-10.3)". If no such line
    follows within ``VERSE_MAX_TRANSLATION_LINES`` lines, only the Devanagari is dropped.
    """
    out: list[Line] = []
    i = 0
    while i < len(lines):
        if not DEVANAGARI.search(lines[i].text):
            out.append(lines[i])
            i += 1
            continue
        page = lines[i].page
        while (
            i < len(lines)
            and lines[i].page == page
            and (DEVANAGARI.search(lines[i].text) or not any(ch.isalnum() for ch in lines[i].text))
        ):
            i += 1  # the verse itself
        for j in range(i, min(i + VERSE_MAX_TRANSLATION_LINES, len(lines))):
            if lines[j].page != page:
                break
            text = lines[j].text
            if lines[j].size <= VERSE_ATTRIBUTION_MAX_SIZE and any(ch.isalnum() for ch in text):
                i = j + 1  # translation and attribution
                break
    return out


def back_matter_start(lines: list[Line]) -> int | None:
    """The first page of a book's back matter: the first page with at least
    ``GLOSSARY_MIN_TAMIL_LINES`` Tamil-script lines. In the SCERT books that is the
    English-Tamil glossary, and only credits pages follow it. None if there is none."""
    counts: Counter[int] = Counter(line.page for line in lines if TAMIL.search(line.text))
    pages = [page for page, n in counts.items() if n >= GLOSSARY_MIN_TAMIL_LINES]
    return min(pages) if pages else None


def place_end_headings(lines: list[Line]) -> list[Line]:
    """Put end-of-chapter headings ("Summary", "Learning further") where they are printed.

    Some PDFs store these headings after the text printed below them, which would file
    that text under the previous section. On a page where an end heading comes after a
    line printed lower down, the page is regrouped: each line goes under the nearest
    heading printed above it, and lines printed above the first heading stay in front
    (they finish the previous section). Each group keeps its stored order. Pages that are
    already in order, and lines without a bounding box, are left alone. Used for NCERT
    only: it assumes one column, and SCERT's two-column pages are stored in order.
    """
    by_page: dict[int, list[Line]] = defaultdict(list)
    for line in lines:
        by_page[line.page].append(line)
    out: list[Line] = []
    for page_lines in by_page.values():
        heads = [ln for ln in page_lines if ln.bbox is not None and end_section(ln.text)]
        out_of_order = any(
            other.bbox is not None and other.bbox[1] > head.bbox[1]
            for head in heads
            for other in page_lines[: page_lines.index(head)]
        )
        if not out_of_order:
            out.extend(page_lines)
            continue
        heads.sort(key=lambda head: head.bbox[1])
        front: list[Line] = []
        groups: list[list[Line]] = [[] for _ in heads]
        for line in page_lines:
            if line in heads:
                continue
            above = [i for i, h in enumerate(heads) if line.bbox and line.bbox[1] >= h.bbox[1]]
            (groups[above[-1]] if above else front).append(line)
        out.extend(front)
        for head, group in zip(heads, groups, strict=True):
            out.extend([head, *group])
    return out


# --- segmentation ------------------------------------------------------------------


def merge_short(topics: list[Topic], min_chars: int) -> list[Topic]:
    """Fold topics whose body is shorter than ``min_chars`` into the previous topic of the
    same chapter and section type (or the next one, if the short one comes first)."""
    merged: list[Topic] = []
    for topic in topics:
        prev = merged[-1] if merged else None
        same = prev and (prev.chapter, prev.section) == (topic.chapter, topic.section)
        if prev and same and len(topic.body) < min_chars:
            prev.lines.extend([topic.title, *topic.lines])
            prev.page_end = max(prev.page_end, topic.page_end)
            if topic.section != "content" and topic.title not in prev.title:
                prev.title = f"{prev.title} and {topic.title}"  # "Summary and Keywords"
        elif prev and same and len(prev.body) < min_chars:
            lead = [] if prev.title == prev.chapter else [prev.title]
            topic.lines = [*lead, *prev.lines, *topic.lines]
            topic.page_start = prev.page_start
            if topic.section != "content":
                topic.title = f"{prev.title} and {topic.title}"
            merged[-1] = topic
        else:
            merged.append(topic)
    return merged


def segment(lines: list[Line], board: str, cfg: dict) -> list[Topic]:
    """Split one book's lines into topics. Front matter is dropped (see module docstring)."""
    lines = [line for line in lines if line.conf is None or line.conf >= MIN_OCR_CONF]
    lines = drop_verses(lines)
    back = back_matter_start(lines)
    if back is not None:
        lines = [line for line in lines if line.page < back]  # glossary and credits
    if not lines:
        return []
    body = body_sizes(lines)
    lines = join_wrapped_headings(lines, body, cfg)
    if board == "NCERT":  # SCERT pages are two-column and already in order
        lines = place_end_headings(lines)
    last_page = max(line.page for line in lines)
    if board == "NCERT":
        title_parts = chapter_title_ncert(lines, body)
        if not title_parts:
            return []  # no chapter title: prelims file (foreword, contents)
        units = fill_forward({0: Unit(None, " ".join(title_parts))}, last_page)
    else:
        title_parts = []
        # Large openers first; slugs when OCR lost the opener art; small marks last.
        found = (
            units_from_openers(lines, body)
            or units_from_slugs(lines)
            or units_from_openers(lines, body, min_ratio=0)
        )
        units = fill_forward(found, last_page)
    # Lines that only repeat the chapter/unit name (shadow copies, running headers).
    skip = {*title_parts, *(unit.name for unit in units.values())}

    topics: list[Topic] = []
    current: Topic | None = None
    unit: Unit | None = None
    for line in lines:
        if line.page not in units:
            continue  # front matter before the first chapter/unit
        if units[line.page] != unit:
            unit = units[line.page]
            current = Topic(unit.name, unit.name, line.page, line.page)
            topics.append(current)
        assert current is not None
        text = line.text.strip()
        if text in skip or despace(text) in skip or UNIT_OPENER.match(text):
            continue
        if is_heading(line, board, body, cfg, unit.number):
            title = heading_text(line)
            section = end_section(text) or (
                "content"
                if current.section == "content" or SECTION_NUMBER.match(text)
                else current.section
            )
            if (
                not current.lines
                and current.title != current.chapter
                and section == current.section
            ):
                if title not in current.title:  # heading wrapped onto 2+ lines
                    current.title = f"{current.title} {title}"
            else:
                current = Topic(unit.name, title, line.page, line.page, section)
                topics.append(current)
            continue
        current.lines.append(text)
        current.page_end = line.page
    return merge_short(topics, cfg["min_topic_chars"])
