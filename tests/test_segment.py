"""Tests for preprocessing.segment, on synthetic lines shaped like the real books."""

from preprocessing.extract import Line
from preprocessing.segment import despace, is_title_case, segment

CFG = {"heading_size_ratio": 1.10, "max_heading_chars": 60, "min_topic_chars": 300}
SENTENCE = "Children count the leaves in each group and write the total in the box. "


def body(page: int, n: int = 6, size: float = 14, conf: float | None = None) -> list[Line]:
    """n lines of ordinary body text."""
    return [Line(SENTENCE, size, False if conf is None else None, page, conf) for _ in range(n)]


def test_helpers():
    assert despace("G E O M E T R Y") == "Geometry"
    assert despace("Geometry") == "Geometry"
    assert is_title_case("Properties of 2D Shapes")
    assert not is_title_case("Name the following shapes")


def test_ncert_chapter_with_subtopic_and_box_labels():
    lines = [
        Line("Chapter", 14, True, 0),
        Line("Double Century", 32, True, 0),
        Line("Double Century", 32, True, 0),  # drawn twice for a shadow effect
        *body(0),
        Line("Let us Do", 18, True, 1),  # box label: not a topic
        *body(1),
        Line("129", 17, True, 1),  # bold page number: not a heading
        Line("How many are 100?", 16, True, 2),
        *body(2),
        Line("Reprint 2026-27", 10, False, 2),
    ]
    topics = segment(lines, "NCERT", CFG)
    assert [t.title for t in topics] == ["Double Century", "How many are 100?"]
    assert all(t.chapter == "Double Century" for t in topics)
    assert "Let us Do" in topics[0].body  # stays in the text; clean_text removes it
    assert (topics[1].page_start, topics[1].page_end) == (2, 2)


def test_ncert_prelims_without_title_are_skipped():
    assert segment(body(0) + body(1), "NCERT", CFG) == []


def test_scert_units_sections_and_front_matter():
    lines = [
        Line("CONTENT", 34, True, 0),  # front matter: before any unit
        *body(0),
        Line("G E O M E T R Y", 30, True, 4),
        Line("UNIT-1", 20, True, 4),
        Line("1.1 Properties of 2D Shapes", 16, True, 4),
        *body(4),
        Line("Exercise 1.1", 18, True, 5),  # not a topic
        *body(5),
        Line("1.2 Solid Shapes", 16, True, 6),
        *body(6),
        Line("N U M B E R S", 30, True, 9),
        Line("UNIT-2", 20, True, 9),
        Line("2.1 Place Value", 16, True, 9),
        *body(9),
    ]
    topics = segment(lines, "SCERT", CFG)
    assert [(t.chapter, t.title) for t in topics] == [
        ("Geometry", "Properties of 2D Shapes"),
        ("Geometry", "Solid Shapes"),
        ("Numbers", "Place Value"),
    ]
    assert "CONTENT" not in "".join(t.body for t in topics)


def test_scert_ocr_heading_with_garbled_badge():
    ocr = 9.0
    lines = [
        Line("UNIT-2", 13, None, 0, 90),
        Line("NUMBERS", 20, None, 0, 90),
        *body(0, size=ocr, conf=90),
        Line("EB Systematic Ordering", 11.5, None, 1, 88),
        *body(1, size=ocr, conf=90),
        Line("Name the following shapes", 11.5, None, 1, 90),  # instruction, not title case
        Line("lIl ~~ ;", 9, None, 1, 12),  # low-confidence junk, dropped
    ]
    topics = segment(lines, "SCERT", CFG)
    assert [t.title for t in topics] == ["Numbers", "Systematic Ordering"]
    assert "lIl" not in topics[1].body
    assert "Name the following shapes" in topics[1].body


def test_short_topics_are_merged():
    lines = [
        Line("Fair Share", 30, True, 0),
        *body(0),
        Line("Tiny Bit", 16, True, 1),
        Line("Only one line here.", 14, False, 1),
        Line("Big Part", 16, True, 2),
        *body(2),
    ]
    topics = segment(lines, "NCERT", CFG)
    assert [t.title for t in topics] == ["Fair Share", "Big Part"]
    assert "Tiny Bit" in topics[0].body and topics[0].page_end == 1


def test_wrapped_heading_lines_join():
    lines = [
        Line("Toy Joy", 30, True, 0),
        *body(0),
        Line("Clap", 16, True, 1),
        Line("Snap and Pat", 16, True, 1),
        *body(1),
    ]
    assert segment(lines, "NCERT", CFG)[1].title == "Clap Snap and Pat"


def test_scert_units_from_print_slugs_and_section_numbers():
    """OCR'd books lose the unit title art but keep the slug and numbered sections."""

    def page(n: int, unit: int, name: str, heading: str | None) -> list[Line]:
        top = [Line(heading, 12.5, None, n, 95)] if heading else []
        slug = Line(f"3rd_Unit_0{unit}_{name}_Term 2indd {n} & 8/8/2022 5:55", 9, None, n, 70)
        return [*top, *body(n, size=11.5, conf=95), slug]

    lines = [
        Line("CONTENT", 30, None, 0, 90),  # front matter: no slug
        *body(0, size=11.5, conf=95),
        *page(1, 1, "Numbers", "1.1 Symbol of multiplication"),
        *page(2, 1, "Numbers", "1.2 Dot multiplication:"),
        *page(3, 1, "Numbers", "Total number of apples = 5.5 45"),  # not a heading
        *page(4, 2, "Patterns", "2.1 Patterns in numbers ate"),
        *page(5, 2, "Patterns", "1.5 Stray number from another unit"),
    ]
    topics = segment(lines, "SCERT", CFG)
    assert [(t.chapter, t.title) for t in topics] == [
        ("Numbers", "Symbol of multiplication"),
        ("Numbers", "Dot multiplication"),
        ("Patterns", "Patterns in numbers"),
    ]
    assert topics[1].page_end == 3
    assert "Stray number" in topics[2].body


def test_scert_glossary_and_credits_are_dropped():
    tamil = "Magnet - காந்தம்"
    lines = [
        Line("Unit", 30, True, 0),
        Line("Magnetism", 20, True, 0),
        Line("1.1 Discovery of Magnets", 14, True, 0),
        *body(0),
        *[Line(tamil, 12, False, 1) for _ in range(12)],  # glossary page
        Line("Experts & Coordinators", 14, True, 2),  # credits page
        *body(2),
    ]
    topics = segment(lines, "SCERT", CFG)
    assert [t.title for t in topics] == ["Discovery of Magnets"]
    assert topics[0].page_end == 0


def test_end_heading_moves_above_text_printed_below_it():
    def at(text: str, y: float, bold: bool = False, size: float = 14) -> Line:
        return Line(text, size, bold, 3, bbox=(50.0, y, 400.0, y + 12))

    lines = [
        Line("Fun with Magnets", 30, True, 0),
        *body(0),
        Line("Let us enhance our learning", 16, True, 3, bbox=(50.0, 60.0, 400.0, 76.0)),
        at(SENTENCE * 3, 100),
        at(SENTENCE * 3, 120),
        at("Gather information on materials that can be recycled. " * 4, 300),  # below heading
        at("Learning further", 280, bold=True, size=16),  # stored after its own text
    ]
    topics = segment(lines, "NCERT", CFG)
    further = [t for t in topics if t.section == "extension"]
    assert len(further) == 1 and further[0].title == "Learning further"
    assert "Gather information" in further[0].body
    exercises = [t for t in topics if t.section == "exercises"][0]
    assert "Gather information" not in exercises.body


def test_two_end_headings_stored_after_their_text():
    """NCERT ch. 5: "Keywords" (top) and "Summary" (middle) both come last in the PDF."""

    def at(text: str, y: float, bold: bool = False, size: float = 14) -> Line:
        return Line(text, size, bold, 3, bbox=(100.0, y, 400.0, y + 12))

    lines = [
        Line("Measurement of Length", 30, True, 0),
        *body(0),
        at("The SI unit of length is metre and its symbol is m. " * 6, 450),  # summary text
        at("Measurement SI unit Reference point Oscillatory motion " * 6, 130),  # keywords
        at("Keywords", 80, bold=True, size=16),
        at("Summary", 400, bold=True, size=16),
    ]
    topics = segment(lines, "NCERT", CFG)
    by_title = {t.title: t for t in topics}
    assert "Reference point" in by_title["Keywords"].body
    assert "SI unit of length" in by_title["Summary"].body
    assert "SI unit of length" not in by_title["Keywords"].body


def test_verse_translation_and_attribution_removed():
    lines = [
        Line("Methods of Separation", 30, True, 0),
        Line("साधु ऐसा चाहिए", 18, True, 0),
        Line("—कबीर", 16, True, 0),
        Line("Just like winnowing makes the husk blow away", 12, False, 0),
        Line("the virtues stay while discarding the vices.", 12, False, 0),
        Line("—Kabir", 10, False, 0),
        *body(0),
    ]
    topics = segment(lines, "NCERT", CFG)
    text = "\n".join(t.body for t in topics)
    assert "winnowing" not in text and "Kabir" not in text
    assert "कबीर" not in text
    assert SENTENCE.strip() in text  # the chapter's own text stays
