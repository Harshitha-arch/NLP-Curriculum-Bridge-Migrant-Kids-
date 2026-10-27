"""Prompt for one bridge pack: a short lesson, key words and practice questions."""

from __future__ import annotations

import re

from bridge.data import TopicSources

PROMPT_VERSION = "v1"
LANGUAGES = ["English", "Hindi"]

SECTIONS = [
    "What you already know",
    "Lesson",
    "Key words",
    "Practice questions",
    "Answers",
    "Note for the teacher",
]

SYSTEM = """You write bridge lessons for Class 6 children who have moved from a state that \
follows the NCERT science textbook ("Curiosity") to an English-medium school in Tamil Nadu that \
follows the Tamil Nadu SCERT textbook. The child may be behind and may not read English well.

Rules:
- Use only facts that appear in the SCERT passage. Do not add facts, numbers, dates, names, \
experiments or examples that are not in the passages. If the passage is unclear or incomplete, \
leave that part out and mention it in the note for the teacher.
- The NCERT passages show what the child has probably studied already. Use them only as a \
starting point the child can connect to. Do not reteach them.
- Write for a child of about 11: short sentences, everyday words, one idea at a time. Explain \
every science word the first time you use it.
- Ignore figure labels, page furniture and broken fragments in the passages.

Output Markdown with exactly these six level-2 headings, in this order:
## What you already know
1-3 sentences that link the topic to something the child knows from NCERT or daily life.
## Lesson
A lesson a teacher can read out in about 10 minutes: 250-400 words in short paragraphs, with \
one everyday example taken from the passage.
## Key words
A bullet list of 8-10 words, each "- **word**: simple meaning".
## Practice questions
5 numbered questions: 2 recall, 2 understanding, 1 that applies the idea to daily life. \
No answers here.
## Answers
The 5 answers, numbered, one or two sentences each.
## Note for the teacher
1-2 sentences: what the lesson is based on and anything that was missing or unclear."""

LABEL_GUIDANCE = {
    "Not Covered": (
        "NCERT does not teach this topic. Assume the child has never met it and teach it from "
        "the start. Under 'What you already know', use daily life, not NCERT."
    ),
    "Partially Covered": (
        "NCERT teaches part of this topic. Compare the passages: start from what NCERT already "
        "covers and spend most of the lesson on what the SCERT passage adds."
    ),
    "Covered": (
        "NCERT already covers this topic. Write a short revision lesson that introduces the "
        "SCERT textbook's own terms and examples."
    ),
}

LANGUAGE_GUIDANCE = {
    "English": "Write the whole pack in simple English.",
    "Hindi": (
        "Write the whole pack in simple Hindi (Devanagari script), except the six headings, "
        "which stay in English. After each science word, give the English word in brackets, "
        "because the child's class is taught in English."
    ),
}


def build_prompt(src: TopicSources, language: str = "English") -> tuple[str, str]:
    """Return (system, user) messages for one SCERT topic."""
    if language not in LANGUAGE_GUIDANCE:
        raise ValueError(f"language must be one of {LANGUAGES}")
    parts = [
        f"Topic: {src.title} (Tamil Nadu SCERT Class 6 Science, unit: {src.unit})",
        f"Gap label: {src.label} (coverage score {src.score:.3f})",
        LABEL_GUIDANCE.get(src.label, LABEL_GUIDANCE["Partially Covered"]),
        LANGUAGE_GUIDANCE[language],
        "",
        f'SCERT passage (what the child must learn): "{src.scert.title}"',
        "<<<",
        src.scert.text,
        ">>>",
    ]
    for i, p in enumerate(src.ncert, 1):
        parts += [
            "",
            f'NCERT passage {i} (closest match, chapter "{p.chapter}", '
            f'section "{p.title}", coverage {p.coverage:.3f}):',
            "<<<",
            p.text,
            ">>>",
        ]
    return SYSTEM, "\n".join(parts)


def pack_markdown(src: TopicSources, body: str, language: str, reviewer: str) -> str:
    """The exported pack: a title block followed by the reviewed lesson text."""
    head = [
        f"# {src.title}",
        "",
        f"*Bridge pack · Tamil Nadu SCERT Class 6 Science · unit: {src.unit} · "
        f"gap label: {src.label} · language: {language} · reviewed by {reviewer}*",
        "",
    ]
    return "\n".join(head) + "\n" + body.strip() + "\n"


def missing_sections(text: str) -> list[str]:
    """Headings from SECTIONS with no level-2 heading line starting with that name in `text`
    (a translation after the name, as in "## Lesson (पाठ)", is allowed)."""
    found = [m.group(1).strip().lower() for m in re.finditer(r"^##[ \t]+(.+)$", text, re.M)]
    return [s for s in SECTIONS if not any(h.startswith(s.lower()) for h in found)]
