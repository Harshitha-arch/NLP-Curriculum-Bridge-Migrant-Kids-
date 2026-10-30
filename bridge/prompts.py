"""Prompt for one bridge pack: a short lesson, key words and practice questions."""

from __future__ import annotations

import re

from bridge.data import TopicSources

PROMPT_VERSION = "v3"

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
- A short passage gives a short pack. Never add content to reach a length or a count: every \
length and count below is a maximum.
- The NCERT passages show what the child has probably studied already. Facts from them may \
appear only under "What you already know". Do not reteach them.
- Write for a child of about 11: short sentences, everyday words, one idea at a time. Explain \
every science word the first time you use it.
- Ignore figure labels, page furniture, broken fragments, and any exercise or multiple-choice \
questions in the passages.

Output Markdown with exactly these six level-2 headings, in this order:
## What you already know
1-3 sentences that link the topic to something the child knows from NCERT or daily life.
## Lesson
A lesson a teacher can read out in at most 10 minutes: up to 400 words in short paragraphs, \
with one everyday example taken from the SCERT passage if it has one.
## Key words
A bullet list of up to 10 science words that appear in the SCERT passage, each \
"- **word**: simple meaning".
## Practice questions
Up to 5 numbered questions that the lesson answers: recall first, then understanding, then \
(if the passage allows it) one that applies the idea to daily life. No answers here.
## Answers
One numbered answer per question, one or two sentences each, taken from the lesson.
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


def build_prompt(src: TopicSources) -> tuple[str, str]:
    """Return (system, user) messages for one SCERT topic."""
    parts = [
        f"Topic: {src.title} (Tamil Nadu SCERT Class 6 Science, unit: {src.unit})",
        f"Gap label: {src.label} (coverage score {src.score:.3f})",
        LABEL_GUIDANCE.get(src.label, LABEL_GUIDANCE["Partially Covered"]),
        "Write the whole pack in simple English.",
        "",
        f'SCERT passage (what the child must learn): "{src.scert.title}"',
        "<<<",
        src.scert.text,
        ">>>",
    ]
    if not src.ncert:
        parts += [
            "",
            "NCERT passages: No related NCERT section was found. Under 'What you already know', "
            "link the topic to daily life only.",
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


def pack_markdown(src: TopicSources, body: str, reviewer: str) -> str:
    """The exported pack: a title block followed by the reviewed lesson text."""
    head = [
        f"# {src.title}",
        "",
        f"*Bridge pack · Tamil Nadu SCERT Class 6 Science · unit: {src.unit} · "
        f"gap label: {src.label} · reviewed by {reviewer}*",
        "",
    ]
    return "\n".join(head) + "\n" + body.strip() + "\n"


def missing_sections(text: str) -> list[str]:
    """Headings from SECTIONS that are not a level-2 heading line of their own in `text`."""
    found = {m.group(1).strip().lower() for m in re.finditer(r"^##[ \t]+(.+?)[ \t#]*$", text, re.M)}
    return [s for s in SECTIONS if s.lower() not in found]


def split_sections(text: str) -> dict[str, str]:
    """Content under each level-2 heading, keyed by the heading text (lower case)."""
    parts = re.split(r"^##[ \t]+(.+?)[ \t#]*$", text, flags=re.M)
    return {parts[i].strip().lower(): parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}


def pack_checks(text: str) -> dict:
    """Automated format checks of one pack (no judgement of the content)."""
    sec = split_sections(text)

    def body(name: str) -> str:
        return sec.get(name.lower(), "")

    return {
        "missing_sections": "; ".join(missing_sections(text)),
        "empty_sections": "; ".join(s for s in SECTIONS if s.lower() in sec and not body(s)),
        "lesson_words": len(body("Lesson").split()),
        "key_word_count": len(re.findall(r"^\s*[-*+]\s+\S", body("Key words"), re.M)),
        "question_count": len(re.findall(r"^\s*\d+[.)]\s+\S", body("Practice questions"), re.M)),
    }
