"""Check the output against the PDFs: account for every line of text in every book.

Usage (after preprocessing.run):
    uv run python -m preprocessing.verify

For each line extracted from a PDF, the check asks whether its text appears in the
output for that book (``topics.jsonl``). A line that does not appear must have a known
reason to be missing; anything else is listed as "unexplained" so a person can look at
it. Reasons, in the order they are tested:

    outside_chapters  page is front matter (cover, contents) or back matter (glossary,
                      credits). Files skipped as a whole (NCERT *1ps.pdf) are not checked.
    verse             Sanskrit/Hindi verse or its translation and attribution
    page_furniture    footer, print slug, page number, box label, lone symbol
    dropped_topic     belongs to a topic dropped as empty/too short
    heading           the line is a topic or chapter title (kept as the title)
    unexplained       none of the above

Writes processed/verification.csv (one row per missing line) and prints a summary.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from preprocessing.clean import build_vocab, clean_text, keep_line
from preprocessing.extract import load_or_extract
from preprocessing.run import discover
from preprocessing.segment import back_matter_start, drop_verses

WORD = re.compile(r"[a-z0-9]+")


def norm(text: str) -> str:
    """Lowercase letters and digits only, single-spaced, for substring comparison."""
    return " ".join(WORD.findall(text.lower()))


def main() -> None:
    """Classify every extracted line that is missing from the output, per book."""
    cfg = yaml.safe_load(Path("configs/preprocessing.yaml").read_text(encoding="utf-8"))
    paths = cfg["paths"]
    out_dir = Path(paths["output_dir"])
    topics = [json.loads(x) for x in open(out_dir / "topics.jsonl", encoding="utf-8")]
    dropped = [json.loads(x) for x in open(out_dir / "dropped_topics.jsonl", encoding="utf-8")]
    kept_text: dict[str, str] = defaultdict(str)
    titles: dict[str, set[str]] = defaultdict(set)
    for t in topics:
        kept_text[t["source_file"]] += " " + norm(t["text"])
        titles[t["source_file"]] |= {norm(t["topic_title"]), norm(t["chapter"])}
    dropped_text: dict[str, str] = defaultdict(str)
    for t in dropped:
        dropped_text[t["source_file"]] += " " + norm(t["text"])

    rows: list[dict] = []
    per_book: list[tuple[str, int, Counter]] = []
    books = []
    for src in discover(
        Path(paths["input_root"]), paths.get("exclude_stems"), paths.get("default_grade")
    ):
        cache = Path(paths["page_cache"]) / src.board / f"Class{src.grade}" / f"{src.stem}.json"
        books.append((src, load_or_extract(src.path, cache, cfg["extraction"], pool=None)[0]))
    vocab = build_vocab([line.text for _, lines in books for line in lines])  # as in run.py
    for src, lines in books:
        rel = src.path.relative_to(paths["input_root"]).as_posix()
        back = back_matter_start(lines)
        verse_ids = {id(x) for x in lines} - {id(x) for x in drop_verses(lines)}
        first_page = min((t["page_start"] for t in topics if t["source_file"] == rel), default=1)
        reasons: Counter = Counter()
        for line in lines:
            if not norm(line.text):
                continue  # no letters or digits at all (icons, rules)
            text = norm(clean_text(line.text, vocab))
            if text and text in kept_text[rel]:
                reasons["found"] += 1
                continue
            if (back is not None and line.page >= back) or line.page + 1 < first_page:
                reason = "outside_chapters"
            elif id(line) in verse_ids:
                reason = "verse"
            elif not keep_line(line.text) or not clean_text(line.text, vocab):
                reason = "page_furniture"
            elif text in dropped_text[rel]:
                reason = "dropped_topic"
            elif (unnumbered := re.sub(r"^(\d+ )+", "", text)) and (
                unnumbered in kept_text[rel] or any(unnumbered in t for t in titles[rel])
            ):
                reason = "heading"  # kept as a topic title, without its section number
            else:
                reason = "unexplained"
            reasons[reason] += 1
            rows.append({"source_file": rel, "page": line.page + 1, "reason": reason,
                         "text": line.text.strip()[:200]})  # fmt: skip
        per_book.append((rel, len(lines), reasons))

    with open(out_dir / "verification.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source_file", "page", "reason", "text"])
        writer.writeheader()
        writer.writerows(rows)
    cols = ["found", "outside_chapters", "verse", "page_furniture", "dropped_topic",
            "heading", "unexplained"]  # fmt: skip
    short = ["found", "outside", "verse", "furniture", "dropped", "heading", "unexpl."]
    print(f"{'book':18}{'lines':>6}" + "".join(f"{c:>10}" for c in short))
    for rel, n, reasons in per_book:
        print(f"{rel:18}{n:>6}" + "".join(f"{reasons[c]:>10}" for c in cols))
    totals = [sum(r[c] for _, _, r in per_book) for c in cols]
    print(f"{'TOTAL':18}{sum(n for _, n, _ in per_book):>6}" + "".join(f"{t:>10}" for t in totals))
    print("columns: outside = cover/contents/glossary/credits pages, furniture = page numbers,")
    print("footers, labels; unexpl. = missing for no known reason (should be 0)")
    print(f"missing lines written to {out_dir / 'verification.csv'}")


if __name__ == "__main__":
    main()
