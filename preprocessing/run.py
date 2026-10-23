"""Run the preprocessing pipeline: PDFs -> lines -> topics -> clean text -> chunks.

Usage:
    uv run python -m preprocessing.run --config configs/preprocessing.yaml
    uv run python -m preprocessing.run --extract-only   # just fill the OCR/page cache

Outputs (in paths.output_dir):
    chunks.jsonl          one chunk (<= cleaning.max_chunk_chars) per line: the embedding input
    topics.jsonl          one kept topic per line with its full, uncut text
    dropped_topics.jsonl  topics removed as empty/noisy, with the reason
    summary.csv           topics and chunks per source PDF
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import yaml

from preprocessing.clean import (
    assess,
    build_vocab,
    clean_text,
    clean_title,
    detect_lang,
    split_chunks,
)
from preprocessing.extract import load_or_extract
from preprocessing.segment import segment

# <BOARD>/Class<N>/<name>.pdf or <BOARD>/<name>.pdf
PDF_PATTERN = re.compile(r"^(?P<board>[A-Za-z]+)/(?:Class(?P<grade>\d+)/)?(?P<stem>[^/]+)\.pdf$")
# NCERT file codes start with the class as a letter: a = Class 1 ... f = Class 6 ... l = 12.
NCERT_CODE = re.compile(r"^(?P<cls>[a-l])[a-z]{3}\d")


@dataclass
class Source:
    """One textbook PDF and the metadata encoded in its path."""

    path: Path
    board: str
    grade: int
    stem: str

    @property
    def term(self) -> int | None:
        """SCERT books are split by school term (Term_1.pdf ...); NCERT books are not."""
        match = re.fullmatch(r"Term_(\d)", self.stem)
        return int(match.group(1)) if match else None


def grade_of(board: str, folder_grade: str | None, stem: str, default: int | None) -> int:
    """The class a PDF belongs to: its Class<N> folder, else the NCERT file code
    ("fecu101" -> f -> Class 6), else the configured default."""
    if folder_grade:
        return int(folder_grade)
    code = NCERT_CODE.match(stem)
    if board == "NCERT" and code:
        return ord(code["cls"]) - ord("a") + 1
    if default is None:
        raise ValueError(f"Cannot tell the class of {board}/{stem}.pdf; set paths.default_grade")
    return default


def discover(
    input_root: Path, exclude_stems: str | None = None, default_grade: int | None = None
) -> list[Source]:
    """Find every <BOARD>/[Class<N>/]<name>.pdf under the input root, sorted, skipping
    files whose name (without .pdf) matches the ``exclude_stems`` regex."""
    sources = []
    for path in sorted(input_root.rglob("*.pdf")):
        match = PDF_PATTERN.match(path.relative_to(input_root).as_posix())
        if match and not (exclude_stems and re.search(exclude_stems, match["stem"])):
            board = match["board"].upper()
            grade = grade_of(board, match["grade"], match["stem"], default_grade)
            sources.append(Source(path, board, grade, match["stem"]))
    return sources


def main() -> None:
    """Parse arguments, run every stage and write the outputs."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/preprocessing.yaml")
    parser.add_argument("--extract-only", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    input_root = Path(cfg["paths"]["input_root"])
    cache_root = Path(cfg["paths"]["page_cache"])
    out_dir = Path(cfg["paths"]["output_dir"])

    kept: list[dict] = []
    chunks: list[dict] = []
    dropped: list[dict] = []
    summary: list[dict] = []
    max_chars = cfg["cleaning"]["max_chunk_chars"]
    books = []
    with ProcessPoolExecutor(max_workers=cfg["extraction"]["workers"]) as pool:
        paths = cfg["paths"]
        for src in discover(input_root, paths.get("exclude_stems"), paths.get("default_grade")):
            cache_path = cache_root / src.board / f"Class{src.grade}" / f"{src.stem}.json"
            lines, ocr_pages = load_or_extract(src.path, cache_path, cfg["extraction"], pool)
            print(f"{src.path}: {len(lines)} lines, {len(ocr_pages)} OCR pages", flush=True)
            books.append((src, lines, ocr_pages))
    if args.extract_only:
        return
    # Words of all books, to recognise diagram labels printed in a shifted font.
    vocab = build_vocab([line.text for _, lines, _ in books for line in lines])

    for src, lines, ocr_pages in books:
        topics = segment(lines, src.board, cfg["segmentation"])
        n_kept = n_chunks = 0
        for i, topic in enumerate(topics, start=1):
            title, body = clean_title(topic.title), clean_text(topic.body, vocab)
            # Lead with the title so the text says what the topic is about.
            full = f"{title}. {body}" if body and not body.startswith(title) else body
            record = {
                "topic_id": f"{src.board}_c{src.grade}_{src.stem}_{i:02d}",
                "board": src.board,
                "grade": src.grade,
                "term": src.term,
                "source_file": src.path.relative_to(input_root).as_posix(),
                "chapter": clean_title(topic.chapter),
                "topic_title": title,
                "section": topic.section,
                "page_start": topic.page_start + 1,  # 1-based, as printed by PDF viewers
                "page_end": topic.page_end + 1,
                "ocr": any(topic.page_start <= p <= topic.page_end for p in ocr_pages),
                "lang": detect_lang(full),
                "chars": len(full),
            }
            reason = assess(full, cfg["cleaning"], record["ocr"])
            if reason:
                dropped.append({**record, "drop_reason": reason, "text": full})
                continue
            parts = split_chunks(body, title, max_chars)
            kept.append({**record, "n_chunks": len(parts), "text": full})
            for k, part in enumerate(parts, start=1):
                chunks.append(
                    {
                        "chunk_id": f"{record['topic_id']}_{k}",
                        **record,
                        "chunk_index": k,
                        "n_chunks": len(parts),
                        "chars": len(part),
                        "text": part,
                    }
                )
            n_kept += 1
            n_chunks += len(parts)
        summary.append(
            {
                "source_file": src.path.relative_to(input_root).as_posix(),
                "ocr_pages": len(ocr_pages),
                "topics_found": len(topics),
                "topics_kept": n_kept,
                "chunks": n_chunks,
            }
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = (("chunks.jsonl", chunks), ("topics.jsonl", kept), ("dropped_topics.jsonl", dropped))
    for name, rows in outputs:
        with open(out_dir / name, "w", encoding="utf-8", newline="\n") as f:
            f.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    with open(out_dir / "summary.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"kept {len(kept)} topics as {len(chunks)} chunks, dropped {len(dropped)} -> {out_dir}")


if __name__ == "__main__":
    main()
