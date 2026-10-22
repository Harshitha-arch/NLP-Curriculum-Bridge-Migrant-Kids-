"""Stage 1: turn each PDF into a list of text lines with layout hints.

Native (selectable) pages are read with PyMuPDF. Pages with almost no selectable text
are rendered and OCR'd with Tesseract. Both paths produce the same ``Line`` records so
the segmentation stage does not need to know where a line came from.

Results are cached as one JSON file per PDF because OCR is slow (~10 s per page).
"""

from __future__ import annotations

import io
import json
import os
import shutil
import statistics
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import pymupdf
import pytesseract
from PIL import Image

WINDOWS_TESSERACT = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")


@dataclass
class Line:
    """One line of text on a page.

    ``size`` is the font size in points for native text, or the median word height
    converted to points for OCR text. Only ratios within one document are compared,
    so the two scales never need to agree. ``bold`` is None when unknown (OCR).
    ``conf`` is Tesseract's mean word confidence (0-100), None for native text.
    ``bbox`` is the line's (x0, y0, x1, y1) on the page in points, None for OCR text and
    for caches written before it was recorded.
    """

    text: str
    size: float
    bold: bool | None
    page: int
    conf: float | None = None
    bbox: tuple[float, float, float, float] | None = None


def find_tesseract(configured: str | None) -> str:
    """Return the Tesseract executable path, or raise if it is not installed."""
    for candidate in (configured, shutil.which("tesseract"), WINDOWS_TESSERACT):
        if candidate and Path(candidate).exists():
            return str(candidate)
    raise FileNotFoundError(
        "Tesseract not found. Install it (Windows: winget install UB-Mannheim.TesseractOCR) "
        "or set extraction.tesseract_cmd in the config."
    )


def native_lines(page: pymupdf.Page, page_no: int) -> list[Line]:
    """Read the selectable text lines of a page, with the dominant span's size and weight."""
    lines = []
    for block in page.get_text("dict")["blocks"]:
        for raw in block.get("lines", []):
            text = "".join(span["text"] for span in raw["spans"]).strip()
            if not text:
                continue
            main = max(raw["spans"], key=lambda span: len(span["text"].strip()))
            bold = "Bold" in main["font"] or bool(main["flags"] & 16)
            bbox = tuple(round(v, 1) for v in raw["bbox"])
            lines.append(Line(text, round(main["size"], 1), bold, page_no, bbox=bbox))
    return lines


def ocr_lines(png: bytes, page_no: int, dpi: int, lang: str, tesseract_cmd: str) -> list[Line]:
    """OCR a rendered page and group Tesseract's words back into lines."""
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    data = pytesseract.image_to_data(
        Image.open(io.BytesIO(png)), lang=lang, output_type=pytesseract.Output.DICT
    )
    grouped: dict[tuple[int, int, int], list[int]] = {}
    for i, word in enumerate(data["text"]):
        if word.strip() and float(data["conf"][i]) >= 0:
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            grouped.setdefault(key, []).append(i)
    lines = []
    for idx in grouped.values():
        text = " ".join(data["text"][i] for i in idx).strip()
        height_pt = statistics.median(data["height"][i] for i in idx) * 72 / dpi
        conf = statistics.fmean(float(data["conf"][i]) for i in idx)
        lines.append(Line(text, round(height_pt, 1), None, page_no, round(conf, 1)))
    return lines


def _ocr_job(args: tuple[str, int, int, str, str]) -> list[Line]:
    """Worker: render one page of a PDF and OCR it (runs in a separate process)."""
    pdf_path, page_no, dpi, lang, tesseract_cmd = args
    os.environ["OMP_THREAD_LIMIT"] = "1"  # one Tesseract thread per worker process
    with pymupdf.open(pdf_path) as doc:
        png = doc[page_no].get_pixmap(dpi=dpi).tobytes("png")
    return ocr_lines(png, page_no, dpi, lang, tesseract_cmd)


def extract_pdf(
    pdf_path: Path, cfg: dict, pool: ProcessPoolExecutor
) -> tuple[list[Line], list[int]]:
    """Extract all lines of a PDF. Returns (lines, page numbers that were OCR'd)."""
    lines: list[Line] = []
    to_ocr: list[int] = []
    with pymupdf.open(pdf_path) as doc:
        for page_no, page in enumerate(doc):
            page_lines = native_lines(page, page_no)
            if sum(len(line.text) for line in page_lines) >= cfg["min_native_chars"]:
                lines.extend(page_lines)
            else:
                to_ocr.append(page_no)
    if to_ocr:
        cmd = find_tesseract(cfg.get("tesseract_cmd"))
        jobs = [(str(pdf_path), p, cfg["ocr_dpi"], cfg["ocr_lang"], cmd) for p in to_ocr]
        for page_lines in pool.map(_ocr_job, jobs):
            lines.extend(page_lines)
    lines.sort(key=lambda line: line.page)  # stable: keeps reading order within a page
    return lines, to_ocr


def load_or_extract(
    pdf_path: Path, cache_path: Path, cfg: dict, pool: ProcessPoolExecutor
) -> tuple[list[Line], list[int]]:
    """Return cached lines for a PDF, extracting (and caching) them on first use."""
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        lines = [Line(**{**line, "bbox": tuple(line["bbox"]) if line.get("bbox") else None})
                 for line in cached["lines"]]  # fmt: skip
        return lines, cached["ocr_pages"]
    lines, ocr_pages = extract_pdf(pdf_path, cfg, pool)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"lines": [asdict(line) for line in lines], "ocr_pages": ocr_pages}
    cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return lines, ocr_pages
