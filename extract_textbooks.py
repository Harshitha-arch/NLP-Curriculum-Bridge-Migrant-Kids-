#!/usr/bin/env python3
"""
extract_textbooks.py

Walks a directory of textbook PDFs (NCERT / SCERT, organised as
<input_root>/<BOARD>/Class<N>/<chapter_or_term>.pdf), detects whether each
PDF is natively selectable or scanned, extracts text and embedded images
page-by-page, OCRs scanned pages and image regions that plausibly contain
text, and writes one JSON manifest + per-page text files + extracted
images per PDF.

Output layout (mirrors input):
    <output_root>/<BOARD>/Class<N>/<chapter_id>/
        manifest.json
        text/page_0001.txt
        text/page_0002.txt
        ...
        images/<file_prefix>_page0001_img01.png
        ...

A single run_log.csv at the output root logs one row per PDF processed,
so extraction issues are visible without opening every manifest.

Usage:
    python extract_textbooks.py --input data --output extracted

Dependencies (pip install -r requirements.txt):
    pymupdf
    pytesseract
    pillow

System dependency:
    Tesseract OCR engine must be installed and on PATH.
    - macOS:   brew install tesseract
    - Ubuntu:  sudo apt install tesseract-ocr
    - Windows: https://github.com/UB-Mannheim/tesseract/wiki
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from PIL import Image, ImageOps
import pytesseract


# ---------------------------------------------------------------------------
# Config / thresholds -- tune these if detection misclassifies specific books
# ---------------------------------------------------------------------------

# A page is treated as having no usable native text below this char count.
MIN_NATIVE_CHARS_PER_PAGE = 20

# Number of pages sampled (evenly spaced) to decide native vs scanned.
SAMPLE_PAGE_COUNT = 5

# Fraction of sampled pages that must look "native" for the whole PDF
# to be classified as native rather than scanned.
NATIVE_PAGE_FRACTION_THRESHOLD = 0.6

# Skip OCR on an extracted image if it's smaller than this on its shorter
# side (likely a bullet/icon/decoration, not a diagram or table worth
# reading) -- avoids wasted OCR calls on pure decoration.
MIN_IMAGE_DIMENSION_FOR_OCR = 60  # pixels

# Minimum characters an image's OCR must produce to be stored (filters out
# OCR noise from pure photographs/diagrams that have no text in them).
MIN_OCR_CHARS_TO_KEEP = 3

# An embedded image smaller than this on its shorter side (pixels) is
# flagged likely_decorative=True -- almost always a logo, corner icon, or
# border element, not content worth reviewing. Still saved, just flagged.
MIN_MEANINGFUL_IMAGE_DIMENSION = 40

# --- Vector-drawn figure detection (triangles, diagrams, charts drawn
#     with PDF line/curve commands rather than embedded raster images) ---

# Expand each individual drawing's bounding box by this many points before
# merging overlapping ones into a cluster -- keeps parts of the same
# figure (e.g. a triangle's three separate line segments) together.
VECTOR_CLUSTER_MARGIN = 8

# A merged cluster smaller than this (points, on its shorter side) is
# almost always a stray line, underline, or ruling -- not a real figure.
MIN_VECTOR_FIGURE_DIMENSION = 25

# If a page has more individual drawing paths than this, skip vector
# figure extraction for that page -- usually means dense decorative
# linework or a complex background, not a small number of distinct
# figures, and clustering that many paths would be slow for little gain.
MAX_DRAWINGS_PER_PAGE_FOR_CLUSTERING = 500

# DPI used when rendering a full page to an image for OCR (scanned pages).
# 300 is sharper but can make each page several times slower to OCR; 200 is
# a faster default. Override per-run with --ocr-dpi when accuracy matters
# more than runtime (e.g. a book whose OCR output has been coming out poor).
FULL_PAGE_OCR_DPI = 200

# Whether to apply preprocess_for_ocr() (grayscale + contrast stretch)
# before OCR. Testing on synthetic pages showed this helps in some
# contrast conditions and very slightly hurts in others -- it is not a
# guaranteed universal improvement, since real scans have noise
# characteristics (paper texture, lighting gradients, scan artifacts)
# that a synthetic test can't fully replicate. Defaults on since the
# effect was net-positive and mild either way in testing; disable with
# --no-ocr-preprocess if you A/B test it on your real books and prefer
# the raw behavior.
OCR_PREPROCESS_ENABLED = True

# Tesseract page segmentation mode. None = let Tesseract auto-detect layout
# (its default, works fine for most pages). Textbook pages that mix running
# text, images and tables in irregular layouts sometimes do better with an
# explicit mode -- e.g. 6 ("assume a single uniform block of text") or 4
# ("assume a single column of text of variable sizes"). Override with
# --ocr-psm to test a specific value against a troublesome book.
TESSERACT_PSM: Optional[int] = None

# Max seconds pytesseract will wait on a single OCR call before giving up.
# Without this, one malformed or unusually large page/image can hang the
# entire batch indefinitely with no error and no way to tell from the log.
OCR_TIMEOUT_SECONDS = 60


# ---------------------------------------------------------------------------
# Data structures -- these define the handoff schema to the structuring step
# ---------------------------------------------------------------------------

@dataclass
class ImageRecord:
    image_id: str
    file: str                  # path to saved image, relative to manifest.json
    bbox: list                 # [x0, y0, x1, y1] in PDF point coordinates
    width_px: int
    height_px: int
    ocr_attempted: bool
    likely_decorative: bool    # True if below MIN_MEANINGFUL_IMAGE_DIMENSION --
                                # a logo/icon/border rather than real content
    conversion_status: str     # "pixmap" | "pillow_fallback" | "raw_unconverted"
                                # -- see extract_page_images(); only
                                # "raw_unconverted" risks being unviewable
                                # in common tools (e.g. still raw JPX/JP2)
    ocr_text: Optional[str] = None


@dataclass
class TableRecord:
    table_id: str
    bbox: list                 # [x0, y0, x1, y1] in PDF point coordinates
    rows: list                 # list[list[str]] -- row-major grid of cell text


@dataclass
class VectorFigureRecord:
    """
    A region of vector-drawn graphics (lines, curves, shapes) rasterized
    and saved as its own image. Needed because page.get_images() only
    returns embedded raster images -- a triangle, angle diagram, or chart
    drawn directly with PDF line/curve commands (very common for math
    exercise figures) is invisible to that call and would otherwise be
    lost entirely, leaving only the exercise's text with no diagram.
    """
    figure_id: str
    file: str
    bbox: list
    ocr_attempted: bool
    ocr_text: Optional[str] = None


@dataclass
class PageRecord:
    page_number: int           # 1-indexed
    text: str
    text_source: str           # "native" | "ocr"
    char_count: int
    images: list = field(default_factory=list)          # list[dict] (ImageRecord)
    tables: list = field(default_factory=list)           # list[dict] (TableRecord)
    vector_figures: list = field(default_factory=list)   # list[dict] (VectorFigureRecord)


@dataclass
class BookManifest:
    source_file: str
    board: str
    class_level: Optional[int]
    chapter_id: str
    num_pages: int
    extraction_method: str     # "native" | "scanned" | "mixed"
    native_page_count: int
    ocr_page_count: int
    total_images: int
    images_with_text: int
    total_tables: int
    total_vector_figures: int
    total_phantom_images_skipped: int  # images referenced but not actually
                                        # drawn on their listed page -- see
                                        # extract_page_images() for why
    pages: list = field(default_factory=list)  # list[dict] (from PageRecord)


# ---------------------------------------------------------------------------
# Path parsing -- infer board / class / chapter_id from folder structure
# ---------------------------------------------------------------------------

CLASS_DIR_RE = re.compile(r"class\s*0*(\d+)", re.IGNORECASE)


def parse_book_identity(pdf_path: Path, input_root: Path) -> tuple[str, Optional[int], str]:
    """
    Given <input_root>/<BOARD>/Class<N>/<chapter_or_term>.pdf, return
    (board, class_level, chapter_id).

    Falls back gracefully if the structure doesn't match exactly, rather
    than crashing -- a malformed path should show up in run_log.csv, not
    kill the whole batch.
    """
    try:
        rel_parts = pdf_path.relative_to(input_root).parts
    except ValueError:
        rel_parts = pdf_path.parts

    board = rel_parts[0] if len(rel_parts) >= 1 else "unknown_board"

    class_level = None
    for part in rel_parts:
        match = CLASS_DIR_RE.search(part)
        if match:
            class_level = int(match.group(1))
            break

    chapter_id = pdf_path.stem  # filename without ".pdf", e.g. "cemm101" or "Term_1"

    return board, class_level, chapter_id


# ---------------------------------------------------------------------------
# Scanned vs native detection
# ---------------------------------------------------------------------------

def sample_page_indices(num_pages: int, sample_count: int) -> list[int]:
    """Evenly spaced page indices (0-indexed) to sample, capped at num_pages."""
    if num_pages <= sample_count:
        return list(range(num_pages))
    step = num_pages / sample_count
    return sorted({int(i * step) for i in range(sample_count)})


def classify_document(doc: fitz.Document) -> str:
    """
    Returns "native", "scanned", or "mixed" based on a sample of pages.

    Heuristic: a page counts as "native" if its extracted text exceeds
    MIN_NATIVE_CHARS_PER_PAGE. If enough sampled pages are native, the
    whole document is "native". If none are, it's "scanned". Otherwise
    "mixed" -- common in textbooks where some pages are mostly a
    full-page diagram with little running text.

    This classification is informational / for the run log; actual
    extraction below is decided per-page regardless, so a "mixed" verdict
    never causes text to be lost.
    """
    indices = sample_page_indices(doc.page_count, SAMPLE_PAGE_COUNT)
    native_flags = []
    for i in indices:
        text = doc[i].get_text("text")
        native_flags.append(len(text.strip()) >= MIN_NATIVE_CHARS_PER_PAGE)

    if not native_flags:
        return "scanned"  # empty doc, treat conservatively

    native_fraction = sum(native_flags) / len(native_flags)
    if native_fraction >= NATIVE_PAGE_FRACTION_THRESHOLD:
        return "native"
    elif native_fraction == 0:
        return "scanned"
    else:
        return "mixed"


# ---------------------------------------------------------------------------
# Per-page extraction
# ---------------------------------------------------------------------------

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

def preprocess_for_ocr(img: "Image.Image") -> "Image.Image":
    """
    Light preprocessing before handing an image to Tesseract.

    Grayscale + autocontrast is the standard recommendation for modern
    (LSTM-based, Tesseract 4+) OCR: it noticeably helps on scans with
    uneven lighting or a slightly grey/off-white background, without the
    downsides of hard binarization. Full black/white thresholding is
    deliberately NOT applied here by default -- the LSTM engine is trained
    on grayscale text and often does *worse* on aggressively binarized
    images than on well-contrasted grayscale ones, especially for smaller
    textbook fonts where binarization can delete thin strokes entirely.
    """
    gray = img.convert("L")
    # cutoff=0 is deliberate, not a placeholder: testing showed any nonzero
    # cutoff treats sparse dark text on a mostly-light page as a
    # statistical "outlier" and clips it away, which silently destroyed
    # OCR output entirely on realistic textbook-page contrast levels
    # during testing (confirmed on this exact script, not a general
    # claim). cutoff=0 performs a plain min/max stretch with no clipping.
    return ImageOps.autocontrast(gray, cutoff=0)


def _tesseract_config() -> str:
    return f"--psm {TESSERACT_PSM}" if TESSERACT_PSM is not None else ""


def ocr_full_page(page: fitz.Page, dpi: Optional[int] = None) -> str:
    """
    Render a page to an image and OCR it. Used for non-native pages.

    dpi defaults to the module-level FULL_PAGE_OCR_DPI, read at call time
    (not bound as a default parameter value) so that a --ocr-dpi override
    applied in main() after module load actually takes effect here.
    """
    if dpi is None:
        dpi = FULL_PAGE_OCR_DPI
    pix = page.get_pixmap(dpi=dpi)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    if OCR_PREPROCESS_ENABLED:
        img = preprocess_for_ocr(img)
    return pytesseract.image_to_string(
        img, timeout=OCR_TIMEOUT_SECONDS, config=_tesseract_config()
    )


def extract_page_images(
    doc: fitz.Document,
    page: fitz.Page,
    page_number: int,
    images_dir: Path,
    file_prefix: str,
    skip_individual_ocr: bool = False,
) -> tuple[list[ImageRecord], int]:
    """
    Extracts every embedded image on a page, saves it to images_dir, and
    OCRs it individually (not the whole page) if it's large enough to
    plausibly contain readable text (a table, a labelled diagram).

    Pure shape/decoration images are still saved -- the structuring step
    may still need them tied to their exercise -- just without a wasted
    OCR call that would return nothing useful.

    skip_individual_ocr=True is used when the page itself just went
    through full-page OCR (text_source == "ocr"): on a scanned page the
    embedded "image" is often the entire page, so OCRing it again here
    would just redo the same work a second time. The image is still
    extracted and saved (position data still matters), just not re-OCR'd.
    """
    records: list[ImageRecord] = []
    phantom_skipped = 0
    img_list = page.get_images(full=True)

    for idx, img_info in enumerate(img_list, start=1):
        xref = img_info[0]

        # page.get_images() lists every image referenced in this page's
        # resource dictionary -- which is not strictly the same as every
        # image actually drawn ON this page. Some PDFs (especially ones
        # produced by extracting a subset of pages from a larger combined
        # document, e.g. a per-subject export from a multi-subject master
        # PDF) don't clean up the shared resource table when pages are
        # dropped, so a page's resource list can still reference images
        # that belong to content no longer even present in the file.
        # get_image_rects returns the image's actual placement(s) on THIS
        # page specifically; an empty result means it's not actually
        # drawn here, so we skip it rather than misattributing unrelated
        # content to this page.
        rects = page.get_image_rects(xref)
        if not rects:
            phantom_skipped += 1
            logging.debug(
                "  page %d: skipping xref=%s -- listed in this page's "
                "resources but not actually drawn here (stale reference, "
                "likely from a page-subset export of a larger document)",
                page_number, xref,
            )
            continue
        bbox = list(rects[0])

        # Re-encode every image to PNG via a Pixmap rather than writing the
        # raw embedded bytes as-is. Embedded images come in whatever codec
        # the source PDF used -- JPEG, JPEG2000/JPX, CMYK JPEG, etc -- and
        # JPX in particular isn't renderable by most image viewers/IDEs
        # without an extra codec. MuPDF can decode all of these internally,
        # so going through Pixmap normalizes everything to one viewable,
        # consistent format.
        #
        # Three-tier fallback if that fails:
        #   1. fitz.Pixmap(doc, xref)      -- MuPDF's own decoder
        #   2. Pillow on the raw bytes     -- a different decode path that
        #                                     sometimes succeeds where #1
        #                                     doesn't (e.g. certain JPEG
        #                                     variants Pillow handles but
        #                                     MuPDF's Pixmap route trips on)
        #   3. raw embedded bytes as-is    -- last resort; may genuinely be
        #                                     unviewable (e.g. still JPX)
        # conversion_status on the saved record always reflects which tier
        # actually produced the file, so this is visible per-image rather
        # than only in the log.
        img_ext = "png"
        width_px = 0
        height_px = 0
        conversion_status = "pixmap"
        img_bytes = None

        try:
            pix = fitz.Pixmap(doc, xref)
            if pix.colorspace is None:
                # Image mask / stencil with no colorspace of its own --
                # composite against a plain RGB pixmap so it still saves
                # as a normal viewable image rather than failing.
                pix = fitz.Pixmap(fitz.csRGB, pix)
            elif pix.colorspace.n >= 4:
                # CMYK or other non-RGB colorspace -- convert so common
                # viewers can open it.
                pix = fitz.Pixmap(fitz.csRGB, pix)
            img_bytes = pix.tobytes("png")
            width_px = pix.width
            height_px = pix.height
        except Exception as e:
            # Pull whatever metadata PyMuPDF already has for this image
            # (filter/codec name, declared size) so the log line is
            # actionable without needing a repro file.
            try:
                raw_info = doc.extract_image(xref)
                codec_hint = raw_info.get("ext", "unknown")
                declared_size = f"{raw_info.get('width')}x{raw_info.get('height')}"
            except Exception:
                codec_hint = "unknown"
                declared_size = "unknown"

            logging.warning(
                "  page %d: Pixmap conversion failed for xref=%s "
                "(codec=%s, size=%s): %s: %s -- trying Pillow fallback",
                page_number, xref, codec_hint, declared_size,
                type(e).__name__, e,
            )

            try:
                base_image = doc.extract_image(xref)
                raw_bytes = base_image["image"]
                pil_img = Image.open(io.BytesIO(raw_bytes))
                pil_img.load()  # force full decode now, not lazily later
                if pil_img.mode not in ("RGB", "RGBA", "L"):
                    pil_img = pil_img.convert("RGB")
                out_buf = io.BytesIO()
                pil_img.save(out_buf, format="PNG")
                img_bytes = out_buf.getvalue()
                width_px, height_px = pil_img.size
                conversion_status = "pillow_fallback"
            except Exception as e2:
                logging.warning(
                    "  page %d: Pillow fallback also failed for xref=%s "
                    "(%s: %s) -- saving raw embedded bytes as-is; this "
                    "image may not be viewable in standard tools if its "
                    "codec is JPX/JP2",
                    page_number, xref, type(e2).__name__, e2,
                )
                try:
                    base_image = doc.extract_image(xref)
                    img_bytes = base_image["image"]
                    img_ext = base_image["ext"]
                    width_px = base_image.get("width", 0)
                    height_px = base_image.get("height", 0)
                    conversion_status = "raw_unconverted"
                except Exception as e3:
                    logging.warning(
                        "  page %d: could not extract image xref=%s at "
                        "all (%s: %s)",
                        page_number, xref, type(e3).__name__, e3,
                    )
                    continue

        image_id = f"{file_prefix}_page{page_number:04d}_img{idx:02d}"
        image_filename = f"{image_id}.{img_ext}"
        image_path = images_dir / image_filename
        image_path.write_bytes(img_bytes)

        smaller_dim = min(width_px, height_px) if width_px and height_px else 0
        likely_decorative = smaller_dim < MIN_MEANINGFUL_IMAGE_DIMENSION

        ocr_attempted = False
        ocr_text = None

        if skip_individual_ocr:
            pass  # page-level OCR already covered this image's text, if any
        elif smaller_dim >= MIN_IMAGE_DIMENSION_FOR_OCR:
            ocr_attempted = True
            try:
                pil_img = Image.open(image_path)
                if OCR_PREPROCESS_ENABLED:
                    pil_img = preprocess_for_ocr(pil_img)
                raw_text = pytesseract.image_to_string(
                    pil_img, timeout=OCR_TIMEOUT_SECONDS,
                    config=_tesseract_config(),
                ).strip()
                if len(raw_text) >= MIN_OCR_CHARS_TO_KEEP:
                    ocr_text = raw_text
            except Exception as e:
                logging.warning(
                    "  page %d: OCR failed on %s (%s)",
                    page_number, image_filename, e,
                )

        records.append(
            ImageRecord(
                image_id=image_id,
                file=f"images/{image_filename}",
                bbox=bbox,
                width_px=width_px,
                height_px=height_px,
                ocr_attempted=ocr_attempted,
                likely_decorative=likely_decorative,
                conversion_status=conversion_status,
                ocr_text=ocr_text,
            )
        )

    return records, phantom_skipped


# ---------------------------------------------------------------------------
# Table detection (ruled tables drawn with PDF lines, not embedded images)
# ---------------------------------------------------------------------------

def extract_page_tables(page: fitz.Page, page_number: int) -> list[TableRecord]:
    """
    Detects and extracts ruled tables on a page using PyMuPDF's built-in
    table finder, returning each as a row-major grid of cell text with its
    bounding box. Running text extraction alone (get_text) loses column
    alignment entirely, which is exactly why table-heavy pages tend to
    come out looking garbled -- this gives the structuring step actual
    rows/columns instead of a flattened string.

    Requires PyMuPDF >= 1.23. If the installed version doesn't have
    find_tables(), this silently returns an empty list rather than
    crashing the run; a one-time warning is logged instead.
    """
    tables_out: list[TableRecord] = []

    find_tables = getattr(page, "find_tables", None)
    if find_tables is None:
        return tables_out  # older PyMuPDF; warned about once in main()

    try:
        table_finder = find_tables()
    except Exception as e:
        logging.debug(
            "  page %d: table detection failed (%s)", page_number, e
        )
        return tables_out

    for i, tbl in enumerate(table_finder.tables, start=1):
        try:
            rows = tbl.extract()
        except Exception as e:
            logging.debug(
                "  page %d: could not extract detected table %d (%s)",
                page_number, i, e,
            )
            rows = []

        tables_out.append(
            TableRecord(
                table_id=f"page{page_number:04d}_table{i:02d}",
                bbox=list(tbl.bbox),
                rows=rows,
            )
        )

    return tables_out


# ---------------------------------------------------------------------------
# Vector-drawn figure extraction (shapes/diagrams drawn with line/curve
# commands directly in the PDF content stream -- NOT embedded images)
# ---------------------------------------------------------------------------

def _merge_overlapping_rects(rects: list, margin: float) -> list:
    """
    Iteratively merges rects that overlap once each is expanded by margin,
    so that separate line/curve segments belonging to one drawn figure
    (e.g. a triangle's three sides, each a distinct path) end up in a
    single combined bounding box instead of fragmenting into pieces.
    """
    boxes = list(rects)
    changed = True
    while changed:
        changed = False
        merged: list = []
        used = [False] * len(boxes)
        for i in range(len(boxes)):
            if used[i]:
                continue
            base = fitz.Rect(boxes[i])
            used[i] = True
            probe = fitz.Rect(base.x0 - margin, base.y0 - margin,
                               base.x1 + margin, base.y1 + margin)
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                if probe.intersects(boxes[j]):
                    base |= boxes[j]
                    probe = fitz.Rect(base.x0 - margin, base.y0 - margin,
                                       base.x1 + margin, base.y1 + margin)
                    used[j] = True
                    changed = True
            merged.append(base)
        boxes = merged
    return boxes


def extract_vector_figures(
    page: fitz.Page,
    page_number: int,
    images_dir: Path,
    file_prefix: str,
    dpi: Optional[int] = None,
) -> list[VectorFigureRecord]:
    """
    Finds regions of vector-drawn graphics (geometric shapes, diagrams --
    the kind of content commonly used for math exercise figures) and
    rasterizes each region to its own saved image.

    This exists because page.get_images() -- used elsewhere in this
    script -- only returns embedded raster images. A triangle or angle
    diagram drawn directly with PDF line/curve commands is a completely
    different mechanism and is invisible to that call. Without this,
    those figures are silently dropped: the exercise's text survives,
    but the diagram it depends on does not, which is the likely cause of
    "exercises with shapes don't make sense" in the extracted output.

    dpi defaults to FULL_PAGE_OCR_DPI, read at call time so a --ocr-dpi
    override applies here too (see ocr_full_page for why this matters).
    """
    if dpi is None:
        dpi = FULL_PAGE_OCR_DPI
    figures: list[VectorFigureRecord] = []

    try:
        drawings = page.get_drawings()
    except Exception as e:
        logging.debug(
            "  page %d: could not read vector drawings (%s)", page_number, e
        )
        return figures

    if not drawings:
        return figures

    if len(drawings) > MAX_DRAWINGS_PER_PAGE_FOR_CLUSTERING:
        logging.debug(
            "  page %d: %d drawing paths, skipping vector-figure "
            "clustering (likely dense decorative linework, not discrete "
            "figures)", page_number, len(drawings),
        )
        return figures

    raw_rects = [fitz.Rect(d["rect"]) for d in drawings if d.get("rect")]
    if not raw_rects:
        return figures

    clusters = _merge_overlapping_rects(raw_rects, VECTOR_CLUSTER_MARGIN)

    page_rect = page.rect
    for i, rect in enumerate(clusters, start=1):
        rect = rect & page_rect  # clip to page bounds
        if rect.is_empty:
            continue
        if min(rect.width, rect.height) < MIN_VECTOR_FIGURE_DIMENSION:
            continue  # almost certainly a stray line/underline, not a figure

        try:
            pix = page.get_pixmap(dpi=dpi, clip=rect)
        except Exception as e:
            logging.warning(
                "  page %d: could not rasterize vector figure %d (%s)",
                page_number, i, e,
            )
            continue

        figure_id = f"{file_prefix}_page{page_number:04d}_vecfig{i:02d}"
        filename = f"{figure_id}.png"
        filepath = images_dir / filename
        pix.save(str(filepath))

        ocr_attempted = False
        ocr_text = None
        smaller_dim = min(pix.width, pix.height)
        if smaller_dim >= MIN_IMAGE_DIMENSION_FOR_OCR:
            ocr_attempted = True
            try:
                fig_img = Image.open(filepath)
                if OCR_PREPROCESS_ENABLED:
                    fig_img = preprocess_for_ocr(fig_img)
                raw_text = pytesseract.image_to_string(
                    fig_img, timeout=OCR_TIMEOUT_SECONDS,
                    config=_tesseract_config(),
                ).strip()
                if len(raw_text) >= MIN_OCR_CHARS_TO_KEEP:
                    ocr_text = raw_text
            except Exception as e:
                logging.warning(
                    "  page %d: OCR failed on vector figure %s (%s)",
                    page_number, filename, e,
                )

        figures.append(
            VectorFigureRecord(
                figure_id=figure_id,
                file=f"images/{filename}",
                bbox=list(rect),
                ocr_attempted=ocr_attempted,
                ocr_text=ocr_text,
            )
        )

    return figures


# ---------------------------------------------------------------------------
# Per-PDF processing
# ---------------------------------------------------------------------------

def manifest_already_exists(pdf_path: Path, input_root: Path, output_root: Path) -> bool:
    """Used by --skip-existing to avoid redoing PDFs from a prior run."""
    board, class_level, chapter_id = parse_book_identity(pdf_path, input_root)
    book_out_dir = (
        output_root / board / f"Class{class_level}" / chapter_id
        if class_level is not None
        else output_root / board / chapter_id
    )
    return (book_out_dir / "manifest.json").exists()


def process_pdf(pdf_path: Path, input_root: Path, output_root: Path) -> dict:
    """
    Processes one PDF end-to-end. Returns a summary dict for the run log.
    Never raises -- any failure is caught, logged, and reported in the
    summary so one bad file doesn't stop the whole batch.
    """
    board, class_level, chapter_id = parse_book_identity(pdf_path, input_root)

    # file_prefix keeps image filenames unique even if two chapters share
    # a page-numbering scheme, e.g. "NCERT_Class3_cemm101"
    file_prefix = (
        f"{board}_Class{class_level}_{chapter_id}"
        if class_level is not None
        else f"{board}_{chapter_id}"
    )

    book_out_dir = (
        output_root / board / f"Class{class_level}" / chapter_id
        if class_level is not None
        else output_root / board / chapter_id
    )
    text_dir = book_out_dir / "text"
    images_dir = book_out_dir / "images"
    text_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "source_file": str(pdf_path),
        "board": board,
        "class_level": class_level,
        "chapter_id": chapter_id,
        "status": "failed",
        "extraction_method": None,
        "num_pages": 0,
        "total_images": 0,
        "images_with_text": 0,
        "total_tables": 0,
        "total_vector_figures": 0,
        "total_phantom_images_skipped": 0,
        "error": None,
    }

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        summary["error"] = f"could not open PDF: {e}"
        logging.error("FAILED to open %s: %s", pdf_path, e)
        return summary

    doc_classification = classify_document(doc)  # "native" | "scanned" | "mixed"
    logging.info(
        "%s -> classified as '%s' (%d pages)",
        pdf_path.name, doc_classification, doc.page_count,
    )

    pages: list[PageRecord] = []
    native_page_count = 0
    ocr_page_count = 0
    total_images = 0
    images_with_text = 0
    total_tables = 0
    total_vector_figures = 0
    total_phantom_images_skipped = 0

    for page_index in range(doc.page_count):
        page_number = page_index + 1
        page = doc[page_index]

        native_text = page.get_text("text")
        page_is_native = len(native_text.strip()) >= MIN_NATIVE_CHARS_PER_PAGE

        if page_is_native:
            text = native_text
            text_source = "native"
            native_page_count += 1
        else:
            # Covers both fully scanned docs and individual pages within
            # an otherwise native doc that are mostly a full-page figure --
            # falls back to OCR so text is never silently dropped.
            # Logged per-page (not just once per PDF) because full-page
            # OCR is the slow step -- without this, a long scanned book
            # looks "stuck" with no way to tell it's still working.
            logging.info(
                "  page %d/%d: running full-page OCR...",
                page_number, doc.page_count,
            )
            try:
                text = ocr_full_page(page)
            except RuntimeError as e:
                # pytesseract raises RuntimeError on timeout
                logging.warning(
                    "  page %d: full-page OCR timed out after %ds (%s)",
                    page_number, OCR_TIMEOUT_SECONDS, e,
                )
                text = ""
            except Exception as e:
                logging.warning(
                    "  page %d: full-page OCR failed (%s)", page_number, e
                )
                text = ""
            text_source = "ocr"
            ocr_page_count += 1

        # Always extract embedded images, regardless of whether the page
        # text itself was native or OCR'd -- diagrams/tables live here,
        # and exercises referencing them need this to make sense later.
        # If the page just went through full-page OCR, skip re-OCRing its
        # images individually: on a scanned page the embedded image is
        # usually the entire page, so this avoids doing the same OCR pass
        # twice (the main cause of the multi-hour runtime).
        image_records, phantom_skipped = extract_page_images(
            doc, page, page_number, images_dir, file_prefix,
            skip_individual_ocr=(text_source == "ocr"),
        )
        total_images += len(image_records)
        images_with_text += sum(1 for r in image_records if r.ocr_text)
        total_phantom_images_skipped += phantom_skipped

        # Ruled tables (rows/columns) -- separate from running text because
        # get_text() flattens tabular layout and loses column alignment.
        table_records = extract_page_tables(page, page_number)
        total_tables += len(table_records)

        # Vector-drawn figures (shapes/diagrams drawn with line/curve
        # commands, not embedded raster images) -- only attempted on
        # native/mixed pages. On a page that already went through
        # full-page OCR, the whole page was already rasterized and saved
        # as page text's source image conceptually; re-detecting vector
        # paths on it would be redundant and OCR of the saved images
        # already covers any labelled text in those pages.
        vector_figure_records: list[VectorFigureRecord] = []
        if text_source == "native":
            vector_figure_records = extract_vector_figures(
                page, page_number, images_dir, file_prefix
            )
        total_vector_figures += len(vector_figure_records)

        # Save page text as its own file too, for quick diffing/review
        # without having to open the full JSON manifest.
        page_text_path = text_dir / f"page_{page_number:04d}.txt"
        page_text_path.write_text(text, encoding="utf-8")

        pages.append(
            PageRecord(
                page_number=page_number,
                text=text,
                text_source=text_source,
                char_count=len(text),
                images=[asdict(r) for r in image_records],
                tables=[asdict(t) for t in table_records],
                vector_figures=[asdict(v) for v in vector_figure_records],
            )
        )

    num_pages = doc.page_count  # captured before close() -- doc.page_count
                                 # raises once the document is closed

    manifest = BookManifest(
        source_file=str(pdf_path),
        board=board,
        class_level=class_level,
        chapter_id=chapter_id,
        num_pages=num_pages,
        extraction_method=doc_classification,
        native_page_count=native_page_count,
        ocr_page_count=ocr_page_count,
        total_images=total_images,
        images_with_text=images_with_text,
        total_tables=total_tables,
        total_vector_figures=total_vector_figures,
        total_phantom_images_skipped=total_phantom_images_skipped,
        pages=[asdict(p) for p in pages],
    )

    manifest_path = book_out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(asdict(manifest), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    doc.close()

    summary.update(
        {
            "status": "success",
            "extraction_method": doc_classification,
            "num_pages": num_pages,
            "total_images": total_images,
            "images_with_text": images_with_text,
            "total_tables": total_tables,
            "total_vector_figures": total_vector_figures,
            "total_phantom_images_skipped": total_phantom_images_skipped,
        }
    )
    return summary


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def find_all_pdfs(input_root: Path) -> list[Path]:
    return sorted(input_root.rglob("*.pdf"))


def write_run_log(summaries: list[dict], output_root: Path) -> None:
    """
    Writes run_log.csv, merging with any existing log from a previous run
    (keyed by source_file) rather than overwriting it. This matters when
    resuming with --skip-existing: without the merge, the log would only
    show the newly processed PDFs and lose the record of everything that
    already succeeded earlier.
    """
    log_path = output_root / "run_log.csv"
    fieldnames = [
        "source_file", "board", "class_level", "chapter_id", "status",
        "extraction_method", "num_pages", "total_images",
        "images_with_text", "total_tables", "total_vector_figures",
        "total_phantom_images_skipped", "error",
    ]

    merged: dict[str, dict] = {}
    if log_path.exists():
        with log_path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                merged[row["source_file"]] = row

    for row in summaries:
        merged[row["source_file"]] = row

    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in merged.values():
            writer.writerow(row)
    logging.info("Run log written to %s (%d total entries)", log_path, len(merged))


def main():
    global FULL_PAGE_OCR_DPI, TESSERACT_PSM, OCR_PREPROCESS_ENABLED

    parser = argparse.ArgumentParser(
        description="Extract text + images from NCERT/SCERT textbook PDFs."
    )
    parser.add_argument(
        "--input", required=True, type=Path,
        help="Root data directory (e.g. 'data') containing <BOARD>/Class<N>/*.pdf",
    )
    parser.add_argument(
        "--output", required=True, type=Path,
        help="Directory to write extracted text/images/manifests into",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help=(
            "Skip any PDF that already has a manifest.json in the output "
            "directory from a previous run. Use this to resume a batch "
            "after stopping it partway through, instead of redoing "
            "books that already finished successfully."
        ),
    )
    parser.add_argument(
        "--ocr-dpi", type=int, default=None,
        help=(
            f"DPI for rendering scanned pages before OCR (default "
            f"{FULL_PAGE_OCR_DPI}). Raise this (e.g. 300) for books where "
            f"OCR text quality is poor -- slower, but often meaningfully "
            f"more accurate, especially on smaller print."
        ),
    )
    parser.add_argument(
        "--no-ocr-preprocess", action="store_true",
        help=(
            "Disable grayscale+contrast preprocessing before OCR and use "
            "raw page renders instead. Testing showed preprocessing is "
            "net-positive but not a guaranteed win on every page -- try "
            "this flag if OCR quality looks worse after upgrading."
        ),
    )
    parser.add_argument(
        "--ocr-psm", type=int, default=None,
        help=(
            "Tesseract page segmentation mode (default: let Tesseract "
            "auto-detect). Try 6 (single uniform text block) or 4 (single "
            "column, variable font sizes) if OCR output looks scrambled "
            "on a book with an unusual page layout."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.ocr_dpi is not None:
        FULL_PAGE_OCR_DPI = args.ocr_dpi
        logging.info("OCR DPI overridden to %d", FULL_PAGE_OCR_DPI)
    if args.ocr_psm is not None:
        TESSERACT_PSM = args.ocr_psm
        logging.info("Tesseract PSM overridden to %d", TESSERACT_PSM)
    if args.no_ocr_preprocess:
        OCR_PREPROCESS_ENABLED = False
        logging.info("OCR preprocessing disabled (--no-ocr-preprocess)")

    if not hasattr(fitz.Page, "find_tables"):
        logging.warning(
            "Installed PyMuPDF does not support find_tables() (needs "
            "PyMuPDF >= 1.23). Table detection will be skipped for every "
            "page -- run 'pip install --upgrade pymupdf' to enable it."
        )

    if not args.input.exists():
        logging.error("Input directory does not exist: %s", args.input)
        sys.exit(1)

    args.output.mkdir(parents=True, exist_ok=True)

    pdfs = find_all_pdfs(args.input)
    if not pdfs:
        logging.warning("No PDFs found under %s", args.input)
        sys.exit(0)

    if args.skip_existing:
        before = len(pdfs)
        pdfs = [
            p for p in pdfs
            if not manifest_already_exists(p, args.input, args.output)
        ]
        skipped = before - len(pdfs)
        if skipped:
            logging.info(
                "--skip-existing: skipping %d PDF(s) already processed",
                skipped,
            )

    logging.info("Found %d PDFs to process", len(pdfs))

    summaries = []
    for pdf_path in pdfs:
        logging.info("Processing: %s", pdf_path)
        summary = process_pdf(pdf_path, args.input, args.output)
        summaries.append(summary)
        if summary["status"] == "failed":
            logging.error("  -> FAILED: %s", summary["error"])

    write_run_log(summaries, args.output)

    succeeded = sum(1 for s in summaries if s["status"] == "success")
    logging.info(
        "Done. %d/%d PDFs processed successfully. See run_log.csv for details.",
        succeeded, len(summaries),
    )


if __name__ == "__main__":
    main()
