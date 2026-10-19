"""Extracts plain text from an uploaded document's raw bytes on disk.

Normalizes detected structure (PDF section headings, found via font size)
into a Markdown-style `# ` line prefix, so backend/ingestion/chunking.py has
one uniform signal to split on regardless of source format -- a `.md` file
already uses that syntax natively, and a plain `.txt` file has no heading
signal at all (chunking falls back to paragraph boundaries there).

Shared by both ingestion legs (backend/documents/service.py calls this once
per document): the graph leg's entity extraction and this ticket's chunking
+ embedding both start from the same extracted text.

Ticket 26: a PDF's page boundaries are threaded through as inline marker
paragraphs (ingestion/chunking.py's PAGE_MARKER_PATTERN) rather than
changing this function's return type to something richer -- extract_text's
plain-`str` signature is depended on by both ingestion legs
(documents/service.py calls it once and hands the same string to each), and
only the vector leg (chunking) actually needs page attribution; the graph
leg (entity extraction) strips the markers back out before using the text
(ingestion/pipeline.py's ingest_document) rather than ever seeing them.
"""
import statistics
from pathlib import Path

import pdfplumber

# A blank-line-delimited paragraph of exactly this shape, inserted before
# each PDF page's own text -- ingestion/chunking.py parses these out to
# attribute a page number to every chunk. Plain ASCII (safe in a Postgres
# text column and any JSON payload, unlike a NUL or other control
# character) and deliberately unlikely to collide with real document text.
PAGE_MARKER_TEMPLATE = "@@ORTHOMATE_PAGE_MARKER_{page_number}@@"

# A line's font size must exceed the page's median body-text size by more
# than this multiplier to count as a heading. Font-size-based rather than
# bold-based: technical documents bold inline implant/product names
# constantly, which would misfire as false section breaks.
_HEADING_SIZE_RATIO = 1.15


class UnsupportedDocumentFormat(Exception):
    """Raised for a file extension this pipeline doesn't know how to read."""


def extract_text(storage_path: str | Path, filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(storage_path)
    if suffix in (".txt", ".md"):
        return Path(storage_path).read_text(encoding="utf-8")
    raise UnsupportedDocumentFormat(
        f"Unsupported document format: {suffix or '(no extension)'!r}"
    )


def _extract_pdf(storage_path: str | Path) -> str:
    parts: list[str] = []
    with pdfplumber.open(storage_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            lines = _lines_with_font_size(page)
            if not lines:
                continue
            body_size = statistics.median(size for _, size in lines)
            page_text = "\n".join(
                f"# {text}" if size > body_size * _HEADING_SIZE_RATIO else text
                for text, size in lines
            )
            # Its own blank-line-delimited paragraph (not tacked onto
            # page_text's first line) so chunking.py's paragraph splitter
            # isolates it cleanly regardless of where headings fall.
            parts.append(PAGE_MARKER_TEMPLATE.format(page_number=page_number))
            parts.append(page_text)
    return "\n\n".join(parts)


def _lines_with_font_size(page) -> list[tuple[str, float]]:
    """Groups pdfplumber's word-level output back into lines (by rounded
    vertical position) and takes each line's median character size as its
    representative font size.

    Issue #26: doubled-font text (a word drawn twice, slightly offset, for
    a bold or shadow effect) is deduplicated first, and a tray page laid
    out as several side-by-side tables is read one column at a time (see
    _table_column_starts), so a row's SKU and description stay together.
    """
    # dedupe_chars only merges copies of the same size; a copy at a
    # slightly different size comes through as a separate word instead
    # ("MEDLINE MEDLINE UNITE UNITE"), which _drop_overlapping_copies takes.
    words = _drop_overlapping_copies(page.dedupe_chars().extract_words(extra_attrs=["size"]))
    if not words:
        return []
    column_starts, table_top = _table_column_starts(words)
    if not column_starts:
        return _rows_by_rounded_top(words)
    above_table = [word for word in words if word["top"] < table_top]
    lines = _rows_by_rounded_top(above_table)
    boundaries = [start - _COLUMN_MARGIN for start in column_starts[1:]]
    for band_index in range(len(column_starts)):
        low = boundaries[band_index - 1] if band_index > 0 else float("-inf")
        high = boundaries[band_index] if band_index < len(boundaries) else float("inf")
        band = [word for word in words if word["top"] >= table_top and low <= word["x0"] < high]
        lines.extend(_rows_by_overlap(band))
    return lines


# Two words with the same text, tops this close (in points) and boxes
# overlapping by this share are one word drawn twice.
_COPY_TOLERANCE = 2.0
_COPY_OVERLAP_SHARE = 0.5
# A column's words can sit slightly left of its "Item No." header.
_COLUMN_MARGIN = 4.0
# Two "Item No." headers this close vertically are one header row; this
# far apart horizontally they head different columns.
_HEADER_ROW_TOLERANCE = 6.0
_MIN_COLUMN_GAP = 40.0
# Within a column, a word whose top is within this share of the row's
# height of the row's own top belongs to that row: a description printed a
# few points above its SKU still lands on the SKU's line.
_ROW_OVERLAP_SHARE = 0.6


def _is_copy_of(word: dict, other: dict) -> bool:
    """Same text, drawn over itself: the boxes overlap by at least half the
    narrower word's width. A word legitimately repeated side by side starts
    where the other ends, so never overlaps it this much."""
    if word["text"] != other["text"] or abs(word["top"] - other["top"]) > _COPY_TOLERANCE:
        return False
    overlap = min(word["x1"], other["x1"]) - max(word["x0"], other["x0"])
    narrower = min(word["x1"] - word["x0"], other["x1"] - other["x0"])
    return overlap >= narrower * _COPY_OVERLAP_SHARE


def _drop_overlapping_copies(words: list[dict]) -> list[dict]:
    kept: list[dict] = []
    for word in words:
        if not any(_is_copy_of(word, other) for other in kept):
            kept.append(word)
    return kept


def _line_from_words(row: list[dict]) -> tuple[str, float]:
    ordered = sorted(row, key=lambda word: word["x0"])
    return " ".join(word["text"] for word in ordered), statistics.median(word["size"] for word in ordered)


def _rows_by_rounded_top(words: list[dict]) -> list[tuple[str, float]]:
    rows: dict[int, list[dict]] = {}
    for word in words:
        rows.setdefault(round(word["top"]), []).append(word)
    return [_line_from_words(rows[top]) for top in sorted(rows)]


def _rows_by_overlap(words: list[dict]) -> list[tuple[str, float]]:
    rows: list[list[dict]] = []
    for word in sorted(words, key=lambda word: (word["top"], word["x0"])):
        if rows:
            row_top = min(other["top"] for other in rows[-1])
            row_height = statistics.median(other["bottom"] - other["top"] for other in rows[-1])
            if word["top"] - row_top <= row_height * _ROW_OVERLAP_SHARE:
                rows[-1].append(word)
                continue
        rows.append([word])
    return [_line_from_words(row) for row in rows]


def _table_column_starts(words: list[dict]) -> tuple[list[float], float]:
    """The left edge of each side-by-side table on a tray page, and the top
    of the region to read column by column; ([], 0) for any other page.

    A tray page's tables each have an "Item No. Description Qty." header,
    so two or more "Item No." headers on one row mark a multi-column page.
    Read row by row, such a page interleaves its columns' rows (the
    production Core Foot text put a reamer's "ø16 mm" on a screw's line).
    The caption line directly above the header row joins the column region
    when none of its words straddles a column edge.

    pdfplumber's own extract_tables was tried for this and left out: on
    this layout (no ruling lines, offset rows) its text strategy splits
    cells mid-word ("Item No. Des" | "cription") and still leaves an offset
    description on a row of its own.
    """
    header_words = [
        word
        for word in words
        if word["text"].casefold() == "item"
        and any(
            other["text"].casefold().rstrip(".") in {"no", "number", "#"}
            and abs(other["top"] - word["top"]) < 1
            and 0 <= other["x0"] - word["x1"] < 10
            for other in words
        )
    ]
    header_words.sort(key=lambda word: word["top"])
    for first in header_words:
        row = [word for word in header_words if 0 <= word["top"] - first["top"] <= _HEADER_ROW_TOLERANCE]
        starts = _distinct_starts(row)
        if len(starts) < 2:
            continue
        # Later sub-tables can open further columns.
        starts = _distinct_starts([*row, *header_words])
        table_top = min(word["top"] for word in row)
        return starts, _include_caption_line(words, starts, table_top)
    return [], 0.0


def _distinct_starts(header_words: list[dict]) -> list[float]:
    starts: list[float] = []
    for x0 in sorted(word["x0"] for word in header_words):
        if not starts or x0 - starts[-1] >= _MIN_COLUMN_GAP:
            starts.append(x0)
    return starts


def _include_caption_line(words: list[dict], starts: list[float], table_top: float) -> float:
    above = [word for word in words if word["top"] < table_top]
    if not above:
        return table_top
    caption_top = max(round(word["top"]) for word in above)
    caption = [word for word in above if round(word["top"]) == caption_top]
    edges = [start - _COLUMN_MARGIN for start in starts[1:]]
    straddles = any(word["x0"] < edge < word["x1"] for word in caption for edge in edges)
    line_height = statistics.median(word["bottom"] - word["top"] for word in caption)
    if straddles or table_top - caption_top > line_height * 2:
        return table_top
    return min(word["top"] for word in caption)
