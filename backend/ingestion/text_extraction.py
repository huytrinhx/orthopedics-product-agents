"""Extracts plain text from an uploaded document's raw bytes on disk.

Normalizes detected structure (PDF section headings, found via font size)
into a Markdown-style `# ` line prefix, so backend/ingestion/chunking.py has
one uniform signal to split on regardless of source format -- a `.md` file
already uses that syntax natively, and a plain `.txt` file has no heading
signal at all (chunking falls back to paragraph boundaries there).

Shared by both ingestion legs (backend/documents/service.py calls this once
per document): the graph leg's entity extraction and this ticket's chunking
+ embedding both start from the same extracted text.
"""
import statistics
from pathlib import Path

import pdfplumber

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
    pages: list[str] = []
    with pdfplumber.open(storage_path) as pdf:
        for page in pdf.pages:
            lines = _lines_with_font_size(page)
            if not lines:
                continue
            body_size = statistics.median(size for _, size in lines)
            pages.append(
                "\n".join(
                    f"# {text}" if size > body_size * _HEADING_SIZE_RATIO else text
                    for text, size in lines
                )
            )
    return "\n\n".join(pages)


def _lines_with_font_size(page) -> list[tuple[str, float]]:
    """Groups pdfplumber's word-level output back into lines (by rounded
    vertical position) and takes each line's median character size as its
    representative font size.
    """
    words = page.extract_words(extra_attrs=["size"])
    if not words:
        return []
    rows: dict[int, list[dict]] = {}
    for word in words:
        rows.setdefault(round(word["top"]), []).append(word)
    lines: list[tuple[str, float]] = []
    for top in sorted(rows):
        row = sorted(rows[top], key=lambda w: w["x0"])
        text = " ".join(w["text"] for w in row)
        size = statistics.median(w["size"] for w in row)
        lines.append((text, size))
    return lines
