"""Splits ingested documents into retrieval-sized chunks.

Operates on text already normalized by backend/ingestion/text_extraction.py:
a line prefixed with `"# "` is a detected heading (PDF: font-size heuristic,
`.md`: literal Markdown syntax). Text is split into sections on those
headings first; a document with no heading markers at all (plain `.txt`,
where no such signal exists) is treated as a single unheaded section, and
splits on blank-line paragraph boundaries instead -- the closest available
structure signal in that case.

Each section's paragraphs are then packed into ~800-token windows (tiktoken,
matching text-embedding-3-small's token accounting) with a 100-token
overlap; a single paragraph that alone exceeds the window budget is
token-windowed on its own.

Ticket 26: a PDF's text also carries text_extraction.py's page-marker
paragraphs (PAGE_MARKER_PATTERN below) -- consumed here to tag every real
paragraph with the page it came from, so each output chunk can carry the
page its content *starts* on (never split at a page boundary -- a window
that starts mid-page and bleeds onto the next keeps the starting page's
number, which is what a "jump to this page" viewer actually needs). Plain
`.txt`/`.md` text has no markers at all, so every paragraph there is
untagged (`page_number: None`) exactly as before this ticket.

Page tagging is done as a single top-to-bottom pass over the *whole*
document (`_tag_lines_with_pages`), before section-splitting -- not
per-section. A PDF page very often starts with a heading as its first real
content line (e.g. "# Preparation & Insertion"), which would otherwise
split a page's marker from that page's own content into two different
`_split_sections()` sections (the marker landing in the section *before*
the heading), silently losing the page tag for everything the marker was
meant to describe. Tagging globally first means the marker/heading order
in the raw text no longer matters.
"""
import re

import tiktoken

from config import tiktoken_cache  # noqa: F401  (sets TIKTOKEN_CACHE_DIR)
from ingestion.text_extraction import PAGE_MARKER_TEMPLATE

DEFAULT_CHUNK_SIZE = 800
DEFAULT_OVERLAP = 100

_ENCODING = tiktoken.get_encoding("cl100k_base")

# Built from the same template text_extraction.py formats with, so the two
# modules can never drift apart on what a page marker looks like.
PAGE_MARKER_PATTERN = re.compile(
    "^" + re.escape(PAGE_MARKER_TEMPLATE).replace(r"\{page_number\}", r"(\d+)") + "$"
)

_Line = tuple[str, int | None]
_Paragraph = tuple[str, int | None]


def chunk_document(
    text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP
) -> list[dict]:
    tagged_lines = _tag_lines_with_pages(text)
    chunks: list[dict] = []
    index = 0
    for title, body_lines in _split_sections(tagged_lines):
        paragraphs = _group_into_paragraphs(body_lines)
        if not paragraphs:
            continue
        for window, page_number in _pack_paragraphs(paragraphs, chunk_size, overlap):
            chunks.append(
                {
                    "chunk_index": index,
                    "content": window,
                    "section_title": title,
                    "page_number": page_number,
                }
            )
            index += 1
    return chunks


def strip_page_markers(text: str) -> str:
    """Removes text_extraction.py's page-marker paragraphs -- for a
    consumer (ingestion/pipeline.py's entity-extraction leg) that wants the
    document's real content without this module's internal page-tracking
    annotation leaking into an LLM prompt.
    """
    kept = [p for p in text.split("\n\n") if not PAGE_MARKER_PATTERN.match(p.strip())]
    return "\n\n".join(kept)


def _tag_lines_with_pages(text: str) -> list[_Line]:
    """One global top-to-bottom pass over every line, carrying the most
    recently seen page marker forward -- independent of where
    `_split_sections` will later cut the text into sections. Marker lines
    themselves are dropped here (not just filtered downstream).
    """
    current_page: int | None = None
    tagged: list[_Line] = []
    for line in text.splitlines():
        match = PAGE_MARKER_PATTERN.match(line.strip())
        if match:
            current_page = int(match.group(1))
            continue
        tagged.append((line, current_page))
    return tagged


def _split_sections(tagged_lines: list[_Line]) -> list[tuple[str | None, list[_Line]]]:
    if not any(line.startswith("# ") for line, _ in tagged_lines):
        return [(None, tagged_lines)]

    sections: list[tuple[str | None, list[_Line]]] = []
    title: str | None = None
    body: list[_Line] = []
    for line, page in tagged_lines:
        if line.startswith("# "):
            if body:
                sections.append((title, body))
            title = line[2:].strip()
            body = []
        else:
            body.append((line, page))
    if body:
        sections.append((title, body))
    return sections


def _group_into_paragraphs(tagged_lines: list[_Line]) -> list[_Paragraph]:
    """Blank-line-delimited paragraph grouping (a run of one or more blank
    lines separates paragraphs, matching the old `body.split("\\n\\n")`
    behavior), tagging each paragraph with its *first* line's page.
    """
    paragraphs: list[_Paragraph] = []
    current_lines: list[str] = []
    current_page: int | None = None
    for line, page in tagged_lines:
        if line.strip() == "":
            if current_lines:
                paragraphs.append(("\n".join(current_lines), current_page))
                current_lines = []
            continue
        if not current_lines:
            current_page = page
        current_lines.append(line)
    if current_lines:
        paragraphs.append(("\n".join(current_lines), current_page))
    return paragraphs


def _pack_paragraphs(
    paragraphs: list[_Paragraph], chunk_size: int, overlap: int
) -> list[_Paragraph]:
    """Same windowing as before, now over (paragraph, page) pairs -- each
    output window is tagged with its *first* paragraph's page number, since
    that's the page the window's content starts on.
    """
    windows: list[_Paragraph] = []
    current: list[_Paragraph] = []
    current_tokens = 0

    def flush() -> None:
        if current:
            windows.append(("\n\n".join(p for p, _ in current), current[0][1]))

    for paragraph, page in paragraphs:
        tokens = _token_count(paragraph)

        if tokens > chunk_size:
            flush()
            current, current_tokens = [], 0
            windows.extend((w, page) for w in _token_windows(paragraph, chunk_size, overlap))
            continue

        if current and current_tokens + tokens > chunk_size:
            flush()
            current = _overlap_tail(current, overlap)
            current_tokens = sum(_token_count(p) for p, _ in current)

        current.append((paragraph, page))
        current_tokens += tokens

    flush()
    return windows


def _overlap_tail(paragraphs: list[_Paragraph], overlap: int) -> list[_Paragraph]:
    """The trailing paragraphs (in order) worth up to `overlap` tokens, to
    seed the next window with continuity from the one just closed.
    """
    tail: list[_Paragraph] = []
    tokens = 0
    for item in reversed(paragraphs):
        if tail and tokens >= overlap:
            break
        tail.insert(0, item)
        tokens += _token_count(item[0])
    return tail


def _token_windows(text: str, chunk_size: int, overlap: int) -> list[str]:
    tokens = _ENCODING.encode(text)
    if not tokens:
        return []
    step = max(chunk_size - overlap, 1)
    windows: list[str] = []
    start = 0
    while True:
        window = tokens[start : start + chunk_size]
        windows.append(_ENCODING.decode(window))
        if start + chunk_size >= len(tokens):
            break
        start += step
    return windows


def _token_count(text: str) -> int:
    return len(_ENCODING.encode(text))
