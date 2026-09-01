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
"""
import tiktoken

from config import tiktoken_cache  # noqa: F401  (sets TIKTOKEN_CACHE_DIR)

DEFAULT_CHUNK_SIZE = 800
DEFAULT_OVERLAP = 100

_ENCODING = tiktoken.get_encoding("cl100k_base")


def chunk_document(
    text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP
) -> list[dict]:
    chunks: list[dict] = []
    index = 0
    for title, body in _split_sections(text):
        paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
        if not paragraphs:
            continue
        for window in _pack_paragraphs(paragraphs, chunk_size, overlap):
            chunks.append({"chunk_index": index, "content": window, "section_title": title})
            index += 1
    return chunks


def _split_sections(text: str) -> list[tuple[str | None, str]]:
    lines = text.splitlines()
    if not any(line.startswith("# ") for line in lines):
        return [(None, text)]

    sections: list[tuple[str | None, str]] = []
    title: str | None = None
    body_lines: list[str] = []
    for line in lines:
        if line.startswith("# "):
            if body_lines:
                sections.append((title, "\n".join(body_lines)))
            title = line[2:].strip()
            body_lines = []
        else:
            body_lines.append(line)
    if body_lines:
        sections.append((title, "\n".join(body_lines)))
    return sections


def _pack_paragraphs(paragraphs: list[str], chunk_size: int, overlap: int) -> list[str]:
    windows: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for paragraph in paragraphs:
        tokens = _token_count(paragraph)

        if tokens > chunk_size:
            if current:
                windows.append("\n\n".join(current))
                current, current_tokens = [], 0
            windows.extend(_token_windows(paragraph, chunk_size, overlap))
            continue

        if current and current_tokens + tokens > chunk_size:
            windows.append("\n\n".join(current))
            current = _overlap_tail(current, overlap)
            current_tokens = sum(_token_count(p) for p in current)

        current.append(paragraph)
        current_tokens += tokens

    if current:
        windows.append("\n\n".join(current))
    return windows


def _overlap_tail(paragraphs: list[str], overlap: int) -> list[str]:
    """The trailing paragraphs (in order) worth up to `overlap` tokens, to
    seed the next window with continuity from the one just closed.
    """
    tail: list[str] = []
    tokens = 0
    for paragraph in reversed(paragraphs):
        if tail and tokens >= overlap:
            break
        tail.insert(0, paragraph)
        tokens += _token_count(paragraph)
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
