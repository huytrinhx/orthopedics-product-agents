"""Exercises backend/ingestion/chunking.py -- pure logic, no external
dependencies (no DB, no LLM) needed to test it.
"""
import tiktoken

from ingestion.chunking import chunk_document, strip_page_markers
from ingestion.text_extraction import PAGE_MARKER_TEMPLATE

_ENCODING = tiktoken.get_encoding("cl100k_base")


def _words(n: int, prefix: str = "word") -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


def _tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


def test_short_plain_text_becomes_a_single_chunk():
    text = "First paragraph.\n\nSecond paragraph."
    chunks = chunk_document(text, chunk_size=800, overlap=100)

    assert len(chunks) == 1
    assert chunks[0]["chunk_index"] == 0
    assert chunks[0]["section_title"] is None
    assert "First paragraph." in chunks[0]["content"]
    assert "Second paragraph." in chunks[0]["content"]


def test_paragraphs_pack_until_the_token_budget_then_split():
    paragraphs = [_words(20, f"p{i}_") for i in range(10)]
    text = "\n\n".join(paragraphs)
    chunk_size = _tokens(paragraphs[0]) * 3  # roughly 3 paragraphs per window

    chunks = chunk_document(text, chunk_size=chunk_size, overlap=5)

    assert len(chunks) > 1
    for paragraph in paragraphs:
        assert any(paragraph in c["content"] for c in chunks)
    assert [c["chunk_index"] for c in chunks] == list(range(len(chunks)))


def test_consecutive_chunks_share_overlap_content():
    paragraphs = [_words(20, f"p{i}_") for i in range(6)]
    text = "\n\n".join(paragraphs)
    chunk_size = _tokens(paragraphs[0]) * 2
    overlap = _tokens(paragraphs[0])

    chunks = chunk_document(text, chunk_size=chunk_size, overlap=overlap)

    assert len(chunks) >= 2
    last_paragraph_of_first_chunk = chunks[0]["content"].split("\n\n")[-1]
    assert last_paragraph_of_first_chunk in chunks[1]["content"]


def test_oversized_single_paragraph_is_token_windowed():
    huge = _words(3000)

    chunks = chunk_document(huge, chunk_size=800, overlap=100)

    assert len(chunks) > 1
    for chunk in chunks:
        assert _tokens(chunk["content"]) <= 800
        assert chunk["section_title"] is None


def test_headings_split_into_sections_with_titles():
    text = (
        "# Sizing Guide\n"
        "Use the 4.0mm screw for standard bone density.\n\n"
        "# Sterilization\n"
        "Autoclave at 270F for 4 minutes."
    )

    chunks = chunk_document(text)

    assert [c["section_title"] for c in chunks] == ["Sizing Guide", "Sterilization"]
    assert [c["chunk_index"] for c in chunks] == [0, 1]
    assert "Use the 4.0mm screw" in chunks[0]["content"]
    assert "Autoclave" in chunks[1]["content"]


def test_blank_or_empty_text_yields_no_chunks():
    assert chunk_document("") == []
    assert chunk_document("   \n\n   ") == []


def _marker(page_number: int) -> str:
    return PAGE_MARKER_TEMPLATE.format(page_number=page_number)


def test_plain_text_with_no_markers_has_no_page_numbers():
    text = "First paragraph.\n\nSecond paragraph."
    chunks = chunk_document(text)

    assert all(c["page_number"] is None for c in chunks)


def test_chunk_is_tagged_with_the_page_its_content_starts_on():
    text = (
        f"{_marker(1)}\n\n"
        "# Intro\nPage one content.\n\n"
        f"{_marker(2)}\n\n"
        "# Prep\nPage two content."
    )

    chunks = chunk_document(text)

    assert [c["page_number"] for c in chunks] == [1, 2]
    assert [c["section_title"] for c in chunks] == ["Intro", "Prep"]


def test_page_marker_survives_when_a_heading_is_the_pages_first_line():
    # Regression: a page marker sits *before* the heading line that starts
    # that page's real content (the common real-world shape -- a PDF page
    # that opens directly with a section title). The marker must still be
    # attributed to the section the heading starts, not lost to a
    # disconnected pre-heading section.
    text = (
        f"{_marker(5)}\n\n"
        "# Sterilization\nAutoclave at 270F for 4 minutes."
    )

    chunks = chunk_document(text)

    assert len(chunks) == 1
    assert chunks[0]["page_number"] == 5
    assert chunks[0]["section_title"] == "Sterilization"


def test_multiple_pages_packed_into_one_window_keep_the_starting_page():
    text = (
        "# Overview\n"
        f"{_marker(5)}\n\nShort a.\n\n"
        f"{_marker(6)}\n\nShort b.\n\n"
        f"{_marker(7)}\n\nShort c."
    )

    chunks = chunk_document(text)

    assert len(chunks) == 1
    assert chunks[0]["page_number"] == 5


def test_strip_page_markers_removes_marker_paragraphs_only():
    text = f"{_marker(1)}\n\n# Intro\nPage one content.\n\n{_marker(2)}\n\nPage two content."

    stripped = strip_page_markers(text)

    assert "ORTHOMATE_PAGE_MARKER" not in stripped
    assert "Page one content." in stripped
    assert "Page two content." in stripped
