"""Exercises backend/ingestion/chunking.py -- pure logic, no external
dependencies (no DB, no LLM) needed to test it.
"""
import tiktoken

from ingestion.chunking import chunk_document

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
