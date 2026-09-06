"""Exercises backend/agents/citations.py's extract_citations directly --
pure regex parsing, no DB/LLM needed.
"""
from agents.citations import extract_citations

_ID_A = "0bb5d399-6661-4790-943a-c965e97c5809"
_ID_B = "e54d9a4b-cac6-4d10-a84d-e86112f9538d"


def test_extracts_a_single_bracketed_citation():
    assert extract_citations(f"A claim [{_ID_A}#3].") == [f"{_ID_A}#3"]


def test_extracts_two_separately_bracketed_citations():
    assert extract_citations(f"One [{_ID_A}#3]. Two [{_ID_B}#5].") == [
        f"{_ID_A}#3",
        f"{_ID_B}#5",
    ]


def test_extracts_adjacent_brackets_with_no_space_between_them():
    assert extract_citations(f"Both [{_ID_A}#3][{_ID_B}#5] support this.") == [
        f"{_ID_A}#3",
        f"{_ID_B}#5",
    ]


def test_extracts_multiple_refs_combined_in_one_bracket():
    """Found live (2026-09-05): the model sometimes cites two sources for
    one claim as a single bracket with a comma between them instead of two
    separate brackets -- the original single-ref-per-bracket pattern left
    the whole bracket unmatched, silently dropping both citations (not just
    a display glitch -- the chip for either source never appeared)."""
    assert extract_citations(f"Both sources agree [{_ID_A}#3, {_ID_B}#5].") == [
        f"{_ID_A}#3",
        f"{_ID_B}#5",
    ]


def test_extracts_three_refs_combined_in_one_bracket():
    assert extract_citations(f"[{_ID_A}#3, {_ID_B}#5, {_ID_A}#7]") == [
        f"{_ID_A}#3",
        f"{_ID_B}#5",
        f"{_ID_A}#7",
    ]


def test_dedupes_the_same_citation_seen_twice_preserving_first_seen_order():
    assert extract_citations(f"[{_ID_B}#5] again here [{_ID_A}#3] and again [{_ID_B}#5].") == [
        f"{_ID_B}#5",
        f"{_ID_A}#3",
    ]


def test_ignores_a_bracketed_non_uuid_identifier():
    """A SKU cited inline without brackets is the convention (ticket 22's
    catalog facts); a bracketed non-UUID string (e.g. a hallucinated or
    malformed reference) must never reach api/routes/chat.py's
    _resolve_citations, which crashes on a non-UUID (see that module's
    docstring)."""
    assert extract_citations("See [MGT14150#0] for the spec.") == []


def test_no_citations_returns_an_empty_list():
    assert extract_citations("No sources cited here at all.") == []
