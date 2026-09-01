"""Exercises backend/ingestion/entity_extraction.py. The "no known parts"
short-circuit needs no LLM call and always runs; the real extraction calls
need OPENAI_API_KEY and are skipped without it -- see ci.yml's note on why
that secret isn't configured there.
"""
import os

import pytest

from ingestion.entity_extraction import extract_entities

needs_openai_key = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), reason="requires a real OPENAI_API_KEY"
)


async def test_extract_entities_returns_empty_without_known_parts():
    assert await extract_entities("some text", known_parts=[], known_trays=[]) == []


@needs_openai_key
async def test_extract_entities_finds_a_real_differentiation():
    known_parts = [
        {"sku": "WIRE-A", "description": "GUIDEWIRE, 1.4MM, SHORT, FOR ACUTE FRACTURES"},
        {"sku": "WIRE-B", "description": "GUIDEWIRE, 1.4MM, LONG, FOR REVISION CASES"},
    ]
    text = (
        "The two 1.4mm guidewires in this set differ only in length: WIRE-A is the "
        "short wire used for acute fracture cases, while WIRE-B is the long wire "
        "reserved for revision cases."
    )

    result = await extract_entities(text, known_parts=known_parts, known_trays=[])

    differentiations = [item for item in result if item["type"] == "differentiation"]
    assert differentiations
    assert {differentiations[0]["sku_a"], differentiations[0]["sku_b"]} == {"WIRE-A", "WIRE-B"}


@needs_openai_key
async def test_extract_entities_ignores_text_about_unmentioned_parts():
    known_parts = [{"sku": "UNRELATED-SKU", "description": "SCREW, 4.0MM"}]
    result = await extract_entities(
        "This paragraph is about tray sterilization procedures, not any specific part.",
        known_parts=known_parts,
        known_trays=[],
    )
    assert result == []
