"""Exercises backend/ingestion/pipeline.py's ingest_document against real
Neo4j -- no mocking. The LLM-calling path (needs OPENAI_API_KEY, costs
money) is exercised separately in test_entity_extraction.py, skipped there
when that key isn't configured; here we only exercise the guard rails that
don't need a real LLM call, so this file runs in CI with no secret.
"""
import uuid

from ingestion.pipeline import ingest_document
from retrieval.graph_client import get_graph_client


def _unique(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


async def test_ingest_document_upserts_document_even_without_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    doc_id = _unique("DOC")

    await ingest_document(
        doc_id, "Some prose about a screw.", filename="a.txt", system="MIS", doc_type="Brochure"
    )

    assert await get_graph_client().document_exists(doc_id)


async def test_ingest_document_skips_extraction_without_a_system(monkeypatch):
    # A dummy key is enough to clear the OPENAI_API_KEY guard -- the missing
    # `system` guard short-circuits before any real LLM call is made, so
    # this stays safe to run without a real key/network access.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")
    doc_id = _unique("DOC")

    await ingest_document(doc_id, "Some prose.", filename="a.txt", system=None, doc_type=None)

    assert await get_graph_client().document_exists(doc_id)


async def test_ingest_document_skips_extraction_when_catalog_not_seeded(monkeypatch):
    # Same reasoning as above: an unseeded system's empty known_parts list
    # short-circuits before any LLM call.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")
    doc_id = _unique("DOC")
    unseeded_system = _unique("UNSEEDED-SYSTEM")

    await ingest_document(
        doc_id, "Some prose.", filename="a.txt", system=unseeded_system, doc_type="Brochure"
    )

    assert await get_graph_client().document_exists(doc_id)
