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
    document_id = _unique("DOC")

    await ingest_document(
        document_id,
        "Some prose about a screw.",
        filename="a.txt",
        product_family="MIS",
        doc_type="Brochure",
    )

    assert await get_graph_client().document_exists(document_id)


async def test_ingest_document_records_family_and_trays_on_the_document_node(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    document_id = _unique("DOC")
    graph = get_graph_client()
    # An older ingestion wrote the legacy `system` property -- re-ingesting
    # replaces it with the family/trays.
    await graph.upsert_document(document_id, "a.txt", doc_type=None, product_family=None)
    async with graph._driver.session() as session:
        await session.run("MATCH (d:Document {id: $id}) SET d.system = 'MIS'", id=document_id)

    await ingest_document(
        document_id,
        "Some prose.",
        filename="a.txt",
        product_family="MIS - Foot Recon System",
        trays=["Tray One", "Tray Two"],
        doc_type="Brochure",
    )

    properties = await graph.get_document_properties(document_id)
    assert properties["product_family"] == "MIS - Foot Recon System"
    assert properties["trays"] == ["Tray One", "Tray Two"]
    assert "system" not in properties


async def test_ingest_document_skips_extraction_without_a_product_family(monkeypatch):
    # A dummy key is enough to clear the OPENAI_API_KEY guard -- the missing
    # `product_family` guard short-circuits before any real LLM call is made,
    # so this stays safe to run without a real key/network access.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")
    document_id = _unique("DOC")

    await ingest_document(
        document_id, "Some prose.", filename="a.txt", product_family=None, doc_type=None
    )

    assert await get_graph_client().document_exists(document_id)


async def test_ingest_document_skips_extraction_when_catalog_not_seeded(monkeypatch):
    # Same reasoning as above: an unseeded family's empty known_parts list
    # short-circuits before any LLM call.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")
    document_id = _unique("DOC")
    unseeded_family = _unique("UNSEEDED-FAMILY")

    await ingest_document(
        document_id,
        "Some prose.",
        filename="a.txt",
        product_family=unseeded_family,
        doc_type="Brochure",
    )

    assert await get_graph_client().document_exists(document_id)
