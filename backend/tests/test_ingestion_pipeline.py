"""Exercises backend/ingestion/pipeline.py's ingest_document against real
Neo4j -- no mocking. The LLM-calling path (needs OPENAI_API_KEY, costs
money) is exercised separately in test_entity_extraction.py, skipped there
when that key isn't configured; here we only exercise the guard rails that
don't need a real LLM call, so this file runs in CI with no secret. The
one exception swaps extract_entities for a stub, to check what scope it's
given and that a re-index replaces its edges (issue #28).
"""
import uuid

from auth.repository import create_user
from document_skus.repository import DocumentSkuRow, replace_document_skus
from documents.repository import create_document, delete_document
from ingestion import pipeline
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


async def test_extraction_scope_widens_to_indexed_families_and_reindex_replaces_edges(monkeypatch):
    """Issue #28: the known parts are the family's plus every family the
    SKU index found parts of, and a re-index leaves only the new run's
    differentiations. The model is stubbed out."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    graph = get_graph_client()
    family, other_family = _unique("FAMILY"), _unique("OTHER-FAMILY")
    family_tray, other_tray = _unique("TRAY"), _unique("TRAY")
    family_sku, other_sku, unrelated_sku = _unique("SKU"), _unique("SKU"), _unique("SKU")
    await graph.upsert_tray(family_tray, family)
    await graph.upsert_tray(other_tray, other_family)
    await graph.upsert_part(family_sku, family_tray, description="FAMILY PART")
    await graph.upsert_part(other_sku, other_tray, description="OTHER FAMILY PART")
    await graph.upsert_part(unrelated_sku, _unique("TRAY"), description="NEVER MENTIONED")

    user = await create_user(f"user-{uuid.uuid4().hex[:10]}@example.com", None, False, True)
    document = await create_document(
        filename="reflex.pdf", storage_path=f"/tmp/{uuid.uuid4().hex}.pdf", uploaded_by=user.id
    )
    document_id = str(document.id)
    await replace_document_skus(
        document.id, [DocumentSkuRow(0, other_sku, other_sku, "exact", False, None, other_sku)]
    )

    calls: list[list[str]] = []
    explanations = iter(["first run", "second run"])

    async def fake_extract_entities(chunk_text, *, known_parts, slot_model=None):
        calls.append(sorted(part["sku"] for part in known_parts))
        return [
            {"type": "differentiation", "sku_a": family_sku, "sku_b": other_sku,
             "explanation": next(explanations)}
        ]

    monkeypatch.setattr(pipeline, "extract_entities", fake_extract_entities)
    try:
        for _ in range(2):
            await pipeline.ingest_document(
                document_id, "One paragraph.", filename="reflex.pdf", product_family=family,
                doc_type=None,
            )

        assert calls == [sorted([family_sku, other_sku])] * 2
        [edge] = await graph.query_related_entities(family_sku, "DIFFERENTIATES_FROM")
        assert edge["relationship_properties"] == {
            "document_id": document_id,
            "explanation": "second run",
        }
    finally:
        await delete_document(document.id)
