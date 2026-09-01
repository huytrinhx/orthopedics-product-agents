"""Exercises backend/retrieval/graph_client.py against a real Neo4j -- no
mocking, same approach as test_documents.py/test_tags.py against Postgres.
Every test uses a fresh uuid-suffixed sku/name so tests don't collide with
each other or with real seeded data. Async tests run on the session-scoped
loop (backend/pyproject.toml's asyncio_default_test_loop_scope) so they all
share the same event loop as get_graph_client()'s cached driver.
"""
import uuid

from retrieval.graph_client import get_graph_client


def _unique(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


async def test_upsert_part_creates_tray_and_family_chain():
    client = get_graph_client()
    await client.ensure_constraints()
    family = _unique("FAMILY")
    tray = _unique("TRAY")
    sku = _unique("SKU")

    await client.upsert_product_family(family)
    await client.upsert_tray(tray, family)
    await client.upsert_part(sku, tray, description="test part", thread="Full")

    related = await client.query_related_entities(sku)
    trays = [r for r in related if r["relationship"] == "BELONGS_TO_TRAY"]
    assert trays and trays[0]["related_entity"] == tray


async def test_compatible_with_is_bidirectionally_queryable():
    client = get_graph_client()
    tray = _unique("TRAY")
    plate_sku = _unique("PLATE")
    screw_sku = _unique("SCREW")

    await client.upsert_tray(tray, None)
    await client.upsert_part(plate_sku, tray)
    await client.upsert_part(screw_sku, tray)
    await client.upsert_compatible_with(plate_sku, screw_sku)

    from_plate = await client.query_related_entities(plate_sku, "COMPATIBLE_WITH")
    from_screw = await client.query_related_entities(screw_sku, "COMPATIBLE_WITH")
    assert any(r["related_entity"] == screw_sku for r in from_plate)
    assert any(r["related_entity"] == plate_sku for r in from_screw)


async def test_differentiation_noops_when_a_sku_is_unknown():
    client = get_graph_client()
    tray = _unique("TRAY")
    known_sku = _unique("KNOWN")
    unknown_sku = _unique("UNKNOWN")
    doc_id = _unique("DOC")

    await client.upsert_document(doc_id, "brochure.pdf", doc_type="Brochure", system="MIS")
    await client.upsert_tray(tray, "MIS")
    await client.upsert_part(known_sku, tray)
    # unknown_sku is never upserted as a Part -- a hallucinated/unresolved
    # SKU from prose extraction must not mint a new node.
    attached = await client.attach_differentiation(known_sku, unknown_sku, "explanation text", doc_id)
    related = await client.query_related_entities(known_sku, "DIFFERENTIATES_FROM")

    assert attached is False
    assert related == []


async def test_differentiation_attaches_and_is_queryable_both_directions():
    client = get_graph_client()
    tray = _unique("TRAY")
    sku_a = _unique("A")
    sku_b = _unique("B")
    doc_id = _unique("DOC")

    await client.upsert_document(doc_id, "brochure.pdf", doc_type="Brochure", system="MIS")
    await client.upsert_tray(tray, "MIS")
    await client.upsert_part(sku_a, tray)
    await client.upsert_part(sku_b, tray)
    attached = await client.attach_differentiation(sku_a, sku_b, "A is shorter than B", doc_id)
    related = await client.query_related_entities(sku_a, "DIFFERENTIATES_FROM")

    assert attached is True
    assert related[0]["related_entity"] == sku_b
    assert related[0]["relationship_properties"]["explanation"] == "A is shorter than B"


async def test_procedure_requires_tray_noops_for_unknown_tray():
    client = get_graph_client()
    doc_id = _unique("DOC")
    procedure = _unique("PROCEDURE")
    unknown_tray = _unique("UNKNOWN-TRAY")

    await client.upsert_document(doc_id, "guide.pdf", doc_type="Surgical Technique", system="MIS")
    assert await client.attach_procedure(procedure, unknown_tray, doc_id) is False


async def test_synonym_groups_include_canonical_and_both_edge_types():
    client = get_graph_client()
    canonical = _unique("CANONICAL")
    alias = _unique("ALIAS")
    abbreviation = _unique("ABBR")

    await client.upsert_synonym_cluster(canonical, [alias], notes="test note")
    await client.upsert_abbreviation(canonical, abbreviation)
    groups = await client.get_synonym_groups()

    group = next(g for g in groups if g[0] == canonical)
    assert alias in group
    assert abbreviation in group
