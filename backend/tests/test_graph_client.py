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
    trays = [related_entry for related_entry in related if related_entry["relationship"] == "BELONGS_TO_TRAY"]
    assert trays and trays[0]["related_entity"] == tray


async def test_compatible_with_is_bidirectionally_queryable():
    client = get_graph_client()
    tray = _unique("TRAY")
    plate_sku = _unique("PLATE")
    screw_sku = _unique("SCREW")

    await client.upsert_tray(tray, _unique("FAMILY"))
    await client.upsert_part(plate_sku, tray)
    await client.upsert_part(screw_sku, tray)
    await client.upsert_compatible_with(plate_sku, screw_sku)

    from_plate = await client.query_related_entities(plate_sku, "COMPATIBLE_WITH")
    from_screw = await client.query_related_entities(screw_sku, "COMPATIBLE_WITH")
    assert any(related_entry["related_entity"] == screw_sku for related_entry in from_plate)
    assert any(related_entry["related_entity"] == plate_sku for related_entry in from_screw)


async def test_batched_upserts_write_every_row_across_batch_boundaries(monkeypatch):
    """A batch size of 2 over 5 parts / 4 pairs forces partial final
    batches, so an off-by-one in the slicing would drop a row."""
    monkeypatch.setattr("retrieval.graph_client.UPSERT_BATCH_SIZE", 2)
    client = get_graph_client()
    tray = _unique("TRAY")
    await client.upsert_tray(tray, _unique("FAMILY"))
    plate_sku = _unique("PLATE")
    screw_skus = [_unique("SCREW") for _ in range(4)]

    await client.upsert_parts(
        [(plate_sku, tray, {"description": "batched plate", "thread": None})]
        + [(screw_sku, tray, {"description": "batched screw"}) for screw_sku in screw_skus]
    )
    await client.upsert_compatible_with_pairs([(plate_sku, screw_sku) for screw_sku in screw_skus])
    await client.upsert_requires_tool_pairs([(screw_sku, plate_sku) for screw_sku in screw_skus])

    from_plate = await client.query_related_entities(plate_sku)
    compatible = {entry["related_entity"] for entry in from_plate if entry["relationship"] == "COMPATIBLE_WITH"}
    required_by = {entry["related_entity"] for entry in from_plate if entry["relationship"] == "REQUIRES_TOOL"}
    assert compatible == set(screw_skus)
    assert required_by == set(screw_skus)
    for screw_sku in screw_skus:
        trays = [
            entry["related_entity"]
            for entry in await client.query_related_entities(screw_sku)
            if entry["relationship"] == "BELONGS_TO_TRAY"
        ]
        assert trays == [tray]


async def test_differentiation_noops_when_a_sku_is_unknown():
    client = get_graph_client()
    tray = _unique("TRAY")
    known_sku = _unique("KNOWN")
    unknown_sku = _unique("UNKNOWN")
    document_id = _unique("DOC")

    await client.upsert_document(document_id, "brochure.pdf", doc_type="Brochure", product_family="MIS")
    await client.upsert_tray(tray, "MIS")
    await client.upsert_part(known_sku, tray)
    # unknown_sku is never upserted as a Part -- a hallucinated/unresolved
    # SKU from prose extraction must not mint a new node.
    attached = await client.attach_differentiation(known_sku, unknown_sku, "explanation text", document_id)
    related = await client.query_related_entities(known_sku, "DIFFERENTIATES_FROM")

    assert attached is False
    assert related == []


async def test_differentiation_attaches_and_is_queryable_both_directions():
    client = get_graph_client()
    tray = _unique("TRAY")
    sku_a = _unique("A")
    sku_b = _unique("B")
    document_id = _unique("DOC")

    await client.upsert_document(document_id, "brochure.pdf", doc_type="Brochure", product_family="MIS")
    await client.upsert_tray(tray, "MIS")
    await client.upsert_part(sku_a, tray)
    await client.upsert_part(sku_b, tray)
    attached = await client.attach_differentiation(sku_a, sku_b, "A is shorter than B", document_id)
    related = await client.query_related_entities(sku_a, "DIFFERENTIATES_FROM")

    assert attached is True
    assert related[0]["related_entity"] == sku_b
    assert related[0]["relationship_properties"]["explanation"] == "A is shorter than B"


async def test_procedure_requires_tray_noops_for_unknown_tray():
    client = get_graph_client()
    document_id = _unique("DOC")
    procedure = _unique("PROCEDURE")
    unknown_tray = _unique("UNKNOWN-TRAY")

    await client.upsert_document(document_id, "guide.pdf", doc_type="Surgical Technique", product_family="MIS")
    assert await client.attach_procedure(procedure, unknown_tray, document_id) is False


async def test_find_parts_returns_each_parts_trays_once_per_part():
    """Issue #21: a Part in two trays of its family comes back as one row
    carrying both trays, not one row per tray."""
    client = get_graph_client()
    family = _unique("FAMILY")
    first_tray, second_tray = sorted([_unique("TRAY"), _unique("TRAY")])
    sku = _unique("SKU")
    await client.upsert_tray(first_tray, family)
    await client.upsert_tray(second_tray, family)
    await client.upsert_part(sku, first_tray, description="PICKUPS")
    await client.upsert_part(sku, second_tray, description="PICKUPS")

    found = await client.find_parts(sku, product_family=family)

    assert [(part["sku"], part["trays"]) for part in found] == [(sku, [first_tray, second_tray])]


async def test_list_part_descriptions_by_tray_groups_one_familys_parts():
    client = get_graph_client()
    family = _unique("FAMILY")
    tray = _unique("TRAY")
    await client.upsert_tray(tray, family)
    await client.upsert_part(_unique("SKU"), tray, description="SCREW, CANN, HEADED, 5.5 X 40MM")

    assert await client.list_part_descriptions_by_tray(family) == {tray: ["SCREW, CANN, HEADED, 5.5 X 40MM"]}


async def test_list_catalog_parts_gives_every_family_a_part_sits_in():
    """Issue #27: the SKU index validates against the whole catalog, so a
    part in trays of two families carries both."""
    client = get_graph_client()
    first_family, second_family = sorted([_unique("FAMILY"), _unique("FAMILY")])
    first_tray, second_tray = _unique("TRAY"), _unique("TRAY")
    sku = _unique("SKU")
    await client.upsert_tray(first_tray, first_family)
    await client.upsert_tray(second_tray, second_family)
    await client.upsert_part(sku, first_tray, description="SCREW, LOCKING, 3.5 X 40MM")
    await client.upsert_part(sku, second_tray, description="SCREW, LOCKING, 3.5 X 40MM")

    found = await client.list_catalog_parts(skus=[sku])

    assert [(part["sku"], part["description"], sorted(part["families"])) for part in found] == [
        (sku, "SCREW, LOCKING, 3.5 X 40MM", [first_family, second_family])
    ]
    assert any(part["sku"] == sku for part in await client.list_catalog_parts())


async def test_list_part_descriptions_by_family_counts_a_part_in_two_trays_once():
    # Issue #30: what the family descriptions are generated from.
    client = get_graph_client()
    family, other_family, unasked_family = _unique("FAMILY"), _unique("FAMILY"), _unique("FAMILY")
    first_tray, second_tray, other_tray, unasked_tray = (_unique("TRAY") for _ in range(4))
    await client.upsert_tray(first_tray, family)
    await client.upsert_tray(second_tray, family)
    await client.upsert_tray(other_tray, other_family)
    await client.upsert_tray(unasked_tray, unasked_family)
    shared_sku = _unique("SKU")
    await client.upsert_part(shared_sku, first_tray, description="STAPLE, REFLEX TETRA, LAPIDUS, 30MM")
    await client.upsert_part(shared_sku, second_tray, description="STAPLE, REFLEX TETRA, LAPIDUS, 30MM")
    await client.upsert_part(_unique("SKU"), second_tray, description="DYNAMIC DISC, NITINOL, REFLEX, 7MM")
    await client.upsert_part(_unique("SKU"), other_tray, description="BURR, BROPHY, 2.0X12MM")
    await client.upsert_part(_unique("SKU"), unasked_tray, description="PICKUPS")

    found = await client.list_part_descriptions_by_family([family, other_family])

    assert set(found) == {family, other_family}
    assert sorted(found[family]) == [
        "DYNAMIC DISC, NITINOL, REFLEX, 7MM",
        "STAPLE, REFLEX TETRA, LAPIDUS, 30MM",
    ]
    assert found[other_family] == ["BURR, BROPHY, 2.0X12MM"]


async def test_synonym_groups_include_canonical_and_both_edge_types():
    client = get_graph_client()
    canonical = _unique("CANONICAL")
    alias = _unique("ALIAS")
    abbreviation = _unique("ABBR")

    await client.upsert_synonym_cluster(canonical, [alias], notes="test note")
    await client.upsert_abbreviation(canonical, abbreviation)
    groups = await client.get_synonym_groups()

    group = next(candidate_group for candidate_group in groups if candidate_group[0] == canonical)
    assert alias in group
    assert abbreviation in group


# ---- Per-document prose facts (issue #28) ----


async def _differentiation_documents(client, sku: str) -> list[str]:
    return sorted(
        entry["relationship_properties"].get("document_id")
        for entry in await client.query_related_entities(sku, "DIFFERENTIATES_FROM")
    )


async def test_each_document_gets_its_own_differentiation_edge_and_replaces_only_its_own():
    client = get_graph_client()
    tray = _unique("TRAY")
    sku_a, sku_b = _unique("A"), _unique("B")
    first_document, second_document = _unique("DOC"), _unique("DOC")
    await client.upsert_tray(tray, "MIS")
    await client.upsert_part(sku_a, tray)
    await client.upsert_part(sku_b, tray)
    for document_id in (first_document, second_document):
        await client.upsert_document(document_id, "guide.pdf", doc_type=None, product_family="MIS")
        await client.replace_document_differentiations(
            document_id, [{"sku_a": sku_a, "sku_b": sku_b, "explanation": f"from {document_id}"}]
        )

    assert await _differentiation_documents(client, sku_a) == sorted([first_document, second_document])

    attached = await client.replace_document_differentiations(first_document, [])

    assert attached == 0
    [remaining] = await client.query_related_entities(sku_a, "DIFFERENTIATES_FROM")
    assert remaining["relationship_properties"] == {
        "document_id": second_document,
        "explanation": f"from {second_document}",
    }


async def _procedure_keys(client, document_id: str) -> list[str]:
    async with client._driver.session() as session:
        result = await session.run(
            "MATCH (p:Procedure)-[:SOURCED_FROM]->(:Document {id: $id}) RETURN p.key AS key",
            id=document_id,
        )
        return sorted([record["key"] async for record in result])


async def test_procedures_are_keyed_per_document_and_replaced_on_reindex():
    client = get_graph_client()
    await client.ensure_constraints()
    tray = _unique("TRAY")
    first_document, second_document = _unique("DOC"), _unique("DOC")
    await client.upsert_tray(tray, "Core Foot Plating System")
    for document_id in (first_document, second_document):
        await client.upsert_document(document_id, "guide.pdf", doc_type=None, product_family=None)
        # The same technique name in two documents is two Procedures.
        await client.replace_document_procedures(
            document_id, [{"name": "Lapidus", "trays": [tray, _unique("UNKNOWN-TRAY")]}]
        )

    requiring = sorted(
        entry["related_properties"]["key"]
        for entry in await client.query_related_entities(tray, "REQUIRES")
    )
    assert requiring == sorted(
        f"{document_id}::Lapidus" for document_id in (first_document, second_document)
    )

    await client.replace_document_procedures(first_document, [{"name": "Akin", "trays": []}])

    assert [
        entry["related_properties"]["key"]
        for entry in await client.query_related_entities(tray, "REQUIRES")
    ] == [f"{second_document}::Lapidus"]
    assert await _procedure_keys(client, first_document) == [f"{first_document}::Akin"]
    assert await _procedure_keys(client, second_document) == [f"{second_document}::Lapidus"]


async def test_unattributed_prose_facts_are_cleaned_up_and_attributed_ones_kept():
    client = get_graph_client()
    tray = _unique("TRAY")
    sku_a, sku_b = _unique("A"), _unique("B")
    document_id = _unique("DOC")
    await client.upsert_tray(tray, "MIS")
    await client.upsert_part(sku_a, tray)
    await client.upsert_part(sku_b, tray)
    await client.upsert_document(document_id, "guide.pdf", doc_type=None, product_family="MIS")
    await client.replace_document_differentiations(
        document_id, [{"sku_a": sku_a, "sku_b": sku_b, "explanation": "kept"}]
    )
    legacy_procedure = _unique("PROCEDURE")
    async with client._driver.session() as session:
        # What extraction wrote before issue #28: no document_id on the
        # edge, a Procedure keyed by its name alone.
        await session.run(
            """
            MATCH (a:Part {sku: $sku_b}), (b:Part {sku: $sku_a})
            CREATE (a)-[:DIFFERENTIATES_FROM {explanation: 'stale'}]->(b)
            CREATE (:Procedure {name: $procedure})
            """,
            sku_a=sku_a,
            sku_b=sku_b,
            procedure=legacy_procedure,
        )

    edges, procedures = await client.delete_unattributed_prose_facts()

    assert edges >= 1 and procedures >= 1
    assert await _differentiation_documents(client, sku_a) == [document_id]
    assert await client.query_related_entities(legacy_procedure) == []
    assert await client.delete_unattributed_prose_facts() == (0, 0)
