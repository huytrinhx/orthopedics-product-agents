"""The one-off legacy System tag -> Product Family/Trays migration
(catalog/migrate_legacy_system_tags.py): the planning rules in isolation,
then a dry run and an apply against real Postgres/Neo4j, no mocking.
"""
import uuid
from datetime import UTC, datetime

from auth.repository import create_user
from catalog.migrate_legacy_system_tags import format_report, migrate, plan_migration
from catalog.repository import (
    ProductFamilyRecord,
    TrayRecord,
    list_families_with_trays,
    upsert_families_and_trays,
)
from documents.repository import DocumentRecord, create_document, get_document
from retrieval.graph_client import get_graph_client
from retrieval.vector_store import get_vector_store
from tags.repository import create_system

# text-embedding-3-small's dimension -- see ..._create_chunks_table.py.
_EMBEDDING_DIM = 1536


def _unique(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _family(name: str, tray_count: int) -> ProductFamilyRecord:
    return ProductFamilyRecord(
        id=uuid.uuid4(),
        name=name,
        display_name=name,
        trays=[
            TrayRecord(uuid.uuid4(), f"{name} tray {index}", f"{name} tray {index}")
            for index in range(tray_count)
        ],
    )


def _document(filename: str, system_name: str | None) -> DocumentRecord:
    now = datetime.now(UTC)
    return DocumentRecord(
        id=uuid.uuid4(),
        filename=filename,
        storage_path=f"/tmp/{filename}",
        status="done",
        error=None,
        uploaded_by=uuid.uuid4(),
        created_at=now,
        updated_at=now,
        system_id=uuid.uuid4() if system_name else None,
        system_name=system_name,
        document_type_id=None,
        document_type_name=None,
    )


# ---- Planning rules ----


def test_single_tray_family_mapping():
    family = _family("Core Foot Plating System", tray_count=1)
    document = _document("core.pdf", "Core Foot Plating System")

    plan = plan_migration([document], [family])

    (migration,) = plan.migrations
    assert migration.family is family
    assert migration.trays == family.trays
    assert not migration.needs_tray_review
    assert plan.documents_per_tag == {"Core Foot Plating System": 1}


def test_multi_tray_family_mapping_is_flagged_and_gets_every_tray():
    # A real legacy-map entry: the tag isn't the family name verbatim.
    family = _family("MIS - Foot Recon System", tray_count=3)
    document = _document("mis.pdf", "MIS - Foot Recon")

    plan = plan_migration([document], [family])

    (migration,) = plan.migrations
    assert migration.family is family
    assert migration.trays == family.trays
    assert migration.needs_tray_review
    report = "\n".join(format_report(plan))
    assert "[REVIEW: 3 trays]" in report
    assert "more than one tray (trim later in the tag modal): 1" in report


def test_unmapped_tag_is_left_without_a_family():
    document = _document("mystery.pdf", "Some Retired System")

    plan = plan_migration([document], [_family("Core Foot Plating System", tray_count=1)])

    assert plan.migrations == []
    assert plan.unmapped == [document]
    report = "\n".join(format_report(plan))
    assert "Some Retired System: 1 -> UNMAPPED" in report
    assert "Unmapped tags (documents left without family/trays): 1 documents" in report


def test_untagged_document_is_left_without_a_family():
    document = _document("untagged.pdf", None)

    plan = plan_migration([document], [_family("Core Foot Plating System", tray_count=1)])

    assert plan.migrations == []
    assert plan.untagged == [document]
    assert "Untagged documents (left without family/trays): 1" in format_report(plan)


def test_document_that_already_has_a_family_is_skipped():
    family = _family("Core Foot Plating System", tray_count=2)
    document = _document("core.pdf", "Core Foot Plating System")
    document.product_family_id = family.id
    document.product_family_display_name = family.display_name

    plan = plan_migration([document], [family])

    assert plan.migrations == []
    assert plan.already_migrated == [document]


# ---- Dry run and apply, against the real databases ----


async def _chunk_family(document_id: uuid.UUID) -> uuid.UUID | None:
    async with get_vector_store() as store, store._connection.cursor() as cursor:
        await cursor.execute(
            "SELECT product_family_id FROM chunks WHERE document_id = %s", (document_id,)
        )
        (product_family_id,) = await cursor.fetchone()
    return product_family_id


async def test_dry_run_writes_nothing_then_apply_writes_everything_once(capsys):
    single_name, multi_name = _unique("Single Family"), _unique("Multi Family")
    await upsert_families_and_trays(
        {
            single_name: [_unique("Single Tray")],
            multi_name: [_unique("Multi Tray A"), _unique("Multi Tray B")],
        }
    )
    families = {family.name: family for family in await list_families_with_trays()}
    single_family, multi_family = families[single_name], families[multi_name]

    user = await create_user(f"{_unique('user')}@example.com", None, False, True)

    async def _create(system_name: str | None) -> uuid.UUID:
        system_id = (await create_system(system_name)).id if system_name else None
        document = await create_document(
            filename=_unique("doc") + ".pdf",
            storage_path=f"/tmp/{_unique('storage')}.pdf",
            uploaded_by=user.id,
            system_id=system_id,
        )
        return document.id

    # Families whose name is the legacy tag verbatim pass through the map.
    single_id = await _create(single_name)
    multi_id = await _create(multi_name)
    unmapped_id = await _create(_unique("Unmapped System"))
    untagged_id = await _create(None)
    document_ids = [single_id, multi_id, unmapped_id, untagged_id]

    # The single-family document was indexed by the old pipeline: it has a
    # chunk and a graph node carrying the legacy `system` property. The
    # multi-family one was never indexed, so it has neither.
    async with get_vector_store() as store:
        await store.upsert_chunks(
            single_id,
            [{"chunk_index": 0, "content": "text", "embedding": [0.0] * _EMBEDDING_DIM}],
        )
    graph = get_graph_client()
    await graph.upsert_document(str(single_id), "doc.pdf", doc_type=None, product_family=None)
    async with graph._driver.session() as session:
        await session.run(
            "MATCH (d:Document {id: $id}) SET d.system = $system",
            id=str(single_id),
            system=single_name,
        )

    async def _documents() -> list[DocumentRecord]:
        return [await get_document(document_id) for document_id in document_ids]

    dry_run = await migrate(apply=False, documents=await _documents())

    assert {migration.document.id for migration in dry_run.migrations} == {single_id, multi_id}
    assert [document.id for document in dry_run.unmapped] == [unmapped_id]
    assert [document.id for document in dry_run.untagged] == [untagged_id]
    assert "Dry run -- nothing written" in capsys.readouterr().out
    for document in await _documents():
        assert document.product_family_id is None
        assert document.trays == []
    assert await _chunk_family(single_id) is None
    assert (await graph.get_document_properties(str(single_id)))["system"] == single_name

    await migrate(apply=True, documents=await _documents())

    single, multi, unmapped, untagged = await _documents()
    assert single.product_family_id == single_family.id
    assert [tray.id for tray in single.trays] == [tray.id for tray in single_family.trays]
    assert multi.product_family_id == multi_family.id
    assert sorted(tray.id for tray in multi.trays) == sorted(
        tray.id for tray in multi_family.trays
    )
    assert unmapped.product_family_id is None and unmapped.trays == []
    assert untagged.product_family_id is None and untagged.trays == []
    assert await _chunk_family(single_id) == single_family.id
    properties = await graph.get_document_properties(str(single_id))
    assert properties["product_family"] == single_name
    assert properties["trays"] == [tray.name for tray in single_family.trays]
    assert "system" not in properties
    # Never indexed: the migration doesn't mint a graph node for it.
    assert await graph.get_document_properties(str(multi_id)) is None

    rerun = await migrate(apply=True, documents=await _documents())

    assert rerun.migrations == []
    assert {document.id for document in rerun.already_migrated} == {single_id, multi_id}
    assert await _documents() == [single, multi, unmapped, untagged]
