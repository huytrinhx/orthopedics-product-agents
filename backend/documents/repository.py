"""Raw SQL against the `documents` table
(backend/migrations/versions/..._create_documents_table.py), joined against
the `systems` and `document_types` lookup tables
(backend/migrations/versions/..._create_systems_and_document_types.py) and
the catalog's `product_families`/`trays` mirror tables
(backend/migrations/versions/..._tag_documents_with_family_and_trays.py)
for display names.

A document has at most one Product Family and any number of Trays, all of
which must belong to that family -- enforced here (TraysOutsideFamily), in
the same transaction that writes them.
"""
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from catalog.repository import TrayRecord
from config.db import get_connection
from documents.models import DocumentStatus


@dataclass
class DocumentRecord:
    id: uuid.UUID
    filename: str
    storage_path: str
    status: DocumentStatus
    error: str | None
    uploaded_by: uuid.UUID
    created_at: datetime
    updated_at: datetime
    system_id: uuid.UUID | None
    system_name: str | None
    document_type_id: uuid.UUID | None
    document_type_name: str | None
    product_family_id: uuid.UUID | None = None
    # The stable catalog name (what the graph's ProductFamily node is keyed
    # on) and the admin-editable label shown in the UI, respectively.
    product_family_name: str | None = None
    product_family_display_name: str | None = None
    trays: list[TrayRecord] = field(default_factory=list)


class TraysOutsideFamily(Exception):
    """A tray set containing trays that don't belong to the document's
    family (or any trays at all on a document with no family). Carries the
    offending tray ids so the route can name them in its 400."""

    def __init__(self, tray_ids: list[uuid.UUID]):
        super().__init__(f"Trays not in the document's product family: {tray_ids}")
        self.tray_ids = tray_ids


_SELECT = """
    SELECT d.id, d.filename, d.storage_path, d.status, d.error, d.uploaded_by,
           d.created_at, d.updated_at, d.system_id, s.name, d.document_type_id, dt.name,
           d.product_family_id, pf.name, pf.display_name
    FROM documents d
    LEFT JOIN systems s ON s.id = d.system_id
    LEFT JOIN document_types dt ON dt.id = d.document_type_id
    LEFT JOIN product_families pf ON pf.id = d.product_family_id
"""


async def _attach_trays(cursor, documents: list[DocumentRecord]) -> list[DocumentRecord]:
    if not documents:
        return documents
    await cursor.execute(
        "SELECT document_trays.document_id, trays.id, trays.name, trays.display_name "
        "FROM document_trays JOIN trays ON trays.id = document_trays.tray_id "
        "WHERE document_trays.document_id = ANY(%s) ORDER BY trays.display_name",
        ([document.id for document in documents],),
    )
    trays_by_document: dict[uuid.UUID, list[TrayRecord]] = defaultdict(list)
    for document_id, tray_id, name, display_name in await cursor.fetchall():
        trays_by_document[document_id].append(TrayRecord(tray_id, name, display_name))
    for document in documents:
        document.trays = trays_by_document[document.id]
    return documents


async def _write_family_and_trays(
    cursor,
    document_id: uuid.UUID,
    product_family_id: uuid.UUID | None,
    tray_ids: list[uuid.UUID],
) -> None:
    """Validates and writes a document's family + tray set, and copies the
    family onto its chunks (a denormalized filter column, like
    chunks.document_type_id -- see merge_document_type in tags.repository).
    Runs on the caller's cursor so it shares the caller's transaction.
    Raises TraysOutsideFamily before writing anything if a tray doesn't
    belong to `product_family_id`; an unknown family id raises
    psycopg.errors.ForeignKeyViolation off the FK.
    """
    unique_tray_ids = list(dict.fromkeys(tray_ids))
    if unique_tray_ids:
        await cursor.execute(
            "SELECT id FROM trays WHERE id = ANY(%s) AND product_family_id IS NOT DISTINCT FROM %s",
            (unique_tray_ids, product_family_id),
        )
        in_family = {row[0] for row in await cursor.fetchall()}
        outside = [tray_id for tray_id in unique_tray_ids if tray_id not in in_family]
        if outside:
            raise TraysOutsideFamily(outside)

    await cursor.execute(
        "UPDATE documents SET product_family_id = %s, updated_at = now() WHERE id = %s",
        (product_family_id, document_id),
    )
    await cursor.execute("DELETE FROM document_trays WHERE document_id = %s", (document_id,))
    for tray_id in unique_tray_ids:
        await cursor.execute(
            "INSERT INTO document_trays (document_id, tray_id) VALUES (%s, %s)",
            (document_id, tray_id),
        )
    await cursor.execute(
        "UPDATE chunks SET product_family_id = %s WHERE document_id = %s",
        (product_family_id, document_id),
    )


async def create_document(
    filename: str,
    storage_path: str,
    uploaded_by: uuid.UUID,
    system_id: uuid.UUID | None = None,
    document_type_id: uuid.UUID | None = None,
) -> DocumentRecord:
    connection = await get_connection()
    try:
        async with connection.cursor() as cursor:
            await cursor.execute(
                "INSERT INTO documents (filename, storage_path, uploaded_by, system_id, document_type_id) "
                "VALUES (%s, %s, %s, %s, %s) RETURNING id",
                (filename, storage_path, uploaded_by, system_id, document_type_id),
            )
            (new_id,) = await cursor.fetchone()
        await connection.commit()
    finally:
        await connection.close()
    document = await get_document(new_id)
    assert document is not None
    return document


async def list_documents() -> list[DocumentRecord]:
    connection = await get_connection()
    try:
        async with connection.cursor() as cursor:
            await cursor.execute(f"{_SELECT} ORDER BY d.created_at DESC")
            rows = await cursor.fetchall()
            return await _attach_trays(cursor, [DocumentRecord(*row) for row in rows])
    finally:
        await connection.close()


async def get_document(document_id: uuid.UUID) -> DocumentRecord | None:
    connection = await get_connection()
    try:
        async with connection.cursor() as cursor:
            await cursor.execute(f"{_SELECT} WHERE d.id = %s", (document_id,))
            row = await cursor.fetchone()
            if row is None:
                return None
            (document,) = await _attach_trays(cursor, [DocumentRecord(*row)])
            return document
    finally:
        await connection.close()


async def replace_file(
    document_id: uuid.UUID, filename: str, storage_path: str
) -> DocumentRecord | None:
    """Points an existing document row at a newly-uploaded file (Re-upload
    button) -- the id, tags, and history stay put; only what's on disk (and
    the display filename) changes. Callers re-run process_document against
    the new storage_path the same way index/set_tags do, so chunks/graph
    entities keyed by this document_id get overwritten (upsert_document,
    upsert_chunks) rather than duplicated.
    """
    if await get_document(document_id) is None:
        return None
    connection = await get_connection()
    try:
        async with connection.cursor() as cursor:
            await cursor.execute(
                "UPDATE documents SET filename = %s, storage_path = %s, updated_at = now() "
                "WHERE id = %s",
                (filename, storage_path, document_id),
            )
        await connection.commit()
    finally:
        await connection.close()
    return await get_document(document_id)


async def set_status(
    document_id: uuid.UUID, status: DocumentStatus, error: str | None = None
) -> None:
    connection = await get_connection()
    try:
        async with connection.cursor() as cursor:
            await cursor.execute(
                "UPDATE documents SET status = %s, error = %s, updated_at = now() WHERE id = %s",
                (status, error, document_id),
            )
        await connection.commit()
    finally:
        await connection.close()


async def delete_document(document_id: uuid.UUID) -> DocumentRecord | None:
    document = await get_document(document_id)
    if document is None:
        return None
    connection = await get_connection()
    try:
        async with connection.cursor() as cursor:
            await cursor.execute("DELETE FROM documents WHERE id = %s", (document_id,))
        await connection.commit()
    finally:
        await connection.close()
    return document


@dataclass
class ChunkRecord:
    chunk_index: int
    content: str
    section_title: str | None
    page_number: int | None


async def list_chunks(document_id: uuid.UUID) -> list[ChunkRecord]:
    """A document's retrieval chunks in order (backend/ingestion/chunking.py
    wrote them, backend/retrieval/vector_store.py searches them) -- used to
    back the chat citation viewer (backend/api/routes/chat.py), not
    retrieval itself, so this is plain config/db.py SQL rather than
    vector_store.py's pgvector-registered connection; no embedding column
    is read here.
    """
    connection = await get_connection()
    try:
        async with connection.cursor() as cursor:
            await cursor.execute(
                "SELECT chunk_index, content, section_title, page_number FROM chunks "
                "WHERE document_id = %s ORDER BY chunk_index",
                (document_id,),
            )
            rows = await cursor.fetchall()
            return [ChunkRecord(*row) for row in rows]
    finally:
        await connection.close()


async def set_tags(
    document_id: uuid.UUID,
    system_id: uuid.UUID | None,
    document_type_id: uuid.UUID | None,
    product_family_id: uuid.UUID | None,
    tray_ids: list[uuid.UUID],
) -> DocumentRecord | None:
    """Replaces every tag on the document in one transaction -- callers
    pass the current value for anything they aren't changing. Raises
    TraysOutsideFamily (nothing written) if a tray isn't in the family."""
    connection = await get_connection()
    try:
        async with connection.transaction(), connection.cursor() as cursor:
            await cursor.execute(
                "UPDATE documents SET system_id = %s, document_type_id = %s, updated_at = now() "
                "WHERE id = %s",
                (system_id, document_type_id, document_id),
            )
            await _write_family_and_trays(cursor, document_id, product_family_id, tray_ids)
    finally:
        await connection.close()
    return await get_document(document_id)


async def set_family_and_trays(
    document_id: uuid.UUID, product_family_id: uuid.UUID | None, tray_ids: list[uuid.UUID]
) -> None:
    """Family + trays only, leaving the other tags alone -- the legacy
    system-tag migration (catalog/migrate_legacy_system_tags.py) writes
    through this. Same validation as set_tags."""
    connection = await get_connection()
    try:
        async with connection.transaction(), connection.cursor() as cursor:
            await _write_family_and_trays(cursor, document_id, product_family_id, tray_ids)
    finally:
        await connection.close()
