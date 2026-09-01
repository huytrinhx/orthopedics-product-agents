"""Raw SQL against the `documents` table
(backend/migrations/versions/..._create_documents_table.py), joined against
the `systems` and `document_types` lookup tables
(backend/migrations/versions/..._create_systems_and_document_types.py) for
display names.
"""
import uuid
from dataclasses import dataclass
from datetime import datetime

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


_SELECT = """
    SELECT d.id, d.filename, d.storage_path, d.status, d.error, d.uploaded_by,
           d.created_at, d.updated_at, d.system_id, s.name, d.document_type_id, dt.name
    FROM documents d
    LEFT JOIN systems s ON s.id = d.system_id
    LEFT JOIN document_types dt ON dt.id = d.document_type_id
"""


async def create_document(
    filename: str,
    storage_path: str,
    uploaded_by: uuid.UUID,
    system_id: uuid.UUID | None = None,
    document_type_id: uuid.UUID | None = None,
) -> DocumentRecord:
    conn = await get_connection()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "INSERT INTO documents (filename, storage_path, uploaded_by, system_id, document_type_id) "
                "VALUES (%s, %s, %s, %s, %s) RETURNING id",
                (filename, storage_path, uploaded_by, system_id, document_type_id),
            )
            (new_id,) = await cur.fetchone()
        await conn.commit()
    finally:
        await conn.close()
    doc = await get_document(new_id)
    assert doc is not None
    return doc


async def list_documents() -> list[DocumentRecord]:
    conn = await get_connection()
    try:
        async with conn.cursor() as cur:
            await cur.execute(f"{_SELECT} ORDER BY d.created_at DESC")
            rows = await cur.fetchall()
            return [DocumentRecord(*row) for row in rows]
    finally:
        await conn.close()


async def get_document(document_id: uuid.UUID) -> DocumentRecord | None:
    conn = await get_connection()
    try:
        async with conn.cursor() as cur:
            await cur.execute(f"{_SELECT} WHERE d.id = %s", (document_id,))
            row = await cur.fetchone()
            return DocumentRecord(*row) if row else None
    finally:
        await conn.close()


async def set_status(
    document_id: uuid.UUID, status: DocumentStatus, error: str | None = None
) -> None:
    conn = await get_connection()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "UPDATE documents SET status = %s, error = %s, updated_at = now() WHERE id = %s",
                (status, error, document_id),
            )
        await conn.commit()
    finally:
        await conn.close()


async def delete_document(document_id: uuid.UUID) -> DocumentRecord | None:
    doc = await get_document(document_id)
    if doc is None:
        return None
    conn = await get_connection()
    try:
        async with conn.cursor() as cur:
            await cur.execute("DELETE FROM documents WHERE id = %s", (document_id,))
        await conn.commit()
    finally:
        await conn.close()
    return doc


@dataclass
class ChunkRecord:
    chunk_index: int
    content: str
    section_title: str | None


async def list_chunks(document_id: uuid.UUID) -> list[ChunkRecord]:
    """A document's retrieval chunks in order (backend/ingestion/chunking.py
    wrote them, backend/retrieval/vector_store.py searches them) -- used to
    back the chat citation viewer (backend/api/routes/chat.py), not
    retrieval itself, so this is plain config/db.py SQL rather than
    vector_store.py's pgvector-registered connection; no embedding column
    is read here.
    """
    conn = await get_connection()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT chunk_index, content, section_title FROM chunks "
                "WHERE document_id = %s ORDER BY chunk_index",
                (document_id,),
            )
            rows = await cur.fetchall()
            return [ChunkRecord(*row) for row in rows]
    finally:
        await conn.close()


async def set_tags(
    document_id: uuid.UUID,
    system_id: uuid.UUID | None,
    document_type_id: uuid.UUID | None,
) -> DocumentRecord | None:
    conn = await get_connection()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "UPDATE documents SET system_id = %s, document_type_id = %s, updated_at = now() "
                "WHERE id = %s",
                (system_id, document_type_id, document_id),
            )
        await conn.commit()
    finally:
        await conn.close()
    return await get_document(document_id)
