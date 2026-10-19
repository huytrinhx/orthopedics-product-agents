"""Raw SQL against the `document_skus` table
(backend/migrations/versions/d4e8a1f37c25_create_document_skus_table.py),
plus the one read of `chunks` the scan needs.

A row references its chunk by (document_id, chunk_index) and stores only
what the chunk row doesn't have. Page number and section title are joined
in from `chunks` when read, so whatever the chunk row carries is what a
reader gets, with nothing here to keep in sync.
"""
import uuid
from dataclasses import dataclass

from config.db import get_connection


@dataclass(frozen=True)
class DocumentSkuRow:
    """One mention of one SKU in one chunk, as the scan writes it."""

    chunk_index: int
    sku: str
    mention: str
    match_kind: str
    in_document_family: bool | None
    table_heading: str | None
    row_text: str


@dataclass
class DocumentSkuRecord:
    """A stored row with its chunk's location joined in."""

    chunk_index: int
    sku: str
    mention: str
    match_kind: str
    in_document_family: bool | None
    table_heading: str | None
    row_text: str
    page_number: int | None
    section_title: str | None


async def list_chunk_texts(document_id: uuid.UUID | str) -> list[tuple[int, str]]:
    """(chunk_index, content) for every vector chunk the document has, in
    order: what the scan reads, so the index always matches the chunks that
    citations point at."""
    connection = await get_connection()
    try:
        async with connection.cursor() as cursor:
            await cursor.execute(
                "SELECT chunk_index, content FROM chunks WHERE document_id = %s ORDER BY chunk_index",
                (document_id,),
            )
            return [(chunk_index, content) for chunk_index, content in await cursor.fetchall()]
    finally:
        await connection.close()


async def replace_document_skus(document_id: uuid.UUID | str, rows: list[DocumentSkuRow]) -> None:
    """Delete-then-insert in one transaction, the same shape as
    vector_store.upsert_chunks and graph_client.replace_tray_sections: a
    re-index leaves exactly the new scan's rows, with no stale ones from a
    chunk that no longer exists."""
    connection = await get_connection()
    try:
        async with connection.cursor() as cursor:
            await cursor.execute("DELETE FROM document_skus WHERE document_id = %s", (document_id,))
            if not rows:
                await connection.commit()
                return
            await cursor.executemany(
                """
                INSERT INTO document_skus
                    (document_id, chunk_index, sku, mention, match_kind,
                     in_document_family, table_heading, row_text)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                [
                    (
                        document_id,
                        row.chunk_index,
                        row.sku,
                        row.mention,
                        row.match_kind,
                        row.in_document_family,
                        row.table_heading,
                        row.row_text,
                    )
                    for row in rows
                ],
            )
        await connection.commit()
    finally:
        await connection.close()


async def list_document_skus(document_id: uuid.UUID | str) -> list[DocumentSkuRecord]:
    """A document's rows in chunk order, each with its chunk's page number
    and section title. LEFT JOIN, so a row still comes back if its chunk is
    somehow gone (its location fields are then None)."""
    connection = await get_connection()
    try:
        async with connection.cursor() as cursor:
            await cursor.execute(
                """
                SELECT ds.chunk_index, ds.sku, ds.mention, ds.match_kind,
                       ds.in_document_family, ds.table_heading, ds.row_text,
                       c.page_number, c.section_title
                FROM document_skus ds
                LEFT JOIN chunks c
                       ON c.document_id = ds.document_id AND c.chunk_index = ds.chunk_index
                WHERE ds.document_id = %s
                ORDER BY ds.chunk_index, ds.sku, ds.match_kind
                """,
                (document_id,),
            )
            return [DocumentSkuRecord(*row) for row in await cursor.fetchall()]
    finally:
        await connection.close()
