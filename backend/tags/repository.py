"""Raw SQL against the `systems` and `document_types` lookup tables
(backend/migrations/versions/..._create_systems_and_document_types.py).
Both tables are the same shape (id, name, created_at), so the CRUD below is
shared -- callers only ever pass one of the two whitelisted table names,
never a caller-supplied string, so the f-string interpolation isn't a SQL
injection surface.
"""
import uuid
from dataclasses import dataclass
from datetime import datetime

from config.db import get_connection

_TABLES = {"systems", "document_types"}


@dataclass
class TagRecord:
    id: uuid.UUID
    name: str
    created_at: datetime


async def _create_tag(table: str, name: str) -> TagRecord:
    assert table in _TABLES
    conn = await get_connection()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                f"INSERT INTO {table} (name) VALUES (%s) RETURNING id, name, created_at",
                (name,),
            )
            row = await cur.fetchone()
        await conn.commit()
        return TagRecord(*row)
    finally:
        await conn.close()


async def _list_tags(table: str) -> list[TagRecord]:
    """Every tag ever created, in use or not -- an admin managing the list
    (creating ahead of the first document that'll use it, or cleaning up an
    unused one) needs to see it regardless of whether any document currently
    references it.
    """
    assert table in _TABLES
    conn = await get_connection()
    try:
        async with conn.cursor() as cur:
            await cur.execute(f"SELECT id, name, created_at FROM {table} ORDER BY name")
            rows = await cur.fetchall()
            return [TagRecord(*row) for row in rows]
    finally:
        await conn.close()


async def _delete_tag(table: str, tag_id: uuid.UUID) -> bool:
    """Returns whether a row was actually deleted (False means no such tag).
    Deleting a tag still assigned to a document raises psycopg.errors.
    ForeignKeyViolation -- the FK columns added in
    ..._create_systems_and_document_types.py have no ON DELETE clause, so
    Postgres itself blocks it; the route layer turns that into a 409 rather
    than silently orphaning documents or cascading the delete into them.
    """
    assert table in _TABLES
    conn = await get_connection()
    try:
        async with conn.cursor() as cur:
            await cur.execute(f"DELETE FROM {table} WHERE id = %s", (tag_id,))
            deleted = cur.rowcount > 0
        await conn.commit()
        return deleted
    finally:
        await conn.close()


async def create_system(name: str) -> TagRecord:
    return await _create_tag("systems", name)


async def list_systems() -> list[TagRecord]:
    return await _list_tags("systems")


async def delete_system(tag_id: uuid.UUID) -> bool:
    return await _delete_tag("systems", tag_id)


async def create_document_type(name: str) -> TagRecord:
    return await _create_tag("document_types", name)


async def list_document_types() -> list[TagRecord]:
    return await _list_tags("document_types")


async def delete_document_type(tag_id: uuid.UUID) -> bool:
    return await _delete_tag("document_types", tag_id)
