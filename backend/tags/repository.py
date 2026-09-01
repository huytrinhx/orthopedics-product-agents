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
# Which `documents` column references each lookup table -- used to scope the
# picker down to tags actually in use (see _list_tags).
_FK_COLUMNS = {"systems": "system_id", "document_types": "document_type_id"}


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
    """Tags currently attached to at least one document -- the picker
    reflects what's actually in use, not everything ever created, so a tag
    stops showing up once its last document is deleted or re-tagged away
    from it (create still leaves the row in place; nothing lists it again
    until a document references it).
    """
    assert table in _TABLES
    fk_column = _FK_COLUMNS[table]
    conn = await get_connection()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                f"SELECT t.id, t.name, t.created_at FROM {table} t "
                f"WHERE EXISTS (SELECT 1 FROM documents d WHERE d.{fk_column} = t.id) "
                "ORDER BY t.name"
            )
            rows = await cur.fetchall()
            return [TagRecord(*row) for row in rows]
    finally:
        await conn.close()


async def create_system(name: str) -> TagRecord:
    return await _create_tag("systems", name)


async def list_systems() -> list[TagRecord]:
    return await _list_tags("systems")


async def create_document_type(name: str) -> TagRecord:
    return await _create_tag("document_types", name)


async def list_document_types() -> list[TagRecord]:
    return await _list_tags("document_types")
