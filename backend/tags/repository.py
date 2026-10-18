"""Raw SQL against the `document_types` lookup table
(backend/migrations/versions/..._create_systems_and_document_types.py --
the legacy `systems` table it also created was dropped in issue #20).
"""
import uuid
from dataclasses import dataclass
from datetime import datetime

from config.db import get_connection


@dataclass
class TagRecord:
    id: uuid.UUID
    name: str
    created_at: datetime


async def create_document_type(name: str) -> TagRecord:
    connection = await get_connection()
    try:
        async with connection.cursor() as cursor:
            await cursor.execute(
                "INSERT INTO document_types (name) VALUES (%s) RETURNING id, name, created_at",
                (name,),
            )
            row = await cursor.fetchone()
        await connection.commit()
        return TagRecord(*row)
    finally:
        await connection.close()


async def list_document_types() -> list[TagRecord]:
    """Every document type ever created, in use or not -- an admin managing
    the list (creating ahead of the first document that'll use it, or
    cleaning up an unused one) needs to see it regardless of whether any
    document currently references it.
    """
    connection = await get_connection()
    try:
        async with connection.cursor() as cursor:
            await cursor.execute("SELECT id, name, created_at FROM document_types ORDER BY name")
            rows = await cursor.fetchall()
            return [TagRecord(*row) for row in rows]
    finally:
        await connection.close()


async def delete_document_type(tag_id: uuid.UUID) -> bool:
    """Returns whether a row was actually deleted (False means no such
    document type). Deleting one still assigned to a document raises
    psycopg.errors.ForeignKeyViolation -- the FK columns added in
    ..._create_systems_and_document_types.py have no ON DELETE clause, so
    Postgres itself blocks it; the route layer turns that into a 409 rather
    than silently orphaning documents or cascading the delete into them.
    """
    connection = await get_connection()
    try:
        async with connection.cursor() as cursor:
            await cursor.execute("DELETE FROM document_types WHERE id = %s", (tag_id,))
            deleted = cursor.rowcount > 0
        await connection.commit()
        return deleted
    finally:
        await connection.close()


async def rename_document_type(tag_id: uuid.UUID, name: str) -> TagRecord | None:
    """Returns the renamed document type, or None if there's no such id.
    Renaming onto a name another document type already has (case-
    insensitively) raises psycopg.errors.UniqueViolation off the same
    lower(name) index that guards create -- the route turns it into a 409.
    Documents and chunks reference the type by id, so nothing else moves.
    """
    connection = await get_connection()
    try:
        async with connection.cursor() as cursor:
            await cursor.execute(
                "UPDATE document_types SET name = %s WHERE id = %s RETURNING id, name, created_at",
                (name, tag_id),
            )
            row = await cursor.fetchone()
        await connection.commit()
        return TagRecord(*row) if row else None
    finally:
        await connection.close()


class TagNotFound(Exception):
    """Raised by merge_document_type when either side of the merge doesn't
    exist; carries the missing id so the route can say which in its 404."""

    def __init__(self, tag_id: uuid.UUID):
        super().__init__(f"No such tag: {tag_id}")
        self.tag_id = tag_id


async def merge_document_type(stray_id: uuid.UUID, surviving_id: uuid.UUID) -> TagRecord:
    """Folds a stray document type into a canonical one: every document and
    every chunk on `stray_id` is re-pointed at `surviving_id`, then the
    stray row is deleted -- one transaction, so a failure partway leaves
    nothing half-moved. Chunks are updated directly rather than re-ingested:
    their document_type_id is a denormalized copy of the document's (see
    ..._create_chunks_table.py), so re-pointing both keeps retrieval's
    doc-type filter in step without re-embedding anything. The stray's
    question-type hierarchy rows are merged into the survivor's the same
    way (see below).

    Returns the surviving document type. Raises TagNotFound if either id
    doesn't exist. Callers must reject stray_id == surviving_id first.
    """
    connection = await get_connection()
    try:
        async with connection.transaction(), connection.cursor() as cursor:
            # Row locks so a concurrent rename/delete/merge of either
            # side can't interleave with the re-pointing below.
            await cursor.execute(
                "SELECT id, name, created_at FROM document_types WHERE id = ANY(%s) FOR UPDATE",
                ([stray_id, surviving_id],),
            )
            found = {row[0]: TagRecord(*row) for row in await cursor.fetchall()}
            for tag_id in (stray_id, surviving_id):
                if tag_id not in found:
                    raise TagNotFound(tag_id)
            await cursor.execute(
                "UPDATE documents SET document_type_id = %s, updated_at = now() "
                "WHERE document_type_id = %s",
                (surviving_id, stray_id),
            )
            await cursor.execute(
                "UPDATE chunks SET document_type_id = %s WHERE document_type_id = %s",
                (surviving_id, stray_id),
            )
            # Question Types -> Doc Types hierarchy rows (issue #17,
            # backend/doc_type_hierarchy/) move with the merge too, so a
            # question type that retrieved from the stray keeps retrieving
            # from the survivor. Where a question type already lists both,
            # the survivor keeps the better (lower) of the two ranks and the
            # stray's row goes; otherwise the stray's row is re-pointed.
            await cursor.execute(
                "UPDATE question_type_document_types AS surviving "
                "SET rank = LEAST(surviving.rank, stray.rank) "
                "FROM question_type_document_types AS stray "
                "WHERE surviving.document_type_id = %s AND stray.document_type_id = %s "
                "AND surviving.question_type = stray.question_type",
                (surviving_id, stray_id),
            )
            await cursor.execute(
                "DELETE FROM question_type_document_types AS stray "
                "WHERE stray.document_type_id = %s AND EXISTS ("
                "SELECT 1 FROM question_type_document_types AS surviving "
                "WHERE surviving.document_type_id = %s "
                "AND surviving.question_type = stray.question_type)",
                (stray_id, surviving_id),
            )
            await cursor.execute(
                "UPDATE question_type_document_types SET document_type_id = %s "
                "WHERE document_type_id = %s",
                (surviving_id, stray_id),
            )
            await cursor.execute("DELETE FROM document_types WHERE id = %s", (stray_id,))
        return found[surviving_id]
    finally:
        await connection.close()
