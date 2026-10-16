"""Raw SQL against `question_type_document_types`
(backend/migrations/versions/..._create_question_type_document_types.py):
which document types each question type retrieves from, ranked. Question
type slugs themselves are code (agents/question_types.py) and aren't
validated here -- the route does that.

Read by the deterministic workflow's hybrid_retrieve every turn (hard
doc-type filter + rerank bonus), written from the Settings page.
"""
import uuid
from dataclasses import dataclass

from config.db import get_connection


@dataclass
class RankedDocumentType:
    id: uuid.UUID
    name: str


async def list_hierarchy() -> dict[str, list[RankedDocumentType]]:
    """Every question type that has at least one document type mapped,
    each with its document types most authoritative first. A question type
    with nothing mapped is simply absent -- callers treat that the same as
    an empty list (no filter).
    """
    connection = await get_connection()
    try:
        async with connection.cursor() as cursor:
            await cursor.execute(
                "SELECT mapping.question_type, document_type.id, document_type.name "
                "FROM question_type_document_types mapping "
                "JOIN document_types document_type ON document_type.id = mapping.document_type_id "
                "ORDER BY mapping.question_type, mapping.rank, document_type.name"
            )
            rows = await cursor.fetchall()
    finally:
        await connection.close()
    hierarchy: dict[str, list[RankedDocumentType]] = {}
    for question_type, document_type_id, document_type_name in rows:
        hierarchy.setdefault(question_type, []).append(
            RankedDocumentType(document_type_id, document_type_name)
        )
    return hierarchy


async def set_document_types(question_type: str, document_type_ids: list[uuid.UUID]) -> None:
    """Replaces `question_type`'s ranked list with `document_type_ids`, in
    that order, in one transaction. An unknown document type id raises
    psycopg.errors.ForeignKeyViolation (and nothing changes); callers must
    reject duplicate ids and the read-only `unclassified` first.
    """
    connection = await get_connection()
    try:
        async with connection.transaction(), connection.cursor() as cursor:
            await cursor.execute(
                "DELETE FROM question_type_document_types WHERE question_type = %s",
                (question_type,),
            )
            for rank, document_type_id in enumerate(document_type_ids):
                await cursor.execute(
                    "INSERT INTO question_type_document_types "
                    "(question_type, document_type_id, rank) VALUES (%s, %s, %s)",
                    (question_type, document_type_id, rank),
                )
    finally:
        await connection.close()
