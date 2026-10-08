"""Raw SQL against the `feedback` table
(backend/migrations/versions/5d95b6897886_create_feedback_table.py) -- human
4-axis scores/flag/comment on a specific chat message. Keyed by message_id
itself (LangGraph's own auto-assigned per-message uuid, see
backend/api/routes/chat.py) rather than a surrogate id, the same way
chat_threads uses thread_id as its own primary key.
"""
import uuid
from dataclasses import dataclass
from datetime import datetime

from config.db import get_connection
from feedback.models import FeedbackOut


@dataclass
class FeedbackRecord:
    message_id: str
    thread_id: str
    flagged: bool
    resolved: bool
    faithfulness: float | None
    relevance: float | None
    style: float | None
    citation: float | None
    comment: str | None
    submitted_by: uuid.UUID
    created_at: datetime
    updated_at: datetime


_COLUMNS = (
    "message_id, thread_id, flagged, resolved, faithfulness, relevance, style, citation, "
    "comment, submitted_by, created_at, updated_at"
)


def to_feedback_out(record: FeedbackRecord) -> FeedbackOut:
    return FeedbackOut(
        message_id=record.message_id,
        thread_id=record.thread_id,
        flagged=record.flagged,
        resolved=record.resolved,
        scores={
            key: value
            for key, value in {
                "faithfulness": record.faithfulness,
                "relevance": record.relevance,
                "style": record.style,
                "citation": record.citation,
            }.items()
            if value is not None
        },
        comment=record.comment,
        submitted_by=record.submitted_by,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


async def upsert_feedback(
    message_id: str,
    thread_id: str,
    flagged: bool,
    faithfulness: float | None,
    relevance: float | None,
    style: float | None,
    citation: float | None,
    comment: str | None,
    submitted_by: uuid.UUID,
) -> FeedbackRecord:
    """One row per message_id -- resubmitting the same message overwrites
    the previous feedback (ticket 11's design: a user can correct a
    misclick, no separate edit flow)."""
    connection = await get_connection()
    try:
        async with connection.cursor() as cursor:
            await cursor.execute(
                f"""
                INSERT INTO feedback
                    (message_id, thread_id, flagged, faithfulness, relevance, style, citation,
                     comment, submitted_by)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (message_id) DO UPDATE SET
                    flagged = EXCLUDED.flagged,
                    faithfulness = EXCLUDED.faithfulness,
                    relevance = EXCLUDED.relevance,
                    style = EXCLUDED.style,
                    citation = EXCLUDED.citation,
                    comment = EXCLUDED.comment,
                    submitted_by = EXCLUDED.submitted_by,
                    updated_at = now()
                RETURNING {_COLUMNS}
                """,
                (
                    message_id,
                    thread_id,
                    flagged,
                    faithfulness,
                    relevance,
                    style,
                    citation,
                    comment,
                    submitted_by,
                ),
            )
            row = await cursor.fetchone()
        await connection.commit()
    finally:
        await connection.close()
    return FeedbackRecord(*row)


async def get_feedback_for_thread(thread_id: str) -> dict[str, FeedbackRecord]:
    """All submitted feedback for one thread, keyed by message_id -- lets
    GET /chat/threads/{id} (backend/api/routes/chat.py) embed each message's
    own feedback (if any) back into the transcript on reload."""
    connection = await get_connection()
    try:
        async with connection.cursor() as cursor:
            await cursor.execute(
                f"SELECT {_COLUMNS} FROM feedback WHERE thread_id = %s",
                (thread_id,),
            )
            rows = await cursor.fetchall()
            return {row[0]: FeedbackRecord(*row) for row in rows}
    finally:
        await connection.close()


async def list_all_feedback() -> list[tuple[FeedbackRecord, str | None]]:
    """Every feedback row, flagged or not, newest first, paired with the
    submitter's email -- the Eval tab's table. No `resolved` filtering here:
    the Unresolved/All filter is applied client-side, and `resolved` only
    means anything on a flagged row (the UI shows no toggle otherwise).
    LEFT JOIN rather than JOIN so a row whose submitter was since removed
    still lists, with no email, instead of silently disappearing.
    """
    prefixed_columns = ", ".join(f"feedback.{column.strip()}" for column in _COLUMNS.split(","))
    connection = await get_connection()
    try:
        async with connection.cursor() as cursor:
            await cursor.execute(
                f"SELECT {prefixed_columns}, users.email FROM feedback "
                "LEFT JOIN users ON users.id = feedback.submitted_by "
                "ORDER BY feedback.created_at DESC"
            )
            rows = await cursor.fetchall()
            return [(FeedbackRecord(*row[:-1]), row[-1]) for row in rows]
    finally:
        await connection.close()


async def delete_feedback(message_id: str) -> bool:
    """Ticket 15 follow-up: an admin can remove a feedback item outright
    (e.g. a duplicate, a misclick, or one confirmed fixed and no longer
    worth keeping around) rather than only ever resolving it. Cascades to
    any legacy rerun chat_threads rows (migration e2f039991c90) -- see
    that migration's own comment for why. Returns whether a row actually
    existed to delete, so the route can 404 rather than silently no-op.
    """
    connection = await get_connection()
    try:
        async with connection.cursor() as cursor:
            await cursor.execute("DELETE FROM feedback WHERE message_id = %s", (message_id,))
            deleted = cursor.rowcount > 0
        await connection.commit()
    finally:
        await connection.close()
    return deleted


async def set_resolved(message_id: str, resolved: bool) -> FeedbackRecord | None:
    """Only a flagged row can be resolved -- "resolved" means "the flagged
    issue was confirmed fixed", which has no meaning for a plain score or
    comment. None if there's no flagged row with that message_id."""
    connection = await get_connection()
    try:
        async with connection.cursor() as cursor:
            await cursor.execute(
                f"UPDATE feedback SET resolved = %s, updated_at = now() "
                f"WHERE message_id = %s AND flagged RETURNING {_COLUMNS}",
                (resolved, message_id),
            )
            row = await cursor.fetchone()
        await connection.commit()
    finally:
        await connection.close()
    return FeedbackRecord(*row) if row else None
