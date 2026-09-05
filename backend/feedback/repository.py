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
            k: v
            for k, v in {
                "faithfulness": record.faithfulness,
                "relevance": record.relevance,
                "style": record.style,
                "citation": record.citation,
            }.items()
            if v is not None
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
    conn = await get_connection()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
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
            row = await cur.fetchone()
        await conn.commit()
    finally:
        await conn.close()
    return FeedbackRecord(*row)


async def get_feedback_for_thread(thread_id: str) -> dict[str, FeedbackRecord]:
    """All submitted feedback for one thread, keyed by message_id -- lets
    GET /chat/threads/{id} (backend/api/routes/chat.py) embed each message's
    own feedback (if any) back into the transcript on reload."""
    conn = await get_connection()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                f"SELECT {_COLUMNS} FROM feedback WHERE thread_id = %s",
                (thread_id,),
            )
            rows = await cur.fetchall()
            return {row[0]: FeedbackRecord(*row) for row in rows}
    finally:
        await conn.close()


async def list_flagged_feedback() -> list[FeedbackRecord]:
    """Every flagged feedback item, newest first -- the Eval tab's (ticket
    15) main list. Unfiltered by `resolved` on purpose: the admin toggles
    that in the UI, not by hiding rows here, so a resolved item stays
    visible (and its rerun history intact) rather than disappearing.
    Uses the partial `feedback_flagged_idx` index (migration
    5d95b6897886), created ahead of this exact need.
    """
    conn = await get_connection()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                f"SELECT {_COLUMNS} FROM feedback WHERE flagged ORDER BY created_at DESC"
            )
            rows = await cur.fetchall()
            return [FeedbackRecord(*row) for row in rows]
    finally:
        await conn.close()


async def set_resolved(message_id: str, resolved: bool) -> FeedbackRecord | None:
    conn = await get_connection()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                f"UPDATE feedback SET resolved = %s, updated_at = now() "
                f"WHERE message_id = %s RETURNING {_COLUMNS}",
                (resolved, message_id),
            )
            row = await cur.fetchone()
        await conn.commit()
    finally:
        await conn.close()
    return FeedbackRecord(*row) if row else None
