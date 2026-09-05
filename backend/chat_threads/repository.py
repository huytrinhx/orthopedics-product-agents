"""Raw SQL against the `chat_threads` table
(backend/migrations/versions/..._create_chat_threads_table.py) --
sidebar-facing metadata (title, recency) for a LangGraph checkpointer
thread, keyed by the same thread_id string backend/api/routes/chat.py mints.
Separate from the checkpointer's own Postgres tables
(backend/memory/checkpointer.py), which own the actual conversation state
and are only ever read through LangGraph's own API (see get_chat_thread in
chat.py), never queried directly here.
"""
import uuid
from dataclasses import dataclass
from datetime import datetime

from config.db import get_connection


def owns_thread(user_id: str, thread_id: str) -> bool:
    """thread_id is always minted as "{user_id}:{uuid4().hex}"
    (backend/api/routes/chat.py's _new_thread_id) -- checking the prefix is
    enough to prove ownership without a separate lookup table. Shared by
    every route that accepts a client-supplied thread_id (chat.py's stream/
    resume/get_chat_thread, feedback.py's submit_feedback)."""
    return thread_id.startswith(f"{user_id}:")


@dataclass
class ChatThreadRecord:
    thread_id: str
    user_id: uuid.UUID
    title: str
    created_at: datetime
    updated_at: datetime
    # Both None for an ordinary thread. Set together only for a ticket 15
    # rerun thread -- see this table's migration (0bb3fb27c761) for why a
    # rerun is a real chat_threads row rather than a separate table.
    rerun_of_message_id: str | None
    workflow_name: str | None


_COLUMNS = "thread_id, user_id, title, created_at, updated_at, rerun_of_message_id, workflow_name"


async def create_thread(
    thread_id: str,
    user_id: uuid.UUID,
    title: str,
    *,
    rerun_of_message_id: str | None = None,
    workflow_name: str | None = None,
) -> ChatThreadRecord:
    conn = await get_connection()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "INSERT INTO chat_threads (thread_id, user_id, title, rerun_of_message_id, workflow_name) "
                "VALUES (%s, %s, %s, %s, %s) "
                f"RETURNING {_COLUMNS}",
                (thread_id, user_id, title, rerun_of_message_id, workflow_name),
            )
            row = await cur.fetchone()
        await conn.commit()
    finally:
        await conn.close()
    return ChatThreadRecord(*row)


async def touch_thread(thread_id: str) -> None:
    conn = await get_connection()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "UPDATE chat_threads SET updated_at = now() WHERE thread_id = %s",
                (thread_id,),
            )
        await conn.commit()
    finally:
        await conn.close()


async def list_threads(user_id: uuid.UUID) -> list[ChatThreadRecord]:
    """A user's own threads for the sidebar -- excludes rerun threads
    (ticket 15), which would otherwise clutter a rep/admin's own
    conversation history with threads nobody actually typed into. See
    list_reruns below for how a rerun thread is found instead."""
    conn = await get_connection()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                f"SELECT {_COLUMNS} FROM chat_threads "
                "WHERE user_id = %s AND rerun_of_message_id IS NULL "
                "ORDER BY updated_at DESC",
                (user_id,),
            )
            rows = await cur.fetchall()
            return [ChatThreadRecord(*row) for row in rows]
    finally:
        await conn.close()


async def get_thread(thread_id: str) -> ChatThreadRecord | None:
    conn = await get_connection()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                f"SELECT {_COLUMNS} FROM chat_threads WHERE thread_id = %s",
                (thread_id,),
            )
            row = await cur.fetchone()
            return ChatThreadRecord(*row) if row else None
    finally:
        await conn.close()


async def list_reruns(message_id: str) -> list[ChatThreadRecord]:
    """Every rerun attempt of one flagged feedback message (ticket 15),
    newest first -- the "history of attempts" the Eval tab nests under each
    flagged item, so re-running after a fix doesn't erase the record of
    what happened before it."""
    conn = await get_connection()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                f"SELECT {_COLUMNS} FROM chat_threads "
                "WHERE rerun_of_message_id = %s ORDER BY created_at DESC",
                (message_id,),
            )
            rows = await cur.fetchall()
            return [ChatThreadRecord(*row) for row in rows]
    finally:
        await conn.close()
