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
    # Both None for an ordinary thread. Set together only on legacy ticket
    # 15 rerun threads (migration 0bb3fb27c761) -- nothing writes them
    # anymore since the Eval tab's inline rerun was removed 2026-10-07 (a
    # rerun is now just an ordinary new chat turn), but existing rows keep
    # them, so they're still read.
    rerun_of_message_id: str | None
    workflow_name: str | None


_COLUMNS = "thread_id, user_id, title, created_at, updated_at, rerun_of_message_id, workflow_name"


async def create_thread(
    thread_id: str,
    user_id: uuid.UUID,
    title: str,
) -> ChatThreadRecord:
    connection = await get_connection()
    try:
        async with connection.cursor() as cursor:
            await cursor.execute(
                "INSERT INTO chat_threads (thread_id, user_id, title) "
                "VALUES (%s, %s, %s) "
                f"RETURNING {_COLUMNS}",
                (thread_id, user_id, title),
            )
            row = await cursor.fetchone()
        await connection.commit()
    finally:
        await connection.close()
    return ChatThreadRecord(*row)


async def touch_thread(thread_id: str) -> None:
    connection = await get_connection()
    try:
        async with connection.cursor() as cursor:
            await cursor.execute(
                "UPDATE chat_threads SET updated_at = now() WHERE thread_id = %s",
                (thread_id,),
            )
        await connection.commit()
    finally:
        await connection.close()


async def list_threads(user_id: uuid.UUID) -> list[ChatThreadRecord]:
    """A user's own threads for the sidebar -- excludes legacy ticket 15
    rerun threads (see ChatThreadRecord), which were created by the old
    inline Eval-tab rerun rather than typed by anyone, and no longer have
    any other UI that lists them."""
    connection = await get_connection()
    try:
        async with connection.cursor() as cursor:
            await cursor.execute(
                f"SELECT {_COLUMNS} FROM chat_threads "
                "WHERE user_id = %s AND rerun_of_message_id IS NULL "
                "ORDER BY updated_at DESC",
                (user_id,),
            )
            rows = await cursor.fetchall()
            return [ChatThreadRecord(*row) for row in rows]
    finally:
        await connection.close()


async def get_thread(thread_id: str) -> ChatThreadRecord | None:
    connection = await get_connection()
    try:
        async with connection.cursor() as cursor:
            await cursor.execute(
                f"SELECT {_COLUMNS} FROM chat_threads WHERE thread_id = %s",
                (thread_id,),
            )
            row = await cursor.fetchone()
            return ChatThreadRecord(*row) if row else None
    finally:
        await connection.close()
