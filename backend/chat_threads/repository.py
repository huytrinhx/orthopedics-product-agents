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


@dataclass
class ChatThreadRecord:
    thread_id: str
    user_id: uuid.UUID
    title: str
    created_at: datetime
    updated_at: datetime


async def create_thread(thread_id: str, user_id: uuid.UUID, title: str) -> ChatThreadRecord:
    conn = await get_connection()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "INSERT INTO chat_threads (thread_id, user_id, title) "
                "VALUES (%s, %s, %s) "
                "RETURNING thread_id, user_id, title, created_at, updated_at",
                (thread_id, user_id, title),
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
    conn = await get_connection()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT thread_id, user_id, title, created_at, updated_at "
                "FROM chat_threads WHERE user_id = %s ORDER BY updated_at DESC",
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
                "SELECT thread_id, user_id, title, created_at, updated_at "
                "FROM chat_threads WHERE thread_id = %s",
                (thread_id,),
            )
            row = await cur.fetchone()
            return ChatThreadRecord(*row) if row else None
    finally:
        await conn.close()
