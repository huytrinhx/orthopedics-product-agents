"""Raw SQL against the `users` table (backend/migrations/versions/..._create_users_table.py).
No ORM — consistent with how backend/retrieval/vector_store.py talks to Postgres.
"""
import uuid
from dataclasses import dataclass
from datetime import datetime

from config.db import get_connection

_COLUMNS = "id, email, hashed_password, is_admin, is_active, created_at"


@dataclass
class UserRecord:
    id: uuid.UUID
    email: str
    hashed_password: str | None
    is_admin: bool
    is_active: bool
    created_at: datetime


async def get_user_by_email(email: str) -> UserRecord | None:
    conn = await get_connection()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                f"SELECT {_COLUMNS} FROM users WHERE email = %s",
                (email,),
            )
            row = await cur.fetchone()
            return UserRecord(*row) if row else None
    finally:
        await conn.close()


async def get_user_by_id(user_id: uuid.UUID) -> UserRecord | None:
    conn = await get_connection()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                f"SELECT {_COLUMNS} FROM users WHERE id = %s",
                (user_id,),
            )
            row = await cur.fetchone()
            return UserRecord(*row) if row else None
    finally:
        await conn.close()


async def create_user(
    email: str, hashed_password: str | None, is_admin: bool, is_active: bool
) -> UserRecord:
    conn = await get_connection()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "INSERT INTO users (email, hashed_password, is_admin, is_active) "
                f"VALUES (%s, %s, %s, %s) RETURNING {_COLUMNS}",
                (email, hashed_password, is_admin, is_active),
            )
            row = await cur.fetchone()
        await conn.commit()
        return UserRecord(*row)
    finally:
        await conn.close()


async def list_users() -> list[UserRecord]:
    conn = await get_connection()
    try:
        async with conn.cursor() as cur:
            await cur.execute(f"SELECT {_COLUMNS} FROM users ORDER BY created_at")
            rows = await cur.fetchall()
            return [UserRecord(*row) for row in rows]
    finally:
        await conn.close()


async def set_user_active(user_id: uuid.UUID, is_active: bool) -> UserRecord | None:
    conn = await get_connection()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                f"UPDATE users SET is_active = %s WHERE id = %s RETURNING {_COLUMNS}",
                (is_active, user_id),
            )
            row = await cur.fetchone()
        await conn.commit()
        return UserRecord(*row) if row else None
    finally:
        await conn.close()
