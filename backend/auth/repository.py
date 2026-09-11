"""Raw SQL against the `users` table (backend/migrations/versions/..._create_users_table.py).
No ORM — consistent with how backend/retrieval/vector_store.py talks to Postgres.
"""
import uuid
from dataclasses import dataclass
from datetime import datetime

from config.db import get_connection

_COLUMNS = "id, email, hashed_password, is_admin, is_active, created_at, last_login_at"


@dataclass
class UserRecord:
    id: uuid.UUID
    email: str
    hashed_password: str | None
    is_admin: bool
    is_active: bool
    created_at: datetime
    last_login_at: datetime | None


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
    # Signup doubles as the account's first login, so last_login_at is
    # stamped here rather than left NULL until some later /auth/login call.
    conn = await get_connection()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "INSERT INTO users (email, hashed_password, is_admin, is_active, last_login_at) "
                f"VALUES (%s, %s, %s, %s, NOW()) RETURNING {_COLUMNS}",
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


async def promote_to_admin(user_id: uuid.UUID) -> UserRecord | None:
    """Also flips is_active=True, matching signup's admin=is_active=True --
    admins bypass the is_active check anyway (auth.dependencies.require_chat_access).
    Also stamps last_login_at, since this only ever runs from the login path.
    """
    conn = await get_connection()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "UPDATE users SET is_admin = TRUE, is_active = TRUE, last_login_at = NOW() "
                f"WHERE id = %s RETURNING {_COLUMNS}",
                (user_id,),
            )
            row = await cur.fetchone()
        await conn.commit()
        return UserRecord(*row) if row else None
    finally:
        await conn.close()


async def touch_last_login(user_id: uuid.UUID) -> UserRecord | None:
    conn = await get_connection()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                f"UPDATE users SET last_login_at = NOW() WHERE id = %s RETURNING {_COLUMNS}",
                (user_id,),
            )
            row = await cur.fetchone()
        await conn.commit()
        return UserRecord(*row) if row else None
    finally:
        await conn.close()
