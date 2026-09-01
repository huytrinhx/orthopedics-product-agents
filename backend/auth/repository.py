"""Raw SQL against the `users` table (backend/migrations/versions/..._create_users_table.py).
No ORM — consistent with how backend/retrieval/vector_store.py talks to Postgres.
"""
import uuid
from dataclasses import dataclass
from datetime import datetime

from config.db import get_connection


@dataclass
class UserRecord:
    id: uuid.UUID
    email: str
    hashed_password: str | None
    is_admin: bool
    created_at: datetime


async def get_user_by_email(email: str) -> UserRecord | None:
    conn = await get_connection()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT id, email, hashed_password, is_admin, created_at "
                "FROM users WHERE email = %s",
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
                "SELECT id, email, hashed_password, is_admin, created_at "
                "FROM users WHERE id = %s",
                (user_id,),
            )
            row = await cur.fetchone()
            return UserRecord(*row) if row else None
    finally:
        await conn.close()


async def create_user(
    email: str, hashed_password: str | None, is_admin: bool
) -> UserRecord:
    conn = await get_connection()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "INSERT INTO users (email, hashed_password, is_admin) "
                "VALUES (%s, %s, %s) "
                "RETURNING id, email, hashed_password, is_admin, created_at",
                (email, hashed_password, is_admin),
            )
            row = await cur.fetchone()
        await conn.commit()
        return UserRecord(*row)
    finally:
        await conn.close()
