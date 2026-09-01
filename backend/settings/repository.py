"""Raw SQL against the single-row `app_settings` table
(backend/migrations/versions/..._create_app_settings_table.py). The
migration seeds the one row (id=1) that will ever exist, so `get_settings`
can assume it's there -- this module only ever UPDATEs it, never INSERTs.
"""
import uuid
from dataclasses import dataclass
from datetime import datetime

from config.db import get_connection


@dataclass
class AppSettingsRecord:
    default_workflow: str
    updated_at: datetime
    updated_by: uuid.UUID | None


async def get_settings() -> AppSettingsRecord:
    conn = await get_connection()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT default_workflow, updated_at, updated_by FROM app_settings WHERE id = 1"
            )
            row = await cur.fetchone()
            return AppSettingsRecord(*row)
    finally:
        await conn.close()


async def set_default_workflow(workflow_name: str, updated_by: uuid.UUID) -> AppSettingsRecord:
    conn = await get_connection()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "UPDATE app_settings SET default_workflow = %s, updated_at = now(), updated_by = %s "
                "WHERE id = 1 "
                "RETURNING default_workflow, updated_at, updated_by",
                (workflow_name, updated_by),
            )
            row = await cur.fetchone()
        await conn.commit()
        return AppSettingsRecord(*row)
    finally:
        await conn.close()
