"""Raw SQL against `model_assignments` (one row per slot an admin has set;
an unset slot has no row and runs the default -- see slots.py), plus the
30-second in-process cache every reader goes through.

The cache is what makes a save take effect within 30 seconds without a
database read per model call: a chat turn reads it once (runtime.py), and
so do the judge and each ingested document. A save or reset clears this
process's copy right away; other processes (there's one today) catch up
when theirs expires.
"""
import time
import uuid
from dataclasses import dataclass
from datetime import datetime

from config.db import get_connection

CACHE_SECONDS = 30.0
# Its own name so a test can move time forward without patching
# time.monotonic for the event loop too.
_clock = time.monotonic

_cached_assignments: dict[str, "ModelAssignmentRecord"] | None = None
_cached_at = 0.0


@dataclass(frozen=True)
class ModelAssignmentRecord:
    slot: str
    model_id: str
    reasoning_effort: str | None
    updated_by: uuid.UUID | None
    updated_by_email: str | None
    updated_at: datetime


_SELECT = (
    "SELECT assignment.slot, assignment.model_id, assignment.reasoning_effort, "
    "assignment.updated_by, users.email, assignment.updated_at "
    "FROM model_assignments AS assignment LEFT JOIN users ON users.id = assignment.updated_by"
)


async def list_assignments() -> dict[str, ModelAssignmentRecord]:
    """Every saved slot, keyed by slot name, read straight from the
    database. Readers on a hot path use cached_assignments instead.
    """
    connection = await get_connection()
    try:
        async with connection.cursor() as cursor:
            await cursor.execute(_SELECT)
            rows = await cursor.fetchall()
            return {row[0]: ModelAssignmentRecord(*row) for row in rows}
    finally:
        await connection.close()


async def cached_assignments() -> dict[str, ModelAssignmentRecord]:
    global _cached_assignments, _cached_at
    if _cached_assignments is None or _clock() - _cached_at >= CACHE_SECONDS:
        _cached_assignments = await list_assignments()
        _cached_at = _clock()
    return _cached_assignments


def invalidate_cache() -> None:
    global _cached_assignments
    _cached_assignments = None


async def save_assignment(
    slot: str, model_id: str, reasoning_effort: str | None, updated_by: uuid.UUID
) -> None:
    connection = await get_connection()
    try:
        async with connection.cursor() as cursor:
            await cursor.execute(
                "INSERT INTO model_assignments (slot, model_id, reasoning_effort, updated_by) "
                "VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (slot) DO UPDATE SET model_id = EXCLUDED.model_id, "
                "reasoning_effort = EXCLUDED.reasoning_effort, "
                "updated_by = EXCLUDED.updated_by, updated_at = now()",
                (slot, model_id, reasoning_effort, updated_by),
            )
        await connection.commit()
    finally:
        await connection.close()
    invalidate_cache()


async def delete_assignment(slot: str) -> None:
    """Puts a slot back on the default. A no-op for a slot that's unset."""
    connection = await get_connection()
    try:
        async with connection.cursor() as cursor:
            await cursor.execute("DELETE FROM model_assignments WHERE slot = %s", (slot,))
        await connection.commit()
    finally:
        await connection.close()
    invalidate_cache()
