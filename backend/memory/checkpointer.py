"""Postgres-backed LangGraph checkpointer: conversation-level memory
(per-thread state), enabling the suspend/resume human-clarification pattern
via interrupt().

Backed by a psycopg AsyncConnectionPool rather than
AsyncPostgresSaver.from_conn_string(): that helper opens a single
AsyncConnection and holds it for as long as the `async with` block lives,
with no reconnect. Here that block is the whole process lifetime (see
backend/api/main.py's lifespan), so once the server side dropped that one
connection (idle timeout, Supabase pooler restart, network blip) every
later checkpointer call failed with "the connection is closed" until the
container restarted. The pool's `check` validates a connection each time
one is borrowed and transparently replaces a dead one.

Still wrapped in the same asynccontextmanager shape as
backend/retrieval/vector_store.py's get_vector_store(), so callers open it
once and reuse the same checkpointer for the process's lifetime, the same
"one long-lived client per event loop" pattern
backend/retrieval/graph_client.py already established for the Neo4j driver.
"""
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

# Tags the pool's connections in pg_stat_activity, so they're identifiable
# when debugging connection usage (and so tests can target exactly them).
APPLICATION_NAME = "orthomate-checkpointer"


@asynccontextmanager
async def get_checkpointer(connection_string: str) -> AsyncIterator[AsyncPostgresSaver]:
    async with AsyncConnectionPool(
        connection_string,
        min_size=1,
        max_size=5,
        # Same connection settings from_conn_string() used: autocommit and
        # dict rows are what AsyncPostgresSaver expects, and
        # prepare_threshold=0 matches config/db.py's get_connection (no
        # server-side prepared statements, for Supabase's pooler).
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
            "application_name": APPLICATION_NAME,
        },
        check=AsyncConnectionPool.check_connection,
        open=False,
    ) as pool:
        checkpointer = AsyncPostgresSaver(conn=pool)
        await checkpointer.setup()
        yield checkpointer
