"""Postgres-backed LangGraph checkpointer: conversation-level memory
(per-thread state), enabling the suspend/resume human-clarification pattern
via interrupt().

AsyncPostgresSaver.from_conn_string() is itself an async context manager
(its connection pool's lifetime is tied to the `async with` block) -- this
wraps that in the same asynccontextmanager shape as
backend/retrieval/vector_store.py's get_vector_store(), so callers open it
once and reuse the same checkpointer for the process's lifetime (see
backend/api/main.py's lifespan), the same "one long-lived client per event
loop" pattern backend/retrieval/graph_client.py already established for the
Neo4j driver.
"""
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver


@asynccontextmanager
async def get_checkpointer(conn_string: str) -> AsyncIterator[AsyncPostgresSaver]:
    async with AsyncPostgresSaver.from_conn_string(conn_string) as checkpointer:
        await checkpointer.setup()
        yield checkpointer
