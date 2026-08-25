"""Postgres-backed LangGraph checkpointer: conversation-level memory
(per-thread state), enabling the suspend/resume human-clarification pattern
via interrupt().
"""
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver


async def get_checkpointer(conn_string: str) -> AsyncPostgresSaver:
    raise NotImplementedError
