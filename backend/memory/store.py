"""Postgres-backed LangGraph Store: user-level memory shared across
conversation threads (preferences, prior context).
"""
from langgraph.store.postgres.aio import AsyncPostgresStore


async def get_store(conn_string: str) -> AsyncPostgresStore:
    raise NotImplementedError
