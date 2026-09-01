"""Shared async Postgres connection, for modules that need plain SQL
(auth, documents, tags, feedback, eval results) rather than LangGraph's own
checkpointer/store connections (backend/memory/) or the vector/chunk store
(backend/retrieval/vector_store.py).
"""
import os

import psycopg


async def get_connection() -> psycopg.AsyncConnection:
    return await psycopg.AsyncConnection.connect(os.environ["DATABASE_URL"])
