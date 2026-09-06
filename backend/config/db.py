"""Shared async Postgres connection, for modules that need plain SQL
(auth, documents, tags, feedback, eval results) rather than LangGraph's own
checkpointer/store connections (backend/memory/) or the vector/chunk store
(backend/retrieval/vector_store.py).
"""
import os

import psycopg


async def get_connection() -> psycopg.AsyncConnection:
    # prepare_threshold=0 disables psycopg3's server-side prepared
    # statements. Without this, a connection made through a PgBouncer
    # transaction-mode pooler (e.g. Supabase's pooled connection string,
    # port 6543) can have its underlying server connection swapped out
    # between queries, and a later "EXECUTE <prepared-name>" fails with
    # "prepared statement ... does not exist". LangGraph's own
    # AsyncPostgresSaver/AsyncPostgresStore already default to this same
    # setting for exactly this reason -- see backend/memory/.
    return await psycopg.AsyncConnection.connect(
        os.environ["DATABASE_URL"], prepare_threshold=0
    )
