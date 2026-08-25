"""LangGraph tool wrapping Postgres/pgvector hybrid retrieval.

See backend/retrieval/vector_store.py for the underlying client.
"""
from langchain_core.tools import tool


@tool
async def vector_search(query: str, top_k: int = 8) -> list[dict]:
    """Hybrid (vector + full-text) search over the document chunk index."""
    raise NotImplementedError
