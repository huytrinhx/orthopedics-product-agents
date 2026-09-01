"""LangGraph tool wrapping Postgres/pgvector hybrid retrieval.

See backend/retrieval/vector_store.py for the underlying client. No filter
args here yet -- nothing calls this tool until the ReAct graph
(backend/agents/workflows/react_agent.py, still NotImplementedError,
tickets 08/09) exists to validate a filtered-call shape against.
"""
from langchain_core.tools import tool

from ingestion.embedding import embed_texts
from retrieval.vector_store import get_vector_store


@tool
async def vector_search(query: str, top_k: int = 8) -> list[dict]:
    """Hybrid (vector + full-text) search over the document chunk index."""
    [vector] = await embed_texts([query])
    async with get_vector_store() as store:
        return await store.hybrid_search(query, vector, top_k=top_k)
