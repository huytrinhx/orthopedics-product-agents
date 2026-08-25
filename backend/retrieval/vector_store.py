"""Postgres/pgvector-backed hybrid retrieval over the document chunk index.

Used by backend/agents/tools/vector_search.py and the ingestion pipeline
(backend/ingestion/pipeline.py). Hybrid here means vector similarity
(pgvector cosine distance) combined with Postgres full-text search
(tsvector/ts_rank) as the BM25-equivalent keyword leg — there's no
AI-Search-style synonym-map index to sync; query-time synonym expansion
comes from backend/agents/tools/synonym_resolve.py querying the graph
directly instead.
"""
from psycopg import AsyncConnection


class VectorStoreClient:
    def __init__(self, connection: AsyncConnection):
        self._connection = connection

    async def hybrid_search(self, query: str, vector: list[float], top_k: int = 8) -> list[dict]:
        raise NotImplementedError

    async def upsert_chunks(self, document_id: str, chunks: list[dict]) -> None:
        """Write chunk text + embedding vectors for one document."""
        raise NotImplementedError
