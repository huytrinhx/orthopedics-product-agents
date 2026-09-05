"""Postgres/pgvector-backed hybrid retrieval over the document chunk index.

Used by backend/agents/tools/vector_search.py and the ingestion pipeline
(backend/ingestion/pipeline.py). Hybrid here means vector similarity
(pgvector cosine distance) combined with Postgres full-text search
(tsvector/ts_rank) as the BM25-equivalent keyword leg, fused with
Reciprocal Rank Fusion -- there's no AI-Search-style synonym-map index to
sync; query-time synonym expansion comes from
backend/agents/tools/synonym_resolve.py querying the graph directly
instead.

Deliberately not backend/config/db.py's get_connection() -- that shared
helper doesn't know about pgvector's Python-list<->`vector` type adapter,
and every other consumer of it (auth, documents, tags, feedback) has no
reason to pay for registering it.
"""
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import psycopg
from pgvector.psycopg import register_vector_async
from psycopg import AsyncConnection
from psycopg.rows import dict_row

# Reciprocal Rank Fusion: 1 / (k + rank). k=10, not the original RRF paper's
# k=60 (tuned for large web-scale rank lists) -- top_k here is single
# digits, and a large k flattens the fusion score into near-indistinguishable
# noise at this scale.
_RRF_K = 10
# Each leg pulls this many times top_k before fusion -- zero oversampling
# would give RRF nothing extra to promote from either leg, defeating the
# point of fusing two ranked lists.
_CANDIDATE_MULTIPLIER = 2


@dataclass
class RetrievalFilters:
    system_id: uuid.UUID | None = None
    document_type_id: uuid.UUID | None = None
    # An inclusion list, not an exclusion filter -- a chunk with NO
    # document_type (tagging is optional, ticket 05) always passes this
    # filter regardless of what's listed here, so an untagged document
    # never becomes silently unreachable for a question type that has a
    # doctype preference. deterministic.py's hybrid_retrieve (2026-09-04)
    # is the one caller: it resolves a question type's allowed doctypes
    # (_DOCTYPE_PRIORITY) to ids and passes them here so the *retrieval*
    # pool itself is narrowed to relevant document types, not just
    # re-ranked afterward -- rerank's existing doctype-priority bonus still
    # ranks an allowed-type chunk above an untagged one within that pool.
    document_type_ids: list[uuid.UUID] | None = None


class VectorStoreClient:
    def __init__(self, connection: AsyncConnection):
        self._connection = connection

    async def upsert_chunks(self, document_id: uuid.UUID | str, chunks: list[dict]) -> None:
        """Write chunk text + embedding vectors for one document.

        Delete-then-insert inside one transaction: makes a Reindex
        idempotent (no stale trailing chunks if the new text produces fewer
        chunks than before) with no extra bookkeeping to track which rows
        are "current."
        """
        async with self._connection.cursor() as cur:
            await cur.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))
            for chunk in chunks:
                await cur.execute(
                    """
                    INSERT INTO chunks
                        (document_id, chunk_index, content, section_title,
                         embedding, system_id, document_type_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        document_id,
                        chunk["chunk_index"],
                        chunk["content"],
                        chunk.get("section_title"),
                        chunk["embedding"],
                        chunk.get("system_id"),
                        chunk.get("document_type_id"),
                    ),
                )
        await self._connection.commit()

    async def hybrid_search(
        self,
        query: str,
        vector: list[float],
        top_k: int = 8,
        filters: RetrievalFilters | None = None,
    ) -> list[dict]:
        pool_size = top_k * _CANDIDATE_MULTIPLIER
        conditions: list[str] = []
        condition_params: list[uuid.UUID] = []
        if filters and filters.system_id:
            conditions.append("c.system_id = %s")
            condition_params.append(filters.system_id)
        if filters and filters.document_type_id:
            conditions.append("c.document_type_id = %s")
            condition_params.append(filters.document_type_id)
        if filters and filters.document_type_ids:
            conditions.append("(c.document_type_id IS NULL OR c.document_type_id = ANY(%s))")
            condition_params.append(list(filters.document_type_ids))
        filter_sql = f"AND {' AND '.join(conditions)}" if conditions else ""

        async with self._connection.cursor(row_factory=dict_row) as cur:
            # The explicit ::vector cast matters: psycopg sends a plain
            # Python list as a `double precision[]` parameter, and while
            # Postgres accepts that in an INSERT (an implicit assignment
            # cast into a known `vector` column), the `<=>` operator has no
            # `vector <=> double precision[]` overload -- it needs the
            # parameter cast explicitly to match `<=>`'s declared operand
            # types.
            #
            # document_type name (not just id) is joined in here -- ticket
            # 24: reranking weighs a candidate partly by which doctype it
            # came from (doctype-hierarchy.csv's priority order), which
            # needs the name, not just the FK, and doing that join per-row
            # later would mean N extra queries instead of one JOIN here.
            await cur.execute(
                f"""
                SELECT c.id, c.document_id, c.chunk_index, c.content, c.section_title,
                       dt.name AS document_type
                FROM chunks c
                LEFT JOIN document_types dt ON dt.id = c.document_type_id
                WHERE true {filter_sql}
                ORDER BY c.embedding <=> %s::vector
                LIMIT %s
                """,
                (*condition_params, vector, pool_size),
            )
            vector_rows = await cur.fetchall()

            await cur.execute(
                f"""
                SELECT c.id, c.document_id, c.chunk_index, c.content, c.section_title,
                       dt.name AS document_type
                FROM chunks c
                LEFT JOIN document_types dt ON dt.id = c.document_type_id
                WHERE c.tsv @@ plainto_tsquery('english', %s) {filter_sql}
                ORDER BY ts_rank(c.tsv, plainto_tsquery('english', %s)) DESC
                LIMIT %s
                """,
                (query, *condition_params, query, pool_size),
            )
            keyword_rows = await cur.fetchall()

        return _reciprocal_rank_fusion(vector_rows, keyword_rows, top_k)


def _reciprocal_rank_fusion(
    vector_rows: list[dict], keyword_rows: list[dict], top_k: int
) -> list[dict]:
    scores: dict[uuid.UUID, float] = {}
    rows_by_id: dict[uuid.UUID, dict] = {}
    for rows in (vector_rows, keyword_rows):
        for rank, row in enumerate(rows, start=1):
            scores[row["id"]] = scores.get(row["id"], 0.0) + 1 / (_RRF_K + rank)
            rows_by_id[row["id"]] = row

    ranked_ids = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)[:top_k]
    return [
        {
            "document_id": str(rows_by_id[chunk_id]["document_id"]),
            "chunk_index": rows_by_id[chunk_id]["chunk_index"],
            "content": rows_by_id[chunk_id]["content"],
            "section_title": rows_by_id[chunk_id]["section_title"],
            "document_type": rows_by_id[chunk_id]["document_type"],
            "score": scores[chunk_id],
            "citation": f"{rows_by_id[chunk_id]['document_id']}#{rows_by_id[chunk_id]['chunk_index']}",
        }
        for chunk_id in ranked_ids
    ]


@asynccontextmanager
async def get_vector_store() -> AsyncIterator[VectorStoreClient]:
    conn = await psycopg.AsyncConnection.connect(os.environ["DATABASE_URL"])
    await register_vector_async(conn)
    try:
        yield VectorStoreClient(conn)
    finally:
        await conn.close()
