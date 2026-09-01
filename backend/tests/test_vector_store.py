"""Exercises backend/retrieval/vector_store.py against real Postgres/pgvector
-- no mocking, same approach as test_documents.py/test_tags.py. Chunks are
written with small deterministic fake vectors (orthogonal unit vectors, not
real embeddings) so the storage/ranking SQL gets real coverage without an
OpenAI key -- only backend/ingestion/embedding.py's actual API call needs
that (see test_embedding.py).

Each test deletes the document it creates when done (cascading to its
chunks) -- hybrid_search has no document_id scoping (correct: cross-document
search is the point), so leftover rows with the same deterministic test
vectors from a prior run would otherwise pollute a later run's ranking pool.
"""
import uuid

from auth.repository import create_user
from documents.repository import create_document, delete_document
from retrieval.vector_store import RetrievalFilters, get_vector_store
from tags.repository import create_system

_EMBEDDING_DIM = 1536


def _unique(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


def _unit_vector(index: int) -> list[float]:
    vector = [0.0] * _EMBEDDING_DIM
    vector[index] = 1.0
    return vector


async def _create_document(system_id=None, document_type_id=None) -> uuid.UUID:
    user = await create_user(f"{_unique('user')}@example.com", None, False)
    doc = await create_document(
        filename=_unique("doc") + ".txt",
        storage_path=f"/tmp/{_unique('storage')}.txt",
        uploaded_by=user.id,
        system_id=system_id,
        document_type_id=document_type_id,
    )
    return doc.id


async def test_upsert_then_reindex_replaces_rather_than_duplicates():
    document_id = await _create_document()
    try:
        async with get_vector_store() as store:
            await store.upsert_chunks(
                document_id,
                [
                    {"chunk_index": 0, "content": "first version chunk one", "embedding": _unit_vector(0)},
                    {"chunk_index": 1, "content": "first version chunk two", "embedding": _unit_vector(1)},
                ],
            )
            await store.upsert_chunks(
                document_id,
                [{"chunk_index": 0, "content": "second version, only one chunk", "embedding": _unit_vector(2)}],
            )

            async with store._connection.cursor() as cur:
                await cur.execute(
                    "SELECT chunk_index, content FROM chunks WHERE document_id = %s", (document_id,)
                )
                rows = await cur.fetchall()

        assert rows == [(0, "second version, only one chunk")]
    finally:
        await delete_document(document_id)


async def test_document_delete_cascades_to_chunks():
    document_id = await _create_document()
    async with get_vector_store() as store:
        await store.upsert_chunks(
            document_id,
            [{"chunk_index": 0, "content": "will be orphaned if cascade is broken", "embedding": _unit_vector(3)}],
        )

    await delete_document(document_id)

    async with get_vector_store() as store, store._connection.cursor() as cur:
        await cur.execute("SELECT count(*) FROM chunks WHERE document_id = %s", (document_id,))
        (count,) = await cur.fetchone()
    assert count == 0


async def test_hybrid_search_fuses_vector_and_keyword_legs():
    document_id = await _create_document()
    query_vector = _unit_vector(10)
    try:
        async with get_vector_store() as store:
            await store.upsert_chunks(
                document_id,
                [
                    # Matches the query vector exactly; shares no keywords with the query.
                    {
                        "chunk_index": 0,
                        "content": "Bearing lubrication interval guidance for reprocessing.",
                        "embedding": query_vector,
                    },
                    # Orthogonal to the query vector and shares no keywords either --
                    # the irrelevant control.
                    {
                        "chunk_index": 1,
                        "content": "Sterile packaging seal integrity checklist.",
                        "embedding": _unit_vector(11),
                    },
                    # Orthogonal to the query vector, but keyword-dense on the query terms.
                    {
                        "chunk_index": 2,
                        "content": "Torque the screwdriver to the specified torque value; screwdriver torque matters.",
                        "embedding": _unit_vector(12),
                    },
                ],
            )

            results = await store.hybrid_search("screwdriver torque", query_vector, top_k=2)

        result_indices = [r["chunk_index"] for r in results]
        # chunk 2 (strong keyword leg + present in the vector pool) and
        # chunk 0 (exact vector match) both outrank chunk 1, which has no
        # signal on either leg.
        assert set(result_indices) == {0, 2}
        assert results[0]["citation"] == f"{document_id}#{results[0]['chunk_index']}"
    finally:
        await delete_document(document_id)


async def test_hybrid_search_respects_system_filter():
    system_a_record = await create_system(_unique("SystemA"))
    system_b_record = await create_system(_unique("SystemB"))
    document_id = await _create_document(system_id=system_a_record.id)
    query_vector = _unit_vector(20)

    try:
        async with get_vector_store() as store:
            await store.upsert_chunks(
                document_id,
                [
                    {
                        "chunk_index": 0,
                        "content": "shared content",
                        "embedding": query_vector,
                        "system_id": system_a_record.id,
                    },
                ],
            )

            matching = await store.hybrid_search(
                "shared content", query_vector, top_k=5,
                filters=RetrievalFilters(system_id=system_a_record.id),
            )
            non_matching = await store.hybrid_search(
                "shared content", query_vector, top_k=5,
                filters=RetrievalFilters(system_id=system_b_record.id),
            )

        assert len(matching) == 1
        assert non_matching == []
    finally:
        await delete_document(document_id)
