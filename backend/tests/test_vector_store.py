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
from catalog.repository import list_families_with_trays, upsert_families_and_trays
from documents.repository import create_document, delete_document
from retrieval.vector_store import RetrievalFilters, get_vector_store
from tags.repository import create_document_type

_EMBEDDING_DIM = 1536


def _unique(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


def _unit_vector(index: int) -> list[float]:
    vector = [0.0] * _EMBEDDING_DIM
    vector[index] = 1.0
    return vector


async def _create_document(document_type_id=None) -> uuid.UUID:
    user = await create_user(f"{_unique('user')}@example.com", None, False, True)
    document = await create_document(
        filename=_unique("doc") + ".txt",
        storage_path=f"/tmp/{_unique('storage')}.txt",
        uploaded_by=user.id,
        document_type_id=document_type_id,
    )
    return document.id


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

            async with store._connection.cursor() as cursor:
                await cursor.execute(
                    "SELECT chunk_index, content FROM chunks WHERE document_id = %s", (document_id,)
                )
                rows = await cursor.fetchall()

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

    async with get_vector_store() as store, store._connection.cursor() as cursor:
        await cursor.execute("SELECT count(*) FROM chunks WHERE document_id = %s", (document_id,))
        (count,) = await cursor.fetchone()
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

        result_indices = [result["chunk_index"] for result in results]
        # chunk 2 (strong keyword leg + present in the vector pool) and
        # chunk 0 (exact vector match) both outrank chunk 1, which has no
        # signal on either leg.
        assert set(result_indices) == {0, 2}
        assert results[0]["citation"] == f"{document_id}#{results[0]['chunk_index']}"
    finally:
        await delete_document(document_id)


async def test_hybrid_search_respects_document_type_filter():
    type_a_record = await create_document_type(_unique("TypeA"))
    type_b_record = await create_document_type(_unique("TypeB"))
    document_id = await _create_document(document_type_id=type_a_record.id)
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
                        "document_type_id": type_a_record.id,
                    },
                ],
            )

            matching = await store.hybrid_search(
                "shared content", query_vector, top_k=5,
                filters=RetrievalFilters(document_type_id=type_a_record.id),
            )
            non_matching = await store.hybrid_search(
                "shared content", query_vector, top_k=5,
                filters=RetrievalFilters(document_type_id=type_b_record.id),
            )

        assert len(matching) == 1
        assert non_matching == []
    finally:
        await delete_document(document_id)


async def test_family_filter_is_hard_and_ands_with_the_document_type_filter():
    """Issue #18: a family-scoped search returns only that family's chunks
    -- never another family's, and never a chunk with no family (unlike
    document_type_ids, which admits untyped chunks) -- and composes with
    the doc-type filter by AND.
    """
    family_name, other_family_name = _unique("Family"), _unique("OtherFamily")
    await upsert_families_and_trays({family_name: [], other_family_name: []})
    families_by_name = {family.name: family for family in await list_families_with_trays()}
    family_id = families_by_name[family_name].id
    other_family_id = families_by_name[other_family_name].id
    wanted_type = await create_document_type(_unique("WantedType"))
    other_type = await create_document_type(_unique("OtherType"))
    query_vector = _unit_vector(40)
    # (family, document type) per document -- every chunk an exact vector
    # and keyword match, so only the filters decide what comes back.
    shapes = {
        "family, wanted type": (family_id, wanted_type.id),
        "family, untyped": (family_id, None),
        "family, other type": (family_id, other_type.id),
        "other family": (other_family_id, wanted_type.id),
        "no family": (None, wanted_type.id),
    }
    document_ids = {label: await _create_document(document_type_id=type_id) for label, (_, type_id) in shapes.items()}
    labels_by_document_id = {str(document_id): label for label, document_id in document_ids.items()}
    try:
        async with get_vector_store() as store:
            for label, (shape_family_id, shape_type_id) in shapes.items():
                await store.upsert_chunks(
                    document_ids[label],
                    [
                        {
                            "chunk_index": 0,
                            "content": "family scoped torque guidance",
                            "embedding": query_vector,
                            "document_type_id": shape_type_id,
                            "product_family_id": shape_family_id,
                        }
                    ],
                )

            async def search(filters: RetrievalFilters) -> set[str]:
                results = await store.hybrid_search(
                    "family scoped torque guidance", query_vector, top_k=10, filters=filters
                )
                return {labels_by_document_id.get(result["document_id"]) for result in results}

            family_only = await search(RetrievalFilters(product_family_id=family_id))
            family_and_type = await search(
                RetrievalFilters(product_family_id=family_id, document_type_ids=[wanted_type.id])
            )

        assert family_only == {"family, wanted type", "family, untyped", "family, other type"}
        assert family_and_type == {"family, wanted type", "family, untyped"}
    finally:
        for document_id in document_ids.values():
            await delete_document(document_id)


async def test_hybrid_search_respects_document_ids_filter():
    """react_agent's document_lookup -> vector_search path: searching inside
    one named document must never return another document's chunks, even
    an exact vector match.
    """
    wanted_document_id = await _create_document()
    other_document_id = await _create_document()
    query_vector = _unit_vector(30)
    try:
        async with get_vector_store() as store:
            for document_id in (wanted_document_id, other_document_id):
                await store.upsert_chunks(
                    document_id,
                    [{"chunk_index": 0, "content": "torque limiting driver", "embedding": query_vector}],
                )

            results = await store.hybrid_search(
                "torque limiting driver", query_vector, top_k=5,
                filters=RetrievalFilters(document_ids=[wanted_document_id]),
            )

        assert [result["document_id"] for result in results] == [str(wanted_document_id)]
    finally:
        await delete_document(wanted_document_id)
        await delete_document(other_document_id)
