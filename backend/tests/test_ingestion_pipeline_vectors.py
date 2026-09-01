"""Exercises backend/ingestion/pipeline.py's ingest_document_vectors (the
vector leg) against real Postgres -- no mocking of storage. embed_texts is
monkeypatched to a deterministic fake so this runs in CI with no OpenAI key;
only backend/ingestion/embedding.py's actual API call needs one (see
test_embedding.py). Mirrors test_ingestion_pipeline.py's split for the graph
leg: guard rails that don't need a real LLM call run here, unconditionally.
"""
import uuid

from auth.repository import create_user
from documents.repository import create_document, delete_document
from ingestion import pipeline
from retrieval.vector_store import get_vector_store


def _unique(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


async def _create_document() -> uuid.UUID:
    user = await create_user(f"{_unique('user')}@example.com", None, False)
    doc = await create_document(
        filename=_unique("doc") + ".txt",
        storage_path=f"/tmp/{_unique('storage')}.txt",
        uploaded_by=user.id,
    )
    return doc.id


async def _fake_embed_texts(texts: list[str]) -> list[list[float]]:
    return [[0.0] * 1536 for _ in texts]


async def test_ingest_document_vectors_writes_chunks(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")
    monkeypatch.setattr(pipeline, "embed_texts", _fake_embed_texts)

    document_id = await _create_document()
    try:
        await pipeline.ingest_document_vectors(
            str(document_id),
            "First paragraph.\n\nSecond paragraph.",
            system_id=None,
            document_type_id=None,
        )

        async with get_vector_store() as store, store._connection.cursor() as cur:
            await cur.execute(
                "SELECT chunk_index, content FROM chunks WHERE document_id = %s ORDER BY chunk_index",
                (document_id,),
            )
            rows = await cur.fetchall()

        assert len(rows) == 1  # both short paragraphs pack into a single chunk
        assert "First paragraph." in rows[0][1]
        assert "Second paragraph." in rows[0][1]
    finally:
        await delete_document(document_id)


async def test_ingest_document_vectors_denormalizes_tags(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")
    monkeypatch.setattr(pipeline, "embed_texts", _fake_embed_texts)
    from tags.repository import create_system

    system = await create_system(_unique("System"))
    document_id = await _create_document()
    try:
        await pipeline.ingest_document_vectors(
            str(document_id), "Some text about a part.", system_id=system.id, document_type_id=None
        )

        async with get_vector_store() as store, store._connection.cursor() as cur:
            await cur.execute(
                "SELECT system_id FROM chunks WHERE document_id = %s", (document_id,)
            )
            (system_id,) = await cur.fetchone()
        assert system_id == system.id
    finally:
        await delete_document(document_id)


async def test_ingest_document_vectors_skips_without_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    document_id = await _create_document()
    try:
        # Would raise (no key, and embed_texts isn't monkeypatched here) if
        # the guard didn't short-circuit before any embedding call.
        await pipeline.ingest_document_vectors(
            str(document_id), "Some text.", system_id=None, document_type_id=None
        )

        async with get_vector_store() as store, store._connection.cursor() as cur:
            await cur.execute("SELECT count(*) FROM chunks WHERE document_id = %s", (document_id,))
            (count,) = await cur.fetchone()
        assert count == 0
    finally:
        await delete_document(document_id)
