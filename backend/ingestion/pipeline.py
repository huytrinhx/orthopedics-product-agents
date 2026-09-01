"""Two independent ingestion legs, both triggered from
backend/documents/service.py on upload/index/re-tag:

- The graph leg (ticket 07): raw document text -> entity extraction ->
  AuraDB graph (backend/ingestion/entity_extraction.py +
  backend/retrieval/graph_client.py). `ingest_document` below.
- The vector leg (ticket 06): raw document text -> chunking -> embedding ->
  pgvector (backend/ingestion/chunking.py + backend/ingestion/embedding.py +
  backend/retrieval/vector_store.py). `ingest_document_vectors` below.

The two are siblings, not a dependency of each other (see backend/evals/'s
ticket 07 schema design note on why these two ingestion paths don't have to
land together) -- they take different auxiliary params (names for the graph
leg's node properties vs. ids for the vector leg's denormalized filter
columns) and fail independently, so they're kept as separate functions
rather than merged into one.
"""
import os
import uuid

from ingestion.chunking import chunk_document
from ingestion.embedding import embed_texts
from ingestion.entity_extraction import extract_entities
from retrieval.graph_client import get_graph_client
from retrieval.vector_store import get_vector_store

_SPLIT_SIZE = 2000


def _split_for_extraction(text: str) -> list[str]:
    """Minimal paragraph-aware splitter so a whole document isn't sent to
    the LLM in one call. Intentionally not backend/ingestion/chunking.py's
    chunk_document — that's ticket 06's retrieval-tuned chunk boundary
    (pgvector-sized, overlap-tuned) and still raises NotImplementedError;
    this only needs "small enough for an extraction prompt," not the same
    boundaries the vector index uses.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if current and len(current) + len(paragraph) > _SPLIT_SIZE:
            chunks.append(current)
            current = paragraph
        else:
            current = f"{current}\n\n{paragraph}" if current else paragraph
    if current:
        chunks.append(current)
    return chunks


async def ingest_document(
    document_id: str,
    text: str,
    *,
    filename: str,
    system: str | None,
    doc_type: str | None,
) -> None:
    client = get_graph_client()
    await client.upsert_document(document_id, filename, doc_type=doc_type, system=system)

    if not os.environ.get("OPENAI_API_KEY"):
        # No LLM configured (e.g. CI without the secret, see ci.yml) --
        # skip prose extraction rather than failing the document. This is
        # an environment-config gap, not a per-document ingestion failure.
        return
    if not system:
        # Nothing to scope the known-parts/known-trays lookup to -- and
        # nothing in the master catalog to attach facts to either.
        return

    known_parts = await client.list_parts_for_family(system)
    known_trays = await client.list_trays_for_family(system)
    if not known_parts:
        # The master catalog (backend/ingestion/seed_master_catalog.py)
        # hasn't been seeded for this system yet -- there's nothing to
        # attach prose facts to.
        return

    for chunk in _split_for_extraction(text):
        for item in await extract_entities(chunk, known_parts=known_parts, known_trays=known_trays):
            if item["type"] == "differentiation":
                await client.attach_differentiation(
                    item["sku_a"], item["sku_b"], item["explanation"], document_id
                )
            elif item["type"] == "procedure_requirement":
                await client.attach_procedure(item["procedure"], item["tray"], document_id)


async def ingest_document_vectors(
    document_id: str,
    text: str,
    *,
    system_id: uuid.UUID | None,
    document_type_id: uuid.UUID | None,
) -> None:
    """Vector leg of ingestion: chunk -> embed -> pgvector."""
    if not os.environ.get("OPENAI_API_KEY"):
        # Same environment-config gap ingest_document guards above -- no LLM
        # configured (e.g. CI without the secret) means no embeddings, not a
        # per-document ingestion failure.
        return
    chunks = chunk_document(text)
    if not chunks:
        return
    vectors = await embed_texts([chunk["content"] for chunk in chunks])
    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector
        chunk["system_id"] = system_id
        chunk["document_type_id"] = document_type_id
    async with get_vector_store() as store:
        await store.upsert_chunks(document_id, chunks)
