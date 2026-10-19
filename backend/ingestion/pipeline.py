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

Since issue #28 the graph leg runs after the vector leg and the SKU index
(backend/documents/service.py), because its scope is widened by the
families that index found; the two still don't share any other state.
"""
import logging
import os
import uuid

from document_skus.service import indexed_families
from ingestion.chunking import chunk_document, strip_page_markers
from ingestion.embedding import embed_texts
from ingestion.entity_extraction import extract_entities
from model_assignments.runtime import configured_slot_model, log_slot_models
from model_assignments.slots import Slot
from retrieval.graph_client import get_graph_client
from retrieval.vector_store import get_vector_store

logger = logging.getLogger(__name__)

_SPLIT_SIZE = 2000


def _split_for_extraction(text: str) -> list[str]:
    """Minimal paragraph-aware splitter so a whole document isn't sent to
    the LLM in one call. Intentionally not backend/ingestion/chunking.py's
    chunk_document — that's ticket 06's retrieval-tuned chunk boundary
    (pgvector-sized, overlap-tuned) and still raises NotImplementedError;
    this only needs "small enough for an extraction prompt," not the same
    boundaries the vector index uses.
    """
    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
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
    product_family: str | None,
    trays: list[str] | None = None,
    doc_type: str | None,
) -> None:
    """`product_family`/`trays` are the document's catalog names (not
    display names). Extraction is scoped to the whole family's parts, not
    just the document's own trays, plus the parts of every other family
    the document's SKU index found (issue #28): the REFLEX document names
    REFLEX HYBRID parts. So this runs after the SKU index is built.

    A re-index replaces the document's differentiations: the edges carry
    its `document_id`, and the old ones are deleted before the new ones
    are written."""
    client = get_graph_client()
    await client.upsert_document(
        document_id, filename, doc_type=doc_type, product_family=product_family, trays=trays
    )

    if not os.environ.get("OPENAI_API_KEY"):
        # No LLM configured (e.g. CI without the secret, see ci.yml) --
        # skip prose extraction rather than failing the document. This is
        # an environment-config gap, not a per-document ingestion failure.
        return
    if not product_family:
        # Nothing to scope the known-parts/known-trays lookup to -- and
        # nothing in the master catalog to attach facts to either.
        logger.warning(
            "entity_extraction_skipped document_id=%s: the document has no product family",
            document_id,
            extra={"event": "entity_extraction_skipped", "document_id": document_id},
        )
        return

    known_parts = await client.list_parts_for_family(product_family)
    if not known_parts:
        # The master catalog (backend/ingestion/seed_master_catalog.py)
        # hasn't been seeded for this family yet, or the document is tagged
        # with a family name the catalog no longer uses -- there's nothing
        # to attach prose facts to. Logged, not silent (issue #27): a silent
        # return here left documents with no extraction and no trace of why.
        # The per-document SKU index (backend/document_skus/) doesn't depend
        # on this and is built either way.
        logger.warning(
            "entity_extraction_skipped document_id=%s product_family=%r: "
            "the family has no catalog parts",
            document_id,
            product_family,
            extra={
                "event": "entity_extraction_skipped",
                "document_id": document_id,
                "product_family": product_family,
            },
        )
        return

    other_families = sorted(set(await indexed_families(document_id)) - {product_family})
    if other_families:
        known_skus = {part["sku"] for part in known_parts}
        known_parts += [
            part
            for part in await client.list_parts_for_families(other_families)
            if part["sku"] not in known_skus
        ]

    # Read once per document (issue #24): a slot change only affects
    # documents ingested after it, and nothing is re-ingested for it.
    slot_model = await configured_slot_model(Slot.ENTITY_EXTRACTION)
    log_slot_models(
        "ingestion_models", {Slot.ENTITY_EXTRACTION.value: slot_model}, document_id=document_id
    )
    differentiations: list[dict] = []
    for chunk in _split_for_extraction(strip_page_markers(text)):
        differentiations += await extract_entities(
            chunk, known_parts=known_parts, slot_model=slot_model
        )
    await client.replace_document_differentiations(document_id, differentiations)


def _embedding_input(chunk: dict) -> str:
    """The chunk's heading path above its content (issue #26): REFLEX's
    three techniques each have a "Step 1", and only the path says which
    one a chunk is. Embedded only -- the stored content, the keyword leg's
    tsvector and what answers quote stay the document's own text."""
    if not chunk.get("heading_path"):
        return chunk["content"]
    return " > ".join(chunk["heading_path"]) + "\n\n" + chunk["content"]


async def ingest_document_vectors(
    document_id: str,
    text: str,
    *,
    document_type_id: uuid.UUID | None,
    product_family_id: uuid.UUID | None = None,
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
    vectors = await embed_texts([_embedding_input(chunk) for chunk in chunks])
    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector
        chunk["document_type_id"] = document_type_id
        chunk["product_family_id"] = product_family_id
    async with get_vector_store() as store:
        await store.upsert_chunks(document_id, chunks)
