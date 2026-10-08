"""LangGraph tool wrapping Postgres/pgvector hybrid retrieval.

See backend/retrieval/vector_store.py for the underlying client.
`document_type_ids` (added 2026-09-04) is a real filter, but a narrow one in
practice: `agents/workflows/deterministic.py`'s hybrid_retrieve is the only
caller that passes it, resolved from the query's classified question type
via `_DOCTYPE_PRIORITY` -- it's exposed as a tool arg (rather than kept
private to that one caller) mainly so react_agent's model *could* pass real
ids it looked up itself in principle, though nothing prompts it to today; it
has no way to guess a real document_type UUID unprompted, so in practice
this stays None for that workflow, unfiltered, same as before this arg
existed. A chunk with no document_type at all always passes this filter
regardless (tagging is optional, ticket 05) -- see RetrievalFilters'
docstring.

`document_ids` is the other half of agents/tools/document_lookup.py: once
the model has found a document by name, it can search inside just that
document instead of the whole corpus.
"""
import uuid

from langchain_core.tools import tool

from ingestion.embedding import embed_texts
from retrieval.vector_store import RetrievalFilters, get_vector_store


@tool
async def vector_search(
    query: str,
    top_k: int = 8,
    document_type_ids: list[str] | None = None,
    document_ids: list[str] | None = None,
) -> list[dict]:
    """Hybrid (vector + full-text) search over the document chunk index.
    document_type_ids optionally restricts results to chunks tagged with one
    of these document type ids, plus any untagged chunk. document_ids
    optionally restricts results to chunks of these documents only (e.g. a
    document_id returned by document_lookup).
    """
    [vector] = await embed_texts([query])
    filters = (
        RetrievalFilters(
            document_type_ids=(
                [uuid.UUID(document_type_id) for document_type_id in document_type_ids] if document_type_ids else None
            ),
            document_ids=[uuid.UUID(document_id) for document_id in document_ids] if document_ids else None,
        )
        if document_type_ids or document_ids
        else None
    )
    async with get_vector_store() as store:
        return await store.hybrid_search(query, vector, top_k=top_k, filters=filters)
