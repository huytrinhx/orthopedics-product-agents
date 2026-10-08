"""LangGraph tool finding ingested documents by name -- plain word matching
over each document's filename and tags, no embedding call and no chunk
search. For when the rep names the document itself ("the MIS Inventory
Control Form", "REFLEX surgical technique") rather than asking about its
contents: vector_search only searches chunk text, so a title is only found
there if it happens to be repeated in the body, and it comes back as
scattered chunks rather than the document.

Each match carries its opening chunks with their citation ids (so the
answer can cite and link the document directly) plus its document_id, which
vector_search's document_ids filter takes to search within just that
document.
"""
import re

from langchain_core.tools import tool

from documents.repository import DocumentRecord, list_chunks, list_documents

# Filler words a rep naturally wraps a document name in ("the brochure for
# MIS") that would otherwise match every filename containing them.
_STOPWORDS = {"a", "an", "and", "doc", "document", "for", "of", "on", "the", "to"}
_OPENING_CHUNKS = 2
_PREVIEW_CHARACTERS = 1500


def _words(text: str) -> list[str]:
    # Filenames use _, -, and . as separators ("MIS_Inventory-Control.pdf"),
    # so split on anything that isn't a letter or digit.
    return [word for word in re.split(r"[^a-z0-9]+", text.lower()) if word]


def rank_documents_by_name(name: str, documents: list[DocumentRecord]) -> list[DocumentRecord]:
    """Documents matching `name`, best first: the whole name appearing in
    order beats scattered word hits, then more matched words beat fewer.
    Matches against filename, system tag, and document-type tag together,
    so "MIS brochure" finds "Brochure.pdf" tagged MIS. A document matching
    none of the name's meaningful words is left out.
    """
    query_words = [word for word in _words(name) if word not in _STOPWORDS]
    if not query_words:
        return []
    query_phrase = " ".join(query_words)
    scored: list[tuple[bool, int, DocumentRecord]] = []
    for document in documents:
        haystack_words = _words(
            " ".join(filter(None, [document.filename, document.system_name, document.document_type_name]))
        )
        haystack = " ".join(haystack_words)
        matched_word_count = sum(1 for word in query_words if word in haystack_words)
        if matched_word_count:
            scored.append((query_phrase in haystack, matched_word_count, document))
    scored.sort(key=lambda entry: (not entry[0], -entry[1], entry[2].filename.lower()))
    return [document for _, _, document in scored]


@tool
async def document_lookup(name: str, limit: int = 3) -> list[dict]:
    """Find ingested documents by name or title (e.g. "MIS Inventory Control
    Form", "REFLEX surgical technique") -- word matching on filename and
    system/document-type tags, not a content search. Returns each match's
    document_id, filename, tags, chunk_count, and its opening chunks with
    citation ids. To search inside a match, pass its document_id to
    vector_search's document_ids.
    """
    documents = [document for document in await list_documents() if document.status == "done"]
    results = []
    for document in rank_documents_by_name(name, documents)[:limit]:
        chunks = await list_chunks(document.id)
        results.append(
            {
                "document_id": str(document.id),
                "filename": document.filename,
                "system": document.system_name,
                "document_type": document.document_type_name,
                "chunk_count": len(chunks),
                "opening_chunks": [
                    {
                        "citation": f"{document.id}#{chunk.chunk_index}",
                        "section_title": chunk.section_title,
                        "content": chunk.content[:_PREVIEW_CHARACTERS],
                    }
                    for chunk in chunks[:_OPENING_CHUNKS]
                ],
            }
        )
    return results
