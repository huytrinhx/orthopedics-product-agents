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

single_best_match is the strict form the deterministic workflow's
documents_lookup fast path uses: it only names a document when one clearly
wins, so a vague request still goes through retrieval and generation.
"""
import math
import re

from langchain_core.tools import tool

from documents.repository import DocumentRecord, list_chunks, list_documents

# Filler words a rep naturally wraps a document name in ("pull the brochure
# for MIS", "can you send me the MIS pdf") that would otherwise match every
# filename containing them -- "pdf" included, since every PDF's filename
# ends in it.
_STOPWORDS = {
    "a", "an", "and", "can", "copy", "could", "doc", "document", "file", "find", "for", "get",
    "give", "i", "is", "me", "need", "of", "on", "open", "pdf", "please", "pull", "send", "show",
    "the", "to", "up", "want", "you",
}
_OPENING_CHUNKS = 2
_PREVIEW_CHARACTERS = 1500
# single_best_match's floor: the winner has to carry at least this share of
# the request's distinct words, so one shared family word ("MIS") can't
# make a document the answer on its own.
_MIN_MATCHED_SHARE = 0.5


def _words(text: str) -> list[str]:
    # Filenames use _, -, and . as separators ("MIS_Inventory-Control.pdf"),
    # so split on anything that isn't a letter or digit.
    return [word for word in re.split(r"[^a-z0-9]+", text.lower()) if word]


def _name_words(name: str) -> list[str]:
    return [word for word in _words(name) if word not in _STOPWORDS]


def _document_words(document: DocumentRecord) -> list[str]:
    """What a document is known by: its filename, Product Family (catalog
    and display name), document-type tag, and Tray tags (catalog and
    display name), together.
    """
    tray_names = [name for tray in document.trays for name in (tray.name, tray.display_name)]
    return _words(
        " ".join(
            filter(
                None,
                [
                    document.filename,
                    document.product_family_name,
                    document.product_family_display_name,
                    document.document_type_name,
                    *tray_names,
                ],
            )
        )
    )


def names_document(name: str, document: DocumentRecord) -> bool:
    """True when every meaningful word of `name` is one of the document's
    words -- the strict form of rank_documents_by_name's match, for the
    answer eval (evals/answer_eval.py) to tell whether a golden dataset's
    expected citation ("MIS Inventory Control Form") is a document an
    answer cited, without one shared word like "MIS" counting.
    """
    query_words = _name_words(name)
    return bool(query_words) and set(query_words) <= set(_document_words(document))


def rank_documents_by_name(name: str, documents: list[DocumentRecord]) -> list[DocumentRecord]:
    """Documents matching `name`, best first: the whole name appearing in
    order beats scattered word hits, then more matched words beat fewer.
    Matches against filename, Product Family (catalog and display name),
    and document-type tag together, so "MIS brochure" finds "Brochure.pdf"
    tagged with the MIS Foot Recon family. A document matching none of the
    name's meaningful words is left out.
    """
    return [document for _, document in _scored_matches(name, documents)]


def _scored_matches(
    name: str, documents: list[DocumentRecord]
) -> list[tuple[tuple[bool, int], DocumentRecord]]:
    """Each document matching any of `name`'s meaningful words with its
    score -- (whole name appears in order, distinct words matched) --
    best first, ties by filename.
    """
    query_words = _name_words(name)
    if not query_words:
        return []
    query_phrase = " ".join(query_words)
    scored: list[tuple[tuple[bool, int], DocumentRecord]] = []
    for document in documents:
        haystack_words = _document_words(document)
        haystack = " ".join(haystack_words)
        matched_word_count = len(set(query_words) & set(haystack_words))
        if matched_word_count:
            scored.append(((query_phrase in haystack, matched_word_count), document))
    scored.sort(key=lambda entry: (not entry[0][0], -entry[0][1], entry[1].filename.lower()))
    return scored


def single_best_match(name: str, documents: list[DocumentRecord]) -> DocumentRecord | None:
    """The one document `name` clearly asks for, or None: the best-ranked
    match has to score strictly above the runner-up and carry at least
    _MIN_MATCHED_SHARE of the request's distinct words. A tie or a thin
    match is None, not a guess.
    """
    scored = _scored_matches(name, documents)
    if not scored:
        return None
    best_score, best_document = scored[0]
    if len(scored) > 1 and scored[1][0] == best_score:
        return None
    distinct_query_words = len(set(_name_words(name)))
    if best_score[1] < math.ceil(distinct_query_words * _MIN_MATCHED_SHARE):
        return None
    return best_document


@tool
async def document_lookup(name: str, limit: int = 3) -> list[dict]:
    """Find ingested documents by name or title (e.g. "MIS Inventory Control
    Form", "REFLEX surgical technique") -- word matching on filename and
    product-family/document-type tags, not a content search. Returns each match's
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
                "product_family": document.product_family_display_name,
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
