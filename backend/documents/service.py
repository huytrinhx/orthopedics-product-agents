"""Document processing behind backend/api/routes/documents.py: the in-place
re-tag (retag_document), and the background indexing an admin triggers
with Index/Reindex or a re-upload (process_document).

Extracts text once (backend/ingestion/text_extraction.py) and runs three
ingestion legs against it (backend/ingestion/pipeline.py): the vector leg
(ticket 06), the graph leg (ticket 07), and the tray-layout visual leg
(ticket 25). After the vector leg, the per-document SKU index
(backend/document_skus/, issue #27) is rebuilt from the stored chunks; the
graph leg reads it to widen its scope, and after the graph leg the
document's techniques and steps (backend/techniques/, issue #28) are
rebuilt from the step chunks and the index.
An extraction failure -- including an unsupported file format
-- now fails the document instead of silently skipping any leg; the
tray-layout leg is the one exception, since it's explicitly best-effort by
design (backend/ingestion/tray_layout_extraction.py never raises itself,
but the log-and-continue below is a second line of defense so a change to
that module can't silently turn a best-effort leg into a hard failure for
every document).
"""
import logging
import uuid
from pathlib import Path

from document_skus.service import index_document_skus
from documents.repository import (
    DocumentRecord,
    get_document,
    mark_indexed,
    set_status,
    set_tags,
)
from ingestion.pipeline import ingest_document, ingest_document_vectors
from ingestion.text_extraction import extract_text
from ingestion.tray_layout_extraction import extract_tray_layout
from retrieval.graph_client import get_graph_client
from techniques.service import index_document_techniques

logger = logging.getLogger(__name__)


def missing_index_tags(document: DocumentRecord) -> list[str]:
    """The tags a document still needs before it may be indexed (issue
    #14): a document is complete with a Product Family, at least one Tray,
    and a Document Type. Empty means complete. Every indexing trigger
    (Index/Reindex, re-upload) checks this; retrieval deliberately doesn't
    -- an incomplete document indexed before the gate existed stays
    searchable, it just can't be re-indexed until it's completed.
    """
    missing = []
    if document.product_family_id is None:
        missing.append("Product Family")
    if not document.trays:
        missing.append("Tray")
    if document.document_type_id is None:
        missing.append("Document Type")
    return missing


async def retag_document(
    document_id: uuid.UUID,
    document_type_id: uuid.UUID | None,
    product_family_id: uuid.UUID | None,
    tray_ids: list[uuid.UUID],
) -> DocumentRecord | None:
    """Saves a document's tags and syncs them onto what indexing already
    wrote, in place (issue #13): set_tags rewrites the chunks' filter
    columns in its transaction, then the graph's Document node gets the new
    doc type/family/trays. No re-extraction, no re-embedding, no status
    change. Facts prose extraction already attached stay as they are --
    a Reindex is what re-scopes those to a new family.

    Raises TraysOutsideFamily / psycopg ForeignKeyViolation (nothing
    written) from set_tags, and lets a Neo4j error propagate after the
    Postgres write has committed -- the caller reports it, and a Reindex
    resyncs the graph.
    """
    document = await set_tags(
        document_id,
        document_type_id=document_type_id,
        product_family_id=product_family_id,
        tray_ids=tray_ids,
    )
    if document is None:
        return None
    # Catalog names, not display names -- what the graph's nodes are keyed on.
    await get_graph_client().set_document_tags(
        str(document_id),
        doc_type=document.document_type_name,
        product_family=document.product_family_name,
        trays=[tray.name for tray in document.trays],
    )
    return document


async def process_document(document_id: uuid.UUID, storage_path: str) -> None:
    try:
        path = Path(storage_path)
        if not path.is_file():
            raise FileNotFoundError(storage_path)
        await set_status(document_id, "processing")

        document = await get_document(document_id)
        assert document is not None

        text = extract_text(path, document.filename)

        await ingest_document_vectors(
            document_id=str(document_id),
            text=text,
            document_type_id=document.document_type_id,
            product_family_id=document.product_family_id,
        )
        # Scans the chunks the vector leg just stored (issue #27), so it
        # runs after it. Deterministic, no model call, and independent of
        # whether the graph leg finds any family parts.
        await index_document_skus(document_id, product_family=document.product_family_name)
        # Graph-side lookups go by the document's own Product Family
        # (catalog name), widened by the SKU index (issue #28), so the graph
        # leg runs after the index now.
        await ingest_document(
            document_id=str(document_id),
            text=text,
            filename=document.filename,
            product_family=document.product_family_name,
            trays=[tray.name for tray in document.trays],
            doc_type=document.document_type_name,
        )
        # Reads the step chunks and the SKU index (issue #28), and links its
        # Procedures to the Document node the graph leg just upserted.
        await index_document_techniques(document_id, product_family=document.product_family_name)

        try:
            await extract_tray_layout(
                document_id=str(document_id),
                storage_path=path,
                filename=document.filename,
                product_family=document.product_family_name,
            )
        except Exception:
            logger.exception(
                "tray-layout extraction raised despite its own best-effort guard "
                "for document_id=%s -- graph/vector legs already succeeded, "
                "not failing the document over this",
                document_id,
            )

        await mark_indexed(document_id)
    except Exception as error:  # noqa: BLE001 - background task: report, don't crash the process
        await set_status(document_id, "failed", error=str(error))
