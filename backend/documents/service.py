"""Background processing kicked off by upload (backend/api/routes/documents.py).

Extracts text once (backend/ingestion/text_extraction.py) and runs three
ingestion legs against it (backend/ingestion/pipeline.py): the graph leg
(ticket 07), the vector leg (ticket 06), and the tray-layout visual leg
(ticket 25). An extraction failure -- including an unsupported file format
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

from documents.repository import get_document, set_status
from ingestion.pipeline import ingest_document, ingest_document_vectors
from ingestion.text_extraction import extract_text
from ingestion.tray_layout_extraction import extract_tray_layout

logger = logging.getLogger(__name__)


async def process_document(document_id: uuid.UUID, storage_path: str) -> None:
    try:
        path = Path(storage_path)
        if not path.is_file():
            raise FileNotFoundError(storage_path)
        await set_status(document_id, "processing")

        doc = await get_document(document_id)
        assert doc is not None

        text = extract_text(path, doc.filename)

        await ingest_document(
            document_id=str(document_id),
            text=text,
            filename=doc.filename,
            system=doc.system_name,
            doc_type=doc.document_type_name,
        )
        await ingest_document_vectors(
            document_id=str(document_id),
            text=text,
            system_id=doc.system_id,
            document_type_id=doc.document_type_id,
        )

        try:
            await extract_tray_layout(
                document_id=str(document_id),
                storage_path=path,
                filename=doc.filename,
                system=doc.system_name,
            )
        except Exception:
            logger.exception(
                "tray-layout extraction raised despite its own best-effort guard "
                "for document_id=%s -- graph/vector legs already succeeded, "
                "not failing the document over this",
                document_id,
            )

        await set_status(document_id, "done")
    except Exception as exc:  # noqa: BLE001 - background task: report, don't crash the process
        await set_status(document_id, "failed", error=str(exc))
