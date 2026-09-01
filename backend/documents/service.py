"""Background processing kicked off by upload (backend/api/routes/documents.py).

Extracts text once (backend/ingestion/text_extraction.py) and runs both
ingestion legs against it (backend/ingestion/pipeline.py): the graph leg
(ticket 07) and the vector leg (ticket 06). An extraction failure --
including an unsupported file format -- now fails the document instead of
silently skipping either leg.
"""
import uuid
from pathlib import Path

from documents.repository import get_document, set_status
from ingestion.pipeline import ingest_document, ingest_document_vectors
from ingestion.text_extraction import extract_text


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

        await set_status(document_id, "done")
    except Exception as exc:  # noqa: BLE001 - background task: report, don't crash the process
        await set_status(document_id, "failed", error=str(exc))
