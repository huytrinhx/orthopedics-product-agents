"""End-to-end ingestion: raw document -> chunks -> embeddings -> pgvector
index, plus entity extraction -> AuraDB graph. Triggered from
backend/api/routes/documents.py on upload. Raw document bytes are written
under INGEST_DATA_DIR, not object storage.
"""
from ingestion.chunking import chunk_document  # noqa: F401
from ingestion.embedding import embed_texts  # noqa: F401
from ingestion.entity_extraction import extract_entities  # noqa: F401


async def ingest_document(document_id: str, text: str) -> None:
    raise NotImplementedError
