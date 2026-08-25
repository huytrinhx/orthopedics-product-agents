"""Document upload/list/status endpoints, backing the Document Manager UI.
Upload writes the raw file under INGEST_DATA_DIR and triggers
backend/ingestion/pipeline.py.
"""
from fastapi import APIRouter, UploadFile

router = APIRouter()


@router.post("/upload")
async def upload_document(file: UploadFile):
    raise NotImplementedError


@router.get("/")
async def list_documents():
    raise NotImplementedError
