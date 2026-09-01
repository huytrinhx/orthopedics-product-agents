"""Pydantic response shapes for the documents routes."""
import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from tags.models import TagOut

DocumentStatus = Literal["pending", "queued", "processing", "done", "failed"]


class DocumentOut(BaseModel):
    id: uuid.UUID
    filename: str
    status: DocumentStatus
    error: str | None
    uploaded_by: uuid.UUID
    created_at: datetime
    updated_at: datetime
    system: TagOut | None
    document_type: TagOut | None


class SetDocumentTagsRequest(BaseModel):
    system_id: uuid.UUID | None = None
    document_type_id: uuid.UUID | None = None


class ChunkOut(BaseModel):
    chunk_index: int
    content: str
    section_title: str | None
