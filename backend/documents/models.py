"""Pydantic response shapes for the documents routes."""
import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from catalog.models import TrayOut
from tags.models import TagOut

DocumentStatus = Literal["pending", "queued", "processing", "done", "failed"]


class DocumentProductFamilyOut(BaseModel):
    id: uuid.UUID
    name: str
    display_name: str


class DocumentOut(BaseModel):
    id: uuid.UUID
    filename: str
    status: DocumentStatus
    error: str | None
    uploaded_by: uuid.UUID
    created_at: datetime
    updated_at: datetime
    # Legacy -- still returned until the legacy-removal ticket drops the
    # `systems` table, but no longer shown in the UI.
    system: TagOut | None
    document_type: TagOut | None
    product_family: DocumentProductFamilyOut | None
    trays: list[TrayOut]


class SetDocumentTagsRequest(BaseModel):
    """A field left out of the request body keeps its current value; one
    sent as null clears it. `product_family_id` and `tray_ids` travel
    together -- sending either replaces both (a missing `tray_ids` then
    means no trays), since a tray set only makes sense against a family."""

    system_id: uuid.UUID | None = None
    document_type_id: uuid.UUID | None = None
    product_family_id: uuid.UUID | None = None
    tray_ids: list[uuid.UUID] = []


class ChunkOut(BaseModel):
    chunk_index: int
    content: str
    section_title: str | None
    # The page this chunk's content starts on, in the source PDF (ticket
    # 26) -- None for non-PDF documents and for chunks written before this
    # shipped (no backfill migration).
    page_number: int | None


class ComponentHealth(BaseModel):
    ok: bool
    detail: str


class SystemHealthOut(BaseModel):
    volume: ComponentHealth
    graph_db: ComponentHealth
    vector_db: ComponentHealth
