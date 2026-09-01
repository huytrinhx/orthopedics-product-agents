"""Pydantic response shapes for the chat threads routes."""
import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class ChatThreadOut(BaseModel):
    thread_id: str
    title: str
    created_at: datetime
    updated_at: datetime


class ChatCitationOut(BaseModel):
    """A resolved citation -- backend/api/routes/chat.py's _resolve_citations
    turns a raw "{document_id}#{chunk_index}" string (the graph state's
    portable, workflow-agnostic representation -- see
    agents/workflows/deterministic.py's finalize) into this display-ready
    shape by joining against the documents table. document_id + chunk_index
    together are what the chat citation viewer's GET
    /documents/{document_id}/chunks lookup needs to scroll to the exact
    chunk.
    """

    document_id: uuid.UUID
    filename: str
    chunk_index: int


class ChatMessageOut(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    citations: list[ChatCitationOut] = []


class ChatTranscriptOut(BaseModel):
    thread_id: str
    title: str
    messages: list[ChatMessageOut]
