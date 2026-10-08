"""Pydantic response shapes for the chat threads routes."""
import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from feedback.models import FeedbackOut


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
    # None when the chunk carries no heading (or was removed/re-chunked
    # since the answer was generated) -- the UI falls back to the chunk
    # index so two citations into the same document are never shown
    # identically (see _resolve_citations' docstring).
    section_title: str | None = None


class ChatMessageOut(BaseModel):
    # LangGraph's own auto-assigned per-message uuid (add_messages) -- what
    # ticket 11's feedback controls key off of (backend/feedback/). Every
    # message has one once it's landed in the checkpointer, user and
    # assistant turns alike, but feedback is only ever collected on the
    # assistant's.
    message_id: str
    role: Literal["user", "assistant"]
    content: str
    citations: list[ChatCitationOut] = []
    # This message's own previously-submitted feedback, if any -- lets the
    # chat UI's scoring control reflect the submitted state on reload
    # instead of always starting blank.
    feedback: FeedbackOut | None = None


class ChatTranscriptOut(BaseModel):
    thread_id: str
    title: str
    messages: list[ChatMessageOut]
