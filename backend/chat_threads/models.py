"""Pydantic response shapes for the chat threads routes."""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class ChatThreadOut(BaseModel):
    thread_id: str
    title: str
    created_at: datetime
    updated_at: datetime


class ChatMessageOut(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatTranscriptOut(BaseModel):
    thread_id: str
    title: str
    messages: list[ChatMessageOut]
