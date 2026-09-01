"""Pydantic response/request shapes for the feedback routes -- human-submitted
4-axis scores (matching backend/agents/judge.py's EvalScores so human and
automated scores stay directly comparable), keyed to a specific chat message.
Every score is optional: ticket 12 (free-text-only "Give feedback") submits
through the same FeedbackRequest with scores omitted entirely.
"""
import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from agents.state import EvalScores


class FeedbackRequest(BaseModel):
    thread_id: str
    message_id: str
    flagged: bool = False
    scores: EvalScores = Field(default_factory=dict)
    comment: str | None = None


class FeedbackOut(BaseModel):
    message_id: str
    thread_id: str
    flagged: bool
    scores: EvalScores
    comment: str | None
    submitted_by: uuid.UUID
    created_at: datetime
    updated_at: datetime
