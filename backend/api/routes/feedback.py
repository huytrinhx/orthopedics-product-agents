"""Captures conversation flags and 4-axis feedback (faithfulness,
relevance, style, citation) from the chat UI. Schema matches
backend/agents/judge.py's EvalScores so human and automated scores are
directly comparable.
"""
from fastapi import APIRouter
from pydantic import BaseModel

from agents.state import EvalScores

router = APIRouter()


class FeedbackRequest(BaseModel):
    thread_id: str
    message_id: str
    flagged: bool
    scores: EvalScores
    comment: str | None = None


@router.post("/")
async def submit_feedback(feedback: FeedbackRequest):
    raise NotImplementedError
