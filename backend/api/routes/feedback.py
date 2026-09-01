"""Captures per-message 4-axis feedback (faithfulness, relevance, style,
citation) plus an independent flag and optional comment from the chat UI.
Schema matches backend/agents/judge.py's EvalScores so human and automated
scores are directly comparable.

Keyed by message_id -- LangGraph's own auto-assigned per-message uuid (see
backend/api/routes/chat.py), not a surrogate id, see feedback/repository.py's
docstring. Resubmitting on the same message_id overwrites the previous row
(ticket 11's design: a user can correct a misclick, no separate edit flow).

Ticket 12 (free-text-only "Give feedback") reuses this same endpoint with
scores omitted -- see FeedbackRequest, every score is optional.

message_id isn't verified against thread_id's checkpointer transcript here --
the frontend only ever sends an id it already got from the backend (the
"done" SSE event or GET /chat/threads/{id}), so thread ownership is the only
real security boundary that matters; a malformed message_id would just be a
harmless orphan row nothing joins against.
"""
from fastapi import APIRouter, Depends, HTTPException, status

from auth.dependencies import get_current_user
from auth.repository import UserRecord
from chat_threads.repository import owns_thread
from feedback.models import FeedbackOut, FeedbackRequest
from feedback.repository import to_feedback_out, upsert_feedback

router = APIRouter()


@router.post("/", response_model=FeedbackOut)
async def submit_feedback(
    feedback: FeedbackRequest, user: UserRecord = Depends(get_current_user)
) -> FeedbackOut:
    if not owns_thread(str(user.id), feedback.thread_id):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Not your thread")

    record = await upsert_feedback(
        message_id=feedback.message_id,
        thread_id=feedback.thread_id,
        flagged=feedback.flagged,
        faithfulness=feedback.scores.get("faithfulness"),
        relevance=feedback.scores.get("relevance"),
        style=feedback.scores.get("style"),
        citation=feedback.scores.get("citation"),
        comment=feedback.comment,
        submitted_by=user.id,
    )
    return to_feedback_out(record)
