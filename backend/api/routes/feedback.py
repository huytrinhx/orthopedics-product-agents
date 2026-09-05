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

Ticket 15 (Eval tab, rescoped from an eval-dataset harness into a
feedback-rerun tool after a 2026-09-05 grilling session -- see
.scratch/chat-documents-evals/issues/15-*.md) adds the admin-only routes
below: GET /flagged lists every flagged item with its actual question/answer
text (read from the checkpointer, the same source of truth GET
/chat/threads/{id} uses -- not duplicated into the feedback table), sorted
resolved-last so what's still outstanding stays on top; PATCH
/{message_id}/resolved toggles the admin's "confirmed fixed" marker; GET
/{message_id}/reruns lists that item's rerun history (POST /chat/rerun,
api/routes/chat.py, is what creates one); DELETE /{message_id} removes a
flagged item outright (a duplicate, a misclick, or one no longer worth
keeping) -- cascades to its rerun chat_threads rows, see migration
e2f039991c90.
"""
from fastapi import APIRouter, Depends, HTTPException, Request, status

from auth.dependencies import get_current_user, require_admin
from auth.repository import UserRecord
from chat_threads.models import RerunOut
from chat_threads.repository import list_reruns, owns_thread
from feedback.models import FeedbackOut, FeedbackRequest, FlaggedFeedbackOut, ResolvedRequest
from feedback.repository import (
    delete_feedback,
    list_flagged_feedback,
    set_resolved,
    to_feedback_out,
    upsert_feedback,
)

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


def _question_and_answer(messages: list, message_id: str) -> tuple[str, str] | None:
    """The flagged AIMessage's own content, plus the HumanMessage
    immediately before it (the question that produced it) -- feedback is
    only ever collected on an assistant turn (ticket 11's UI only renders
    the scoring control there), so the preceding message is always the
    rep's actual question, not another assistant turn. None if the id
    isn't found (the message/thread was since deleted) or has no question
    before it (shouldn't happen for a real assistant turn, but a flagged
    row pointing at a message id from a fresher answer schema than
    expected shouldn't crash the whole list).
    """
    for i, m in enumerate(messages):
        if m.id == message_id:
            if i == 0 or messages[i - 1].type != "human":
                return None
            return messages[i - 1].content, m.content
    return None


@router.get("/flagged", response_model=list[FlaggedFeedbackOut])
async def list_flagged(
    request: Request, admin: UserRecord = Depends(require_admin)
) -> list[FlaggedFeedbackOut]:
    checkpointer = request.app.state.checkpointer
    records = await list_flagged_feedback()
    out = []
    for record in records:
        checkpoint_tuple = await checkpointer.aget_tuple(
            {"configurable": {"thread_id": record.thread_id}}
        )
        if checkpoint_tuple is None:
            continue  # original thread's checkpoint is gone; nothing to show
        messages = checkpoint_tuple.checkpoint["channel_values"].get("messages", [])
        qa = _question_and_answer(messages, record.message_id)
        if qa is None:
            continue
        question, answer = qa
        out.append(
            FlaggedFeedbackOut(**to_feedback_out(record).model_dump(), question=question, answer=answer)
        )
    return out


@router.patch("/{message_id}/resolved", response_model=FeedbackOut)
async def update_resolved(
    message_id: str, body: ResolvedRequest, admin: UserRecord = Depends(require_admin)
) -> FeedbackOut:
    record = await set_resolved(message_id, body.resolved)
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No feedback for that message")
    return to_feedback_out(record)


@router.get("/{message_id}/reruns", response_model=list[RerunOut])
async def list_message_reruns(
    message_id: str, admin: UserRecord = Depends(require_admin)
) -> list[RerunOut]:
    return [
        RerunOut(thread_id=t.thread_id, workflow_name=t.workflow_name, created_at=t.created_at)
        for t in await list_reruns(message_id)
    ]


@router.delete("/{message_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_flagged(message_id: str, admin: UserRecord = Depends(require_admin)) -> None:
    if not await delete_feedback(message_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No feedback for that message")
