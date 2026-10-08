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

Ticket 15 (Eval tab) adds the admin-only routes below: GET / lists every
feedback row, flagged or not, newest first, with its actual question/answer
text (read from the checkpointer, the same source of truth GET
/chat/threads/{id} uses -- not duplicated into the feedback table) and the
submitter's email; PATCH /{message_id}/resolved toggles the admin's
"confirmed fixed" marker (the UI only offers it on flagged rows); DELETE
/{message_id} removes an item outright (a duplicate, a misclick, or one no
longer worth keeping).

Rerunning an item used to be an inline, history-replaying POST /chat/rerun
with its own per-item rerun history (GET /{message_id}/reruns). Both were
removed 2026-10-07: the Eval tab now just opens the chat tab with the
question pre-sent in a fresh conversation (see build-log.md).
"""
from fastapi import APIRouter, Depends, HTTPException, Request, status

from auth.dependencies import get_current_user, require_admin
from auth.repository import UserRecord
from chat_threads.repository import owns_thread
from feedback.models import FeedbackListItemOut, FeedbackOut, FeedbackRequest, ResolvedRequest
from feedback.repository import (
    delete_feedback,
    list_all_feedback,
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
    """The rated AIMessage's own content, plus the HumanMessage
    immediately before it (the question that produced it) -- feedback is
    only ever collected on an assistant turn (ticket 11's UI only renders
    the scoring control there), so the preceding message is always the
    rep's actual question, not another assistant turn. None if the id
    isn't found (the message/thread was since deleted) or has no question
    before it (shouldn't happen for a real assistant turn, but a feedback
    row pointing at a message id from a fresher answer schema than
    expected shouldn't crash the whole list).
    """
    for index, message in enumerate(messages):
        if message.id == message_id:
            if index == 0 or messages[index - 1].type != "human":
                return None
            return messages[index - 1].content, message.content
    return None


@router.get("/", response_model=list[FeedbackListItemOut])
async def list_feedback(
    request: Request, admin: UserRecord = Depends(require_admin)
) -> list[FeedbackListItemOut]:
    checkpointer = request.app.state.checkpointer
    out = []
    for record, submitted_by_email in await list_all_feedback():
        checkpoint_tuple = await checkpointer.aget_tuple(
            {"configurable": {"thread_id": record.thread_id}}
        )
        if checkpoint_tuple is None:
            continue  # original thread's checkpoint is gone; nothing to show
        messages = checkpoint_tuple.checkpoint["channel_values"].get("messages", [])
        question_and_answer = _question_and_answer(messages, record.message_id)
        if question_and_answer is None:
            continue
        question, answer = question_and_answer
        out.append(
            FeedbackListItemOut(
                **to_feedback_out(record).model_dump(),
                question=question,
                answer=answer,
                submitted_by_email=submitted_by_email,
            )
        )
    return out


@router.patch("/{message_id}/resolved", response_model=FeedbackOut)
async def update_resolved(
    message_id: str, body: ResolvedRequest, admin: UserRecord = Depends(require_admin)
) -> FeedbackOut:
    record = await set_resolved(message_id, body.resolved)
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No flagged feedback for that message")
    return to_feedback_out(record)


@router.delete("/{message_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_feedback_item(message_id: str, admin: UserRecord = Depends(require_admin)) -> None:
    if not await delete_feedback(message_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No feedback for that message")
