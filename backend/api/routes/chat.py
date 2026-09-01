"""Streaming chat endpoint. Streams LangGraph astream_events over SSE.

Also exposes the suspend/resume human-clarification pattern (ticket 09):
detect_intent (agents/workflows/deterministic.py, the graph's entry node)
calls interrupt() when it can't confidently tell which product system a
query is about; astream_events surfaces that as an on_chain_stream "LangGraph"
chunk carrying an "__interrupt__" tuple rather than raising, so _stream_graph
below watches for that chunk shape and emits a "clarification" SSE event
instead of "done". POST /{workflow_name}/resume (and its default-workflow
sibling POST /resume, mirroring the /stream pair) resumes the same suspended
graph via langgraph.types.Command(resume=...) -- not a fresh turn, so no new
create_thread/HumanMessage, just touch_thread for recency.

POST /stream (no workflow name) is what the chat UI actually calls -- it
runs whatever workflow is configured as the admin default (ticket 14,
settings/repository.py). POST /{workflow_name}/stream still exists
alongside it for explicit selection (evals/testing); /resume mirrors that
same pairing for the same reason.

Also exposes the ticket 10 (chat history sidebar) read endpoints:
GET /threads (a user's threads, most-recent-first) and
GET /threads/{thread_id} (one thread's transcript, read straight from the
checkpointer rather than duplicated in chat_threads -- see
chat_threads/repository.py's docstring).
"""
import json
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from pydantic import BaseModel

from agents.registry import get_workflow, list_workflows
from auth.dependencies import get_current_user
from auth.repository import UserRecord
from chat_threads.models import ChatCitationOut, ChatMessageOut, ChatThreadOut, ChatTranscriptOut
from chat_threads.repository import (
    create_thread,
    get_thread,
    list_threads,
    owns_thread,
    touch_thread,
)
from documents.repository import get_document
from feedback.repository import get_feedback_for_thread, to_feedback_out
from observability.langfuse_setup import get_trace_url, new_callback_handler, score_trace
from settings.repository import get_settings

router = APIRouter()

# Node whose token-by-token output should stream to the UI as it's written.
# rerank/self_eval/reformulate also call the chat model, but for structured
# (JSON) output the user isn't meant to watch arrive token by token.
_STREAMED_NODES = {"generate"}

# Sidebar thread titles are derived from the first message rather than set
# by the user -- long enough to stay recognizable, short enough to fit one
# sidebar row without wrapping.
_TITLE_MAX_LEN = 60


def _title_from_message(message: str) -> str:
    collapsed = " ".join(message.split())
    if len(collapsed) <= _TITLE_MAX_LEN:
        return collapsed
    return collapsed[:_TITLE_MAX_LEN].rstrip() + "…"


# Each registered workflow tracks retry/tool-call iterations under its own
# state field name and meaning (see agents/workflows/*.py) -- this maps
# workflow_name -> that field, for the one place (score_trace) that needs a
# single "loop_count" number rather than each workflow's own semantics.
# Workflows not listed here just report no loop_count.
_LOOP_COUNT_FIELD = {"deterministic": "retrieval_loop_count"}


class ChatStreamRequest(BaseModel):
    message: str
    # Omitted (or the empty string) starts a new conversation; an existing
    # thread_id resumes that thread's history via the checkpointer. Never
    # trust a client-supplied thread_id as-is -- see chat_threads.repository.owns_thread.
    thread_id: str | None = None


class ChatResumeRequest(BaseModel):
    thread_id: str
    # Either the exact text of a clicked clarification option, or free
    # text -- detect_intent's _match_option tries to match it against the
    # options it offered, and falls back to leaving resolved_system
    # unresolved (not the raw text) if nothing matches, rather than
    # recording an answer that isn't actually one of the real systems.
    human_input: str


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _new_thread_id(user_id: str) -> str:
    # Prefixing with the owning user's id is what lets a resumed thread_id
    # be checked against the caller without a separate ownership table --
    # ticket 10 (chat history sidebar) can list a user's threads by the same
    # prefix once it needs to.
    return f"{user_id}:{uuid.uuid4().hex}"


async def _resolve_citations(raw_citations: list[str]) -> list[ChatCitationOut]:
    """Turns the graph state's raw "{document_id}#{chunk_index}" strings
    (agents/workflows/deterministic.py's _extract_citations) into the
    filename the UI actually shows and needs to open the citation viewer --
    used both for a turn that just streamed and for a resumed thread's
    transcript (get_chat_thread below), since both start from the same raw
    strings. One documents lookup per unique document_id, not per citation
    -- a single answer can (and often does) cite several chunks from the
    same document.
    """
    filenames: dict[str, str] = {}
    resolved: list[ChatCitationOut] = []
    for raw in raw_citations:
        document_id, _, chunk_index = raw.rpartition("#")
        if document_id not in filenames:
            doc = await get_document(uuid.UUID(document_id))
            # A citation can outlive the document it points to (deleted
            # after the answer was generated) -- still worth showing the
            # chip rather than dropping the citation silently, just without
            # a real filename to click through to.
            filenames[document_id] = doc.filename if doc else "(document removed)"
        resolved.append(
            ChatCitationOut(
                document_id=uuid.UUID(document_id),
                filename=filenames[document_id],
                chunk_index=int(chunk_index),
            )
        )
    return resolved


@router.post("/{workflow_name}/stream")
async def stream_chat(
    workflow_name: str,
    body: ChatStreamRequest,
    request: Request,
    user: UserRecord = Depends(get_current_user),
) -> StreamingResponse:
    """Explicit workflow selection -- used by evals/testing to run a named
    workflow directly. The chat UI itself never calls this; it calls
    POST /chat/stream below, which resolves the admin-configured default
    (ticket 14) and delegates here. Kept as a separate route (rather than
    making workflow_name optional on one route) so this 404-on-unknown-name
    behavior and every existing explicit-workflow test stay unchanged.
    """
    return await _stream_chat(workflow_name, body, request, user)


@router.post("/stream")
async def stream_chat_default(
    body: ChatStreamRequest,
    request: Request,
    user: UserRecord = Depends(get_current_user),
) -> StreamingResponse:
    settings = await get_settings()
    return await _stream_chat(settings.default_workflow, body, request, user)


async def _stream_chat(
    workflow_name: str,
    body: ChatStreamRequest,
    request: Request,
    user: UserRecord,
) -> StreamingResponse:
    if workflow_name not in list_workflows():
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown workflow: {workflow_name}")

    is_new_thread = body.thread_id is None
    thread_id = body.thread_id or _new_thread_id(str(user.id))
    if body.thread_id and not owns_thread(str(user.id), body.thread_id):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Not your thread")

    # Sidebar bookkeeping (ticket 10) -- a separate table from the
    # checkpointer's own (see chat_threads/repository.py), so this can't
    # fail/skip the actual chat turn; it only ever adds a row or bumps one.
    if is_new_thread:
        await create_thread(thread_id, user.id, _title_from_message(body.message))
    else:
        await touch_thread(thread_id)

    graph = get_workflow(workflow_name, request.app.state.checkpointer)
    handler, langfuse_config = new_callback_handler(
        user_id=str(user.id), session_id=thread_id, tags=[workflow_name]
    )
    config = {"configurable": {"thread_id": thread_id}, **langfuse_config}
    inputs = {
        "messages": [HumanMessage(content=body.message)],
        "query": body.message,
        "user_id": str(user.id),
        "thread_id": thread_id,
        # search_query/retrieval_loop_count have no reducer, so an earlier
        # turn's checkpointed value would otherwise silently carry over
        # (e.g. a prior turn's retry already having spent the loop budget)
        # -- explicitly reset both for every new turn.
        "search_query": body.message,
        "retrieval_loop_count": 0,
    }

    return StreamingResponse(
        _stream_graph(graph, workflow_name, thread_id, inputs, config, handler),
        media_type="text/event-stream",
    )


async def _stream_graph(
    graph: CompiledStateGraph,
    workflow_name: str,
    thread_id: str,
    run_input,
    config: dict,
    handler,
) -> AsyncIterator[str]:
    """Drives one graph run (a fresh turn's `inputs` dict, or a resume's
    `Command(resume=...)`) and turns its astream_events into SSE frames --
    shared by stream_chat/stream_chat_default and resume_chat/
    resume_chat_default below, since both sides of the suspend/resume pair
    need identical status/token/done handling, just a different `run_input`.

    A node calling interrupt() doesn't raise through astream_events -- it
    surfaces as an on_chain_stream event named "LangGraph" whose chunk is
    `{"__interrupt__": (Interrupt(value=..., id=...),)}` (verified against
    the installed langgraph 1.2 directly, since the docs don't show this
    astream_events shape). The graph's own on_chain_end "LangGraph" event
    still fires right after with the pre-interrupt state as `output` --
    `interrupted` is what stops that from being mistaken for a real answer
    and emitted as "done".
    """
    yield _sse("thread", {"thread_id": thread_id})

    interrupted = False
    try:
        async for event in graph.astream_events(run_input, config, version="v2"):
            node = event.get("metadata", {}).get("langgraph_node")
            if event["event"] == "on_chain_start" and event.get("name") == node:
                yield _sse("status", {"node": node})
            elif event["event"] == "on_chat_model_stream" and node in _STREAMED_NODES:
                chunk_content = event["data"]["chunk"].content
                if chunk_content:
                    yield _sse("token", {"content": chunk_content})
            elif event["event"] == "on_chain_stream" and event.get("name") == "LangGraph":
                interrupt_tuple = (event["data"].get("chunk") or {}).get("__interrupt__")
                if interrupt_tuple:
                    interrupted = True
                    payload = interrupt_tuple[0].value or {}
                    yield _sse(
                        "clarification",
                        {
                            "thread_id": thread_id,
                            "question": payload.get("question", ""),
                            "options": payload.get("options", []),
                        },
                    )
            elif event["event"] == "on_chain_end" and event.get("name") == "LangGraph":
                if interrupted:
                    continue
                output = event["data"].get("output") or {}
                loop_count_field = _LOOP_COUNT_FIELD.get(workflow_name)
                loop_count = output.get(loop_count_field) if loop_count_field else None
                score_trace(
                    handler.last_trace_id,
                    eval_scores=output.get("eval_scores"),
                    loop_count=loop_count,
                )
                citations = await _resolve_citations(output.get("citations") or [])
                # finalize (agents/workflows/deterministic.py) is the only
                # place messages gets an AI turn appended, so the last
                # message in the merged state is always the answer just
                # generated -- add_messages already assigned it a stable
                # uuid .id, which is what ticket 11's feedback controls key
                # off of.
                final_messages = output.get("messages") or []
                message_id = final_messages[-1].id if final_messages else None
                yield _sse(
                    "done",
                    {
                        "thread_id": thread_id,
                        "message_id": message_id,
                        "answer": output.get("answer"),
                        "citations": [c.model_dump(mode="json") for c in citations],
                        "eval_scores": output.get("eval_scores"),
                        "trace_url": get_trace_url(handler.last_trace_id),
                    },
                )
    except Exception as exc:  # noqa: BLE001 - report over the stream rather than a bare 500 mid-stream
        yield _sse("error", {"message": str(exc)})


@router.post("/{workflow_name}/resume")
async def resume_chat(
    workflow_name: str,
    body: ChatResumeRequest,
    request: Request,
    user: UserRecord = Depends(get_current_user),
) -> StreamingResponse:
    """Explicit-workflow counterpart to resume_chat_default below, same
    split as stream_chat/stream_chat_default and for the same reason."""
    return await _resume_chat(workflow_name, body, request, user)


@router.post("/resume")
async def resume_chat_default(
    body: ChatResumeRequest,
    request: Request,
    user: UserRecord = Depends(get_current_user),
) -> StreamingResponse:
    settings = await get_settings()
    return await _resume_chat(settings.default_workflow, body, request, user)


async def _resume_chat(
    workflow_name: str,
    body: ChatResumeRequest,
    request: Request,
    user: UserRecord,
) -> StreamingResponse:
    if workflow_name not in list_workflows():
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown workflow: {workflow_name}")
    if not owns_thread(str(user.id), body.thread_id):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Not your thread")

    graph = get_workflow(workflow_name, request.app.state.checkpointer)
    config = {"configurable": {"thread_id": body.thread_id}}
    state = await graph.aget_state(config)
    if not state.next:
        raise HTTPException(
            status.HTTP_409_CONFLICT, "Thread is not waiting on a clarification"
        )

    await touch_thread(body.thread_id)

    handler, langfuse_config = new_callback_handler(
        user_id=str(user.id), session_id=body.thread_id, tags=[workflow_name]
    )
    config = {**config, **langfuse_config}
    return StreamingResponse(
        _stream_graph(
            graph, workflow_name, body.thread_id, Command(resume=body.human_input), config, handler
        ),
        media_type="text/event-stream",
    )


@router.get("/threads", response_model=list[ChatThreadOut])
async def list_chat_threads(user: UserRecord = Depends(get_current_user)) -> list[ChatThreadOut]:
    return [
        ChatThreadOut(
            thread_id=t.thread_id, title=t.title, created_at=t.created_at, updated_at=t.updated_at
        )
        for t in await list_threads(user.id)
    ]


@router.get("/threads/{thread_id}", response_model=ChatTranscriptOut)
async def get_chat_thread(
    thread_id: str,
    request: Request,
    user: UserRecord = Depends(get_current_user),
) -> ChatTranscriptOut:
    if not owns_thread(str(user.id), thread_id):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Not your thread")

    # Transcript comes straight from the checkpointer (the actual source of
    # conversation state, see memory/checkpointer.py) rather than from
    # chat_threads -- that table only ever tracks sidebar metadata (title,
    # recency), never message content.
    checkpointer = request.app.state.checkpointer
    checkpoint_tuple = await checkpointer.aget_tuple({"configurable": {"thread_id": thread_id}})
    thread = await get_thread(thread_id)
    if checkpoint_tuple is None or thread is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Thread not found")

    messages = checkpoint_tuple.checkpoint["channel_values"].get("messages", [])
    feedback_by_message = await get_feedback_for_thread(thread_id)
    out_messages = []
    for m in messages:
        # Citations only ever live on the AI's turn (see deterministic.py's
        # finalize) -- additional_kwargs is a plain dict on every message
        # type, so .get is safe on a HumanMessage too, it's just always
        # empty there.
        citations = await _resolve_citations(m.additional_kwargs.get("citations") or [])
        record = feedback_by_message.get(m.id)
        out_messages.append(
            ChatMessageOut(
                message_id=m.id,
                role="user" if m.type == "human" else "assistant",
                content=m.content,
                citations=citations,
                feedback=to_feedback_out(record) if record else None,
            )
        )
    return ChatTranscriptOut(thread_id=thread_id, title=thread.title, messages=out_messages)
