"""Streaming chat endpoint. Streams LangGraph astream_events over SSE and
exposes a /resume endpoint for the suspend/resume human-clarification
pattern (interrupt() -> UI prompt -> resume with human input) -- that
pattern itself is ticket 09's scope; /resume stays a stub until then.
"""
import json
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from agents.registry import get_workflow, list_workflows
from auth.dependencies import get_current_user
from auth.repository import UserRecord

router = APIRouter()

# Node whose token-by-token output should stream to the UI as it's written.
# rerank/self_eval/reformulate also call the chat model, but for structured
# (JSON) output the user isn't meant to watch arrive token by token.
_STREAMED_NODES = {"generate"}


class ChatStreamRequest(BaseModel):
    message: str
    # Omitted (or the empty string) starts a new conversation; an existing
    # thread_id resumes that thread's history via the checkpointer. Never
    # trust a client-supplied thread_id as-is -- see _owns_thread below.
    thread_id: str | None = None


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _new_thread_id(user_id: str) -> str:
    # Prefixing with the owning user's id is what lets a resumed thread_id
    # be checked against the caller without a separate ownership table --
    # ticket 10 (chat history sidebar) can list a user's threads by the same
    # prefix once it needs to.
    return f"{user_id}:{uuid.uuid4().hex}"


def _owns_thread(user_id: str, thread_id: str) -> bool:
    return thread_id.startswith(f"{user_id}:")


@router.post("/{workflow_name}/stream")
async def stream_chat(
    workflow_name: str,
    body: ChatStreamRequest,
    request: Request,
    user: UserRecord = Depends(get_current_user),
) -> StreamingResponse:
    if workflow_name not in list_workflows():
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown workflow: {workflow_name}")

    thread_id = body.thread_id or _new_thread_id(str(user.id))
    if body.thread_id and not _owns_thread(str(user.id), body.thread_id):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Not your thread")

    graph = get_workflow(workflow_name, request.app.state.checkpointer)

    async def event_generator():
        yield _sse("thread", {"thread_id": thread_id})

        config = {"configurable": {"thread_id": thread_id}}
        inputs = {
            "messages": [HumanMessage(content=body.message)],
            "query": body.message,
            "user_id": str(user.id),
            "thread_id": thread_id,
            # search_query/retrieval_loop_count have no reducer, so an
            # earlier turn's checkpointed value would otherwise silently
            # carry over (e.g. a prior turn's retry already having spent
            # the loop budget) -- explicitly reset both for every new turn.
            "search_query": body.message,
            "retrieval_loop_count": 0,
        }

        try:
            async for event in graph.astream_events(inputs, config, version="v2"):
                node = event.get("metadata", {}).get("langgraph_node")
                if event["event"] == "on_chain_start" and event.get("name") == node:
                    yield _sse("status", {"node": node})
                elif event["event"] == "on_chat_model_stream" and node in _STREAMED_NODES:
                    chunk_content = event["data"]["chunk"].content
                    if chunk_content:
                        yield _sse("token", {"content": chunk_content})
                elif event["event"] == "on_chain_end" and event.get("name") == "LangGraph":
                    output = event["data"].get("output") or {}
                    yield _sse(
                        "done",
                        {
                            "thread_id": thread_id,
                            "answer": output.get("answer"),
                            "citations": output.get("citations") or [],
                            "eval_scores": output.get("eval_scores"),
                        },
                    )
        except Exception as exc:  # noqa: BLE001 - report over the stream rather than a bare 500 mid-stream
            yield _sse("error", {"message": str(exc)})

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/{workflow_name}/resume")
async def resume_chat(workflow_name: str, thread_id: str, human_input: str):
    raise NotImplementedError
