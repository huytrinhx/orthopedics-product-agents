"""Streaming chat endpoint. Streams LangGraph astream_events over SSE and
exposes a /resume endpoint for the suspend/resume human-clarification
pattern (interrupt() -> UI prompt -> resume with human input).
"""
from fastapi import APIRouter

router = APIRouter()


@router.post("/{workflow_name}/stream")
async def stream_chat(workflow_name: str, thread_id: str, user_id: str, message: str):
    raise NotImplementedError


@router.post("/{workflow_name}/resume")
async def resume_chat(workflow_name: str, thread_id: str, human_input: str):
    raise NotImplementedError
