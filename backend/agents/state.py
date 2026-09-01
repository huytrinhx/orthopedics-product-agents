"""Shared LangGraph state schema used by every registered workflow.

Individual workflows may extend this with extra fields, but the API layer,
streaming transport, and eval harness only rely on the fields defined here —
that's what lets any workflow be run interchangeably.
"""
from typing import Annotated

from langgraph.graph.message import add_messages
from typing_extensions import (
    TypedDict,  # pydantic v2 needs this, not typing.TypedDict, on Python <3.12
)


class EvalScores(TypedDict, total=False):
    faithfulness: float
    relevance: float
    style: float
    citation: float


class RetrievedPassage(TypedDict):
    chunk_id: str
    document_id: str
    text: str
    score: float


class BaseAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    query: str
    retrieved: list[RetrievedPassage]
    eval_scores: EvalScores
    user_id: str
    thread_id: str
