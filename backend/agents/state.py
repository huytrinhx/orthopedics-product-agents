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


# The four-axis rubric shared by agents/judge.py and human feedback
# (feedback/models.py). No longer part of a chat turn's own state -- inline
# self-eval checks facts deterministically now (agents/fact_check.py).
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
    # The chunk's document_type name (e.g. "Surgical Technique",
    # "Inventory Control"), or None for an untagged document -- ticket 24:
    # rerank weighs a candidate partly by which doctype it came from
    # (doctype-hierarchy.csv's priority order), which needs this.
    document_type: str | None


class BaseAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    query: str
    retrieved: list[RetrievedPassage]
    # Set by every workflow's self_eval (agents/fact_check.py): what's
    # wrong with the final draft, empty when it passed. Reset per turn by
    # the API layer, since workflows read it to decide whether the next
    # generate is a correction pass.
    fact_check_issues: list[str]
    user_id: str
    thread_id: str
