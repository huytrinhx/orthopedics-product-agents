"""Exercises backend/memory/checkpointer.py against real Postgres -- no
mocking, same approach as test_documents.py/test_tags.py. Uses a trivial
throwaway LangGraph graph (not the real deterministic workflow, which needs
OPENAI_API_KEY/Neo4j) purely to exercise checkpointer persistence through
the public StateGraph API rather than reimplementing its lower-level
internals by hand.
"""
import os
import uuid

from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from memory.checkpointer import get_checkpointer


class _CounterState(TypedDict):
    count: int


def _increment(state: _CounterState) -> dict:
    return {"count": state["count"] + 1}


def _build_counter_graph(checkpointer):
    graph = StateGraph(_CounterState)
    graph.add_node("increment", _increment)
    graph.set_entry_point("increment")
    graph.add_edge("increment", END)
    return graph.compile(checkpointer=checkpointer)


async def test_setup_is_idempotent_across_repeated_opens():
    async with get_checkpointer(os.environ["DATABASE_URL"]):
        pass
    async with get_checkpointer(os.environ["DATABASE_URL"]):
        pass  # setup() ran twice against the same DB without error


async def test_thread_state_persists_across_separate_invocations():
    thread_id = f"test-{uuid.uuid4().hex[:10]}"
    config = {"configurable": {"thread_id": thread_id}}

    async with get_checkpointer(os.environ["DATABASE_URL"]) as checkpointer:
        compiled = _build_counter_graph(checkpointer)

        result1 = await compiled.ainvoke({"count": 0}, config)
        assert result1["count"] == 1

        # No `count` in this input -- if the checkpoint weren't actually
        # restored, `_increment` would KeyError on a missing "count" rather
        # than continuing from the persisted value.
        result2 = await compiled.ainvoke({}, config)
        assert result2["count"] == 2


async def test_different_threads_do_not_share_state():
    thread_a = f"test-{uuid.uuid4().hex[:10]}"
    thread_b = f"test-{uuid.uuid4().hex[:10]}"

    async with get_checkpointer(os.environ["DATABASE_URL"]) as checkpointer:
        compiled = _build_counter_graph(checkpointer)

        await compiled.ainvoke({"count": 0}, {"configurable": {"thread_id": thread_a}})
        await compiled.ainvoke({"count": 0}, {"configurable": {"thread_id": thread_a}})
        result_b = await compiled.ainvoke({"count": 0}, {"configurable": {"thread_id": thread_b}})

    assert result_b["count"] == 1
