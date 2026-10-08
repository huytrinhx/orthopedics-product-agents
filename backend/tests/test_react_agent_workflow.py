"""Exercises backend/agents/workflows/react_agent.py's fact-check/correction
loop with the real generate, call_tools, and self_eval nodes -- only the
chat model (scripted replies, including tool calls), the tools themselves,
and detect_intent are faked, so no LLM, Neo4j, or Postgres is needed.
"""
import uuid

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

import agents.workflows.react_agent as react

_FULL_THREAD_PART = {"sku": "MSK42020", "description": "MIS HV CHAMFER FT 4.0 X 20", "thread": "Full"}
_LOOKUP_CALL = AIMessage(
    content="", tool_calls=[{"name": "part_lookup", "args": {"term": "4.0 chamfer"}, "id": "call-1"}]
)


class _ScriptedModel:
    def __init__(self, replies: list[AIMessage]):
        self._replies = list(replies)
        self.prompts: list[list] = []

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        self.prompts.append(messages)
        return self._replies.pop(0)


class _FakePartLookup:
    name = "part_lookup"

    async def ainvoke(self, args):
        return [_FULL_THREAD_PART]


def _patch(monkeypatch, replies: list[AIMessage]) -> _ScriptedModel:
    async def fake_detect_intent(state):
        return {"resolved_system": None, "resolved_system_id": None, "resolved_question_type": []}

    model = _ScriptedModel(replies)
    monkeypatch.setattr(react, "detect_intent", fake_detect_intent)
    monkeypatch.setattr(react, "get_chat_model", lambda: model)
    monkeypatch.setattr(react, "_TOOLS_BY_NAME", {"part_lookup": _FakePartLookup()})
    return model


async def _run(query: str) -> dict:
    graph = react.build_graph(InMemorySaver())
    thread_id = f"test-{uuid.uuid4().hex[:10]}"
    return await graph.ainvoke(
        {
            "messages": [HumanMessage(content=query)],
            "query": query,
            "search_query": query,
            "correction_rounds": 0,
            "fact_check_issues": [],
            "user_id": "u1",
            "thread_id": thread_id,
        },
        {"configurable": {"thread_id": thread_id}},
    )


def test_document_lookup_is_in_the_agent_toolbox():
    assert "document_lookup" in {tool.name for tool in react._TOOLS}


async def test_a_clean_answer_ships_without_a_correction(monkeypatch):
    model = _patch(monkeypatch, [_LOOKUP_CALL, AIMessage(content="- MSK42020: 4.0 x 20mm FT screw")])

    result = await _run("which 4.0 chamfer screw?")

    assert len(model.prompts) == 2
    assert result["fact_check_issues"] == []
    assert result["answer"] == "- MSK42020: 4.0 x 20mm FT screw"


async def test_failed_fact_check_gets_one_correction_against_the_tool_results(monkeypatch):
    model = _patch(
        monkeypatch,
        [
            _LOOKUP_CALL,
            AIMessage(content="- MSK42020: 4.0 x 20mm PT screw"),
            AIMessage(content="- MSK42020: 4.0 x 20mm FT screw"),
        ],
    )

    result = await _run("which 4.0 chamfer screw?")

    assert len(model.prompts) == 3
    correction_request = model.prompts[2][-1]
    assert isinstance(correction_request, HumanMessage)
    assert "MSK42020 is Full Thread (FT) in the catalog" in correction_request.content
    assert result["correction_rounds"] == 1
    ai_messages = [message for message in result["messages"] if isinstance(message, AIMessage)]
    assert [message.content for message in ai_messages] == ["- MSK42020: 4.0 x 20mm FT screw"]


async def test_correction_is_bounded_to_one_round_per_turn(monkeypatch):
    # Separate message objects, not one reused: add_messages assigns each an
    # id on first append, and a reused object would replace itself in place.
    model = _patch(
        monkeypatch,
        [_LOOKUP_CALL, *(AIMessage(content="- MSK42020: 4.0 x 20mm PT screw") for _ in range(3))],
    )

    result = await _run("which 4.0 chamfer screw?")

    assert len(model.prompts) == 3
    assert result["answer"] == "- MSK42020: 4.0 x 20mm PT screw"
    assert len(result["fact_check_issues"]) == 1
