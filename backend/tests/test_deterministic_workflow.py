"""Exercises backend/agents/workflows/deterministic.py's graph wiring.

Node functions are monkeypatched with deterministic fakes to test control
flow (the clarification and correction loops, and that only the *final*
answer becomes permanent history) without needing a real LLM --
`graph.add_node(name, generate)` looks the function up from this module's
globals at `build_graph()` call time, so patching `deterministic.generate`
etc. before building the graph is enough. The correction-loop tests keep
the real generate and self_eval (a pure fact check, no model call) and
swap only the chat model for a scripted one, so the actual correction
logic runs. A real end-to-end run (needs a real LLM, real retrieval) is
covered at the API layer instead -- see test_chat_routes.py's
needs_openai_key tests.
"""
import uuid

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

import agents.workflows.deterministic as det


def _thread_config() -> tuple[str, dict]:
    thread_id = f"test-{uuid.uuid4().hex[:10]}"
    return thread_id, {"configurable": {"thread_id": thread_id}}


def _initial_state(thread_id: str, message: str) -> dict:
    return {
        "messages": [HumanMessage(content=message)],
        "query": message,
        "search_query": message,
        "clarification_rounds": 0,
        "clarification_reply": None,
        "correction_rounds": 0,
        "fact_check_issues": [],
        "user_id": "u1",
        "thread_id": thread_id,
    }


class _ScriptedModel:
    """Stands in for get_chat_model(): answers each call with the next
    scripted reply and records every prompt it was sent."""

    def __init__(self, replies: list[str]):
        self._replies = list(replies)
        self.prompts: list[list] = []

    async def ainvoke(self, messages):
        self.prompts.append(messages)
        return AIMessage(content=self._replies.pop(0))


# The one catalog Part every correction-loop test resolves -- a Full Thread
# screw, so an answer calling it PT fails the fact check.
_FULL_THREAD_PART = {"sku": "MSK42020", "description": "MIS HV CHAMFER FT 4.0 X 20", "thread": "Full"}
_WRONG_THREAD_ANSWER = "- MSK42020: 4.0 x 20mm PT screw"
_RIGHT_THREAD_ANSWER = "- MSK42020: 4.0 x 20mm FT screw"


def _patch_fixed_nodes(monkeypatch, *, generate=None, self_eval=None) -> None:
    """Patches every node except generate/self_eval with a fake that does
    the minimum to keep the pipeline moving -- shared across tests below
    since only generate/self_eval's behavior actually matters to what's
    being tested in each. generate/self_eval are only replaced when a test
    passes its own; otherwise the real ones run.
    """

    async def fake_detect_intent(state):
        return {"resolved_system": None, "resolved_system_id": None, "resolved_question_type": []}

    async def fake_resolve_synonyms(state):
        return {"resolved_canonical_terms": [], "synonym_ambiguity": None}

    async def fake_hybrid_retrieve(state):
        return {
            "retrieved": [
                {"chunk_id": "doc-1#0", "document_id": "doc-1", "text": "fake passage", "score": 1.0}
            ]
        }

    async def fake_rerank(state):
        return {"reranked": state["retrieved"]}

    async def fake_resolve_skus(state):
        return {"resolved_parts": [_FULL_THREAD_PART]}

    async def fake_aggregate_facts(state):
        return {"aggregated_facts": det._format_part(_FULL_THREAD_PART)}

    monkeypatch.setattr(det, "detect_intent", fake_detect_intent)
    monkeypatch.setattr(det, "resolve_synonyms", fake_resolve_synonyms)
    monkeypatch.setattr(det, "hybrid_retrieve", fake_hybrid_retrieve)
    monkeypatch.setattr(det, "rerank", fake_rerank)
    monkeypatch.setattr(det, "resolve_skus", fake_resolve_skus)
    monkeypatch.setattr(det, "aggregate_facts", fake_aggregate_facts)
    if generate is not None:
        monkeypatch.setattr(det, "generate", generate)
    if self_eval is not None:
        monkeypatch.setattr(det, "self_eval", self_eval)


def _patch_model(monkeypatch, replies: list[str]) -> _ScriptedModel:
    model = _ScriptedModel(replies)
    monkeypatch.setattr(det, "get_chat_model", lambda: model)
    return model


def test_build_graph_compiles_with_the_expected_nodes():
    graph = det.build_graph(InMemorySaver())
    nodes = set(graph.get_graph().nodes) - {"__start__", "__end__"}
    assert nodes == {
        "detect_intent",
        "resolve_synonyms",
        "hybrid_retrieve",
        "rerank",
        "resolve_skus",
        "aggregate_facts",
        "generate",
        "self_eval",
        "request_clarification",
        "finalize",
    }


async def test_a_clean_answer_ships_with_one_model_call(monkeypatch):
    _patch_fixed_nodes(monkeypatch)
    model = _patch_model(monkeypatch, [_RIGHT_THREAD_ANSWER])

    graph = det.build_graph(InMemorySaver())
    thread_id, config = _thread_config()
    result = await graph.ainvoke(_initial_state(thread_id, "which 4.0 screw?"), config)

    assert len(model.prompts) == 1
    assert result["fact_check_issues"] == []
    assert result["answer"] == _RIGHT_THREAD_ANSWER


async def test_failed_fact_check_gets_one_correction_and_only_the_corrected_answer_is_committed(
    monkeypatch,
):
    _patch_fixed_nodes(monkeypatch)
    model = _patch_model(monkeypatch, [_WRONG_THREAD_ANSWER, _RIGHT_THREAD_ANSWER])

    graph = det.build_graph(InMemorySaver())
    thread_id, config = _thread_config()
    result = await graph.ainvoke(_initial_state(thread_id, "which 4.0 screw?"), config)

    assert len(model.prompts) == 2
    # The correction pass shows the model its own failed draft, then the
    # specific problem the fact check found.
    failed_draft, correction_request = model.prompts[1][-2:]
    assert failed_draft.content == _WRONG_THREAD_ANSWER
    assert "MSK42020 is Full Thread (FT) in the catalog" in correction_request.content
    assert result["correction_rounds"] == 1
    assert result["fact_check_issues"] == []
    # Only the corrected answer becomes permanent history -- the rejected
    # draft must not also appear.
    ai_messages = [message for message in result["messages"] if isinstance(message, AIMessage)]
    assert [message.content for message in ai_messages] == [_RIGHT_THREAD_ANSWER]


async def test_correction_is_bounded_to_one_round_per_turn(monkeypatch):
    _patch_fixed_nodes(monkeypatch)
    # Always wrong -- without a bound this would loop forever.
    model = _patch_model(monkeypatch, [_WRONG_THREAD_ANSWER, _WRONG_THREAD_ANSWER, _WRONG_THREAD_ANSWER])

    graph = det.build_graph(InMemorySaver())
    thread_id, config = _thread_config()
    result = await graph.ainvoke(_initial_state(thread_id, "which 4.0 screw?"), config)

    # Initial draft + exactly one correction, then it ships anyway with the
    # problem still recorded.
    assert len(model.prompts) == 2
    assert result["answer"] == _WRONG_THREAD_ANSWER
    assert len(result["fact_check_issues"]) == 1


async def test_correction_rounds_do_not_leak_into_a_fresh_turn(monkeypatch):
    """correction_rounds/fact_check_issues have no reducer, so without the
    API layer explicitly resetting them per turn (see
    backend/api/routes/chat.py's inputs dict), a prior turn's spent
    correction would silently carry over via the checkpointer and suppress
    the next turn's own -- or a prior turn's leftover issues would turn the
    next turn's first draft into a "correction".
    """
    _patch_fixed_nodes(monkeypatch)
    model = _patch_model(monkeypatch, [_WRONG_THREAD_ANSWER] * 4)

    graph = det.build_graph(InMemorySaver())
    thread_id, config = _thread_config()

    await graph.ainvoke(_initial_state(thread_id, "first turn"), config)
    assert len(model.prompts) == 2

    # Same thread, fresh call with the per-turn fields reset -- exactly what
    # chat.py always does for a new turn.
    await graph.ainvoke(_initial_state(thread_id, "second turn"), config)
    assert len(model.prompts) == 4
    # The second turn's first draft was a plain draft, not a correction.
    assert model.prompts[2][-1].content.endswith("Question: second turn")


async def test_synonym_ambiguity_clarifies_before_retrieval_ever_runs(monkeypatch):
    """resolve_synonyms/_should_clarify_synonyms: a single extracted word
    matching more than one distinct canonical concept routes straight to
    request_clarification, without ever reaching hybrid_retrieve -- it
    fires before any retrieval or generation has happened at all.
    """
    hybrid_retrieve_calls = 0

    async def fake_detect_intent(state):
        return {"resolved_system": None, "resolved_system_id": None, "resolved_question_type": []}

    async def fake_hybrid_retrieve(state):
        nonlocal hybrid_retrieve_calls
        hybrid_retrieve_calls += 1
        return {"retrieved": []}

    async def fake_rerank(state):
        return {"reranked": []}

    async def fake_resolve_skus(state):
        return {"resolved_parts": []}

    async def fake_aggregate_facts(state):
        return {"aggregated_facts": ""}

    async def fake_generate(state):
        return {"answer": "final answer", "citations": []}

    async def fake_self_eval(state):
        return {"fact_check_issues": []}

    async def fake_resolve_synonyms(state):
        # First pass (no clarification_reply yet): "wire" is ambiguous.
        # Second pass (after resume): the rep's reply resolves it cleanly.
        if state.get("clarification_reply"):
            return {"resolved_canonical_terms": ["guidepin"], "synonym_ambiguity": None}
        return {
            "resolved_canonical_terms": [],
            "synonym_ambiguity": {"wire": ["guidepin", "drill bit"]},
        }

    monkeypatch.setattr(det, "detect_intent", fake_detect_intent)
    monkeypatch.setattr(det, "resolve_synonyms", fake_resolve_synonyms)
    monkeypatch.setattr(det, "hybrid_retrieve", fake_hybrid_retrieve)
    monkeypatch.setattr(det, "rerank", fake_rerank)
    monkeypatch.setattr(det, "resolve_skus", fake_resolve_skus)
    monkeypatch.setattr(det, "aggregate_facts", fake_aggregate_facts)
    monkeypatch.setattr(det, "generate", fake_generate)
    monkeypatch.setattr(det, "self_eval", fake_self_eval)

    graph = det.build_graph(InMemorySaver())
    thread_id, config = _thread_config()

    result = await graph.ainvoke(_initial_state(thread_id, "what wire do I need"), config)
    assert hybrid_retrieve_calls == 0  # paused before retrieval ever ran
    assert result["__interrupt__"][0].value == {
        "question": 'Just to confirm, by "wire" do you mean guidepin or drill bit?',
        "options": [],
    }

    result = await graph.ainvoke(Command(resume="guidepin"), config)
    assert hybrid_retrieve_calls == 1
    assert result["answer"] == "final answer"
    assert result["clarification_rounds"] == 1
