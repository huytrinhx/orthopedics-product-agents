"""Exercises backend/agents/workflows/deterministic.py's graph wiring.

Node functions are monkeypatched with deterministic fakes to test control
flow (the clarification loop, and that only the *accepted* answer becomes
permanent history) without needing a real LLM -- `graph.add_node(name,
generate)` looks the function up from this module's globals at
`build_graph()` call time, so patching `deterministic.generate` etc. before
building the graph is enough; no LangChain/OpenAI mocking needed. A real
end-to-end run (needs a real LLM, real retrieval) is covered at the API
layer instead -- see test_chat_routes.py's needs_openai_key tests.
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
        "user_id": "u1",
        "thread_id": thread_id,
    }


def _patch_fixed_nodes(monkeypatch, *, generate, self_eval) -> None:
    """Patches every node except generate/self_eval (the two each test
    varies) with a fake that does the minimum to keep the pipeline moving --
    shared across tests below since only generate/self_eval's behavior
    actually matters to what's being tested in each.
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
        return {"resolved_parts": []}

    async def fake_aggregate_facts(state):
        return {"aggregated_facts": ""}

    monkeypatch.setattr(det, "detect_intent", fake_detect_intent)
    monkeypatch.setattr(det, "resolve_synonyms", fake_resolve_synonyms)
    monkeypatch.setattr(det, "hybrid_retrieve", fake_hybrid_retrieve)
    monkeypatch.setattr(det, "rerank", fake_rerank)
    monkeypatch.setattr(det, "resolve_skus", fake_resolve_skus)
    monkeypatch.setattr(det, "aggregate_facts", fake_aggregate_facts)
    monkeypatch.setattr(det, "generate", generate)
    monkeypatch.setattr(det, "self_eval", self_eval)


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


async def test_clarification_pauses_then_resumes_and_only_commits_the_accepted_answer(
    monkeypatch,
):
    self_eval_calls = 0

    async def fake_generate(state):
        return {
            "answer": f"draft for '{state['search_query']}'"
            + (f" (rep said: {state['clarification_reply']})" if state.get("clarification_reply") else ""),
            "citations": ["doc-1#0"],
        }

    async def fake_self_eval(state):
        nonlocal self_eval_calls
        self_eval_calls += 1
        # Low the first time (forces a clarification round), high once the
        # rep has replied.
        score = 0.1 if self_eval_calls == 1 else 0.9
        return {"eval_scores": {"faithfulness": score, "relevance": score, "style": 1.0, "citation": 1.0}}

    async def fake_request_clarification(state):
        return {
            "search_query": f"{state['search_query']} more specific",
            "clarification_reply": "the bunion screw",
            "clarification_rounds": state.get("clarification_rounds", 0) + 1,
        }

    _patch_fixed_nodes(monkeypatch, generate=fake_generate, self_eval=fake_self_eval)
    monkeypatch.setattr(det, "request_clarification", fake_request_clarification)

    graph = det.build_graph(InMemorySaver())
    thread_id, config = _thread_config()
    result = await graph.ainvoke(_initial_state(thread_id, "hi"), config)

    assert self_eval_calls == 2  # scored low once, clarified, scored high on retry
    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    # Only the accepted (post-clarification) answer should be permanent
    # history -- the discarded first draft ("draft for 'hi'") must not also
    # appear.
    assert len(ai_messages) == 1
    assert ai_messages[0].content == "draft for 'hi more specific' (rep said: the bunion screw)"
    assert result["answer"] == ai_messages[0].content


async def test_clarification_is_bounded_to_one_round_per_turn(monkeypatch):
    self_eval_calls = 0
    clarification_calls = 0

    async def fake_generate(state):
        return {"answer": "always inadequate", "citations": []}

    async def fake_self_eval(state):
        nonlocal self_eval_calls
        self_eval_calls += 1
        # Always scores low -- without a bound this would loop forever.
        return {"eval_scores": {"faithfulness": 0.0, "relevance": 0.0, "style": 0.0, "citation": 0.0}}

    async def fake_request_clarification(state):
        nonlocal clarification_calls
        clarification_calls += 1
        return {
            "search_query": "still bad",
            "clarification_reply": "no further detail",
            "clarification_rounds": state.get("clarification_rounds", 0) + 1,
        }

    _patch_fixed_nodes(monkeypatch, generate=fake_generate, self_eval=fake_self_eval)
    monkeypatch.setattr(det, "request_clarification", fake_request_clarification)

    graph = det.build_graph(InMemorySaver())
    thread_id, config = _thread_config()
    result = await graph.ainvoke(_initial_state(thread_id, "hi"), config)

    # Initial attempt + exactly one clarification round, then it ships anyway.
    assert clarification_calls == 1
    assert self_eval_calls == 2
    assert result["answer"] == "always inadequate"


async def test_clarification_rounds_does_not_leak_into_a_fresh_turn(monkeypatch):
    """Regression test for the same class of bug retrieval_loop_count had:
    clarification_rounds/clarification_reply have no reducer, so without
    the API layer explicitly resetting them per turn (see
    backend/api/routes/chat.py's inputs dict), a prior turn's spent
    clarification round would silently carry over via the checkpointer and
    suppress the next turn's own clarification.
    """
    self_eval_calls = 0
    clarification_calls = 0

    async def fake_generate(state):
        return {"answer": "always inadequate", "citations": []}

    async def fake_self_eval(state):
        nonlocal self_eval_calls
        self_eval_calls += 1
        return {"eval_scores": {"faithfulness": 0.0, "relevance": 0.0, "style": 0.0, "citation": 0.0}}

    async def fake_request_clarification(state):
        nonlocal clarification_calls
        clarification_calls += 1
        return {
            "search_query": "still bad",
            "clarification_reply": "no further detail",
            "clarification_rounds": state.get("clarification_rounds", 0) + 1,
        }

    _patch_fixed_nodes(monkeypatch, generate=fake_generate, self_eval=fake_self_eval)
    monkeypatch.setattr(det, "request_clarification", fake_request_clarification)

    graph = det.build_graph(InMemorySaver())
    thread_id, config = _thread_config()

    await graph.ainvoke(_initial_state(thread_id, "first turn"), config)
    assert clarification_calls == 1

    clarification_calls = 0
    # Same thread, but a fresh call with clarification_rounds/reply
    # explicitly reset -- exactly what chat.py always does for a new turn.
    await graph.ainvoke(_initial_state(thread_id, "second turn"), config)
    assert clarification_calls == 1


async def test_request_clarification_interrupts_and_resume_continues_the_turn(monkeypatch):
    """Exercises the real interrupt()/resume mechanic (not monkeypatched
    away) -- request_clarification itself is real here, only the LLM calls
    inside it and the rest of the pipeline are faked, mirroring how
    test_chat_routes.py's ambiguous-query test covers detect_intent's own
    interrupt().
    """
    self_eval_calls = 0

    async def fake_generate(state):
        reply = state.get("clarification_reply")
        return {"answer": f"answer{f' informed by: {reply}' if reply else ''}", "citations": []}

    async def fake_self_eval(state):
        nonlocal self_eval_calls
        self_eval_calls += 1
        score = 0.1 if self_eval_calls == 1 else 0.9
        return {"eval_scores": {"faithfulness": score, "relevance": score, "style": 1.0, "citation": 1.0}}

    async def fake_generate_clarifying_question(query, draft_answer):
        return "Which procedure is this for?"

    _patch_fixed_nodes(monkeypatch, generate=fake_generate, self_eval=fake_self_eval)
    monkeypatch.setattr(det, "_generate_clarifying_question", fake_generate_clarifying_question)

    graph = det.build_graph(InMemorySaver())
    thread_id, config = _thread_config()

    result = await graph.ainvoke(_initial_state(thread_id, "hi"), config)
    state = await graph.aget_state(config)
    assert state.next == ("request_clarification",)
    assert result["__interrupt__"][0].value == {"question": "Which procedure is this for?", "options": []}

    result = await graph.ainvoke(Command(resume="the Akin osteotomy"), config)
    assert result["answer"] == "answer informed by: the Akin osteotomy"
    assert result["clarification_rounds"] == 1


async def test_synonym_ambiguity_clarifies_before_retrieval_ever_runs(monkeypatch):
    """resolve_synonyms/_should_clarify_synonyms: a single extracted word
    matching more than one distinct canonical concept routes straight to
    request_clarification, without ever reaching hybrid_retrieve -- unlike
    self_eval's clarification path, this one fires before any retrieval or
    generation has happened at all.
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
        return {"eval_scores": {"faithfulness": 1.0, "relevance": 1.0, "style": 1.0, "citation": 1.0}}

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
