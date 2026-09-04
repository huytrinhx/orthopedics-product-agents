"""Exercises backend/agents/workflows/deterministic.py's graph wiring.

Node functions are monkeypatched with deterministic fakes to test control
flow (the retry loop, and that only the *accepted* answer becomes permanent
history) without needing a real LLM -- `graph.add_node(name, generate)`
looks the function up from this module's globals at `build_graph()` call
time, so patching `deterministic.generate` etc. before building the graph
is enough; no LangChain/OpenAI mocking needed. A real end-to-end run
(needs a real LLM, real retrieval) is covered at the API layer instead --
see test_chat_routes.py's needs_openai_key test.
"""
import uuid

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

import agents.workflows.deterministic as det


def _thread_config() -> tuple[str, dict]:
    thread_id = f"test-{uuid.uuid4().hex[:10]}"
    return thread_id, {"configurable": {"thread_id": thread_id}}


def _initial_state(thread_id: str, message: str) -> dict:
    return {
        "messages": [HumanMessage(content=message)],
        "query": message,
        "search_query": message,
        "retrieval_loop_count": 0,
        "user_id": "u1",
        "thread_id": thread_id,
    }


def test_build_graph_compiles_with_the_expected_nodes():
    graph = det.build_graph(InMemorySaver())
    nodes = set(graph.get_graph().nodes) - {"__start__", "__end__"}
    assert nodes == {
        "detect_intent",
        "resolve_synonyms",
        "resolve_query_entities",
        "hybrid_retrieve",
        "rerank",
        "resolve_skus",
        "aggregate_facts",
        "generate",
        "self_eval",
        "reformulate",
        "finalize",
    }


async def test_retry_loop_reformulates_and_only_commits_the_accepted_answer(monkeypatch):
    self_eval_calls = 0

    async def fake_detect_intent(state):
        return {"resolved_system": None, "resolved_system_id": None, "resolved_question_type": []}

    async def fake_resolve_synonyms(state):
        return {"resolved_synonyms": []}

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

    async def fake_generate(state):
        return {"answer": f"draft for '{state['search_query']}'", "citations": ["doc-1#0"]}

    async def fake_self_eval(state):
        nonlocal self_eval_calls
        self_eval_calls += 1
        # Low the first time (forces a retry), high once reformulate has run.
        score = 0.1 if self_eval_calls == 1 else 0.9
        return {"eval_scores": {"faithfulness": score, "relevance": score, "style": 1.0, "citation": 1.0}}

    async def fake_reformulate(state):
        return {
            "search_query": "reformulated query",
            "retrieval_loop_count": state.get("retrieval_loop_count", 0) + 1,
        }

    monkeypatch.setattr(det, "detect_intent", fake_detect_intent)
    monkeypatch.setattr(det, "resolve_synonyms", fake_resolve_synonyms)
    monkeypatch.setattr(det, "hybrid_retrieve", fake_hybrid_retrieve)
    monkeypatch.setattr(det, "rerank", fake_rerank)
    monkeypatch.setattr(det, "resolve_skus", fake_resolve_skus)
    monkeypatch.setattr(det, "aggregate_facts", fake_aggregate_facts)
    monkeypatch.setattr(det, "generate", fake_generate)
    monkeypatch.setattr(det, "self_eval", fake_self_eval)
    monkeypatch.setattr(det, "reformulate", fake_reformulate)

    graph = det.build_graph(InMemorySaver())
    thread_id, config = _thread_config()
    result = await graph.ainvoke(_initial_state(thread_id, "hi"), config)

    assert self_eval_calls == 2  # scored low once, reformulated, scored high on retry
    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    # Only the accepted (retried) answer should be permanent history -- the
    # discarded first draft ("draft for 'hi'") must not also appear.
    assert len(ai_messages) == 1
    assert ai_messages[0].content == "draft for 'reformulated query'"
    assert result["answer"] == "draft for 'reformulated query'"


async def test_retry_loop_is_bounded_by_max_retrieval_loops(monkeypatch):
    self_eval_calls = 0

    async def fake_detect_intent(state):
        return {"resolved_system": None, "resolved_system_id": None, "resolved_question_type": []}

    async def fake_resolve_synonyms(state):
        return {"resolved_synonyms": []}

    async def fake_hybrid_retrieve(state):
        return {"retrieved": []}

    async def fake_rerank(state):
        return {"reranked": []}

    async def fake_resolve_skus(state):
        return {"resolved_parts": []}

    async def fake_aggregate_facts(state):
        return {"aggregated_facts": ""}

    async def fake_generate(state):
        return {"answer": "always inadequate", "citations": []}

    async def fake_self_eval(state):
        nonlocal self_eval_calls
        self_eval_calls += 1
        # Always scores low -- without a bound this would loop forever.
        return {"eval_scores": {"faithfulness": 0.0, "relevance": 0.0, "style": 0.0, "citation": 0.0}}

    async def fake_reformulate(state):
        return {
            "search_query": "still bad",
            "retrieval_loop_count": state.get("retrieval_loop_count", 0) + 1,
        }

    monkeypatch.setattr(det, "detect_intent", fake_detect_intent)
    monkeypatch.setattr(det, "resolve_synonyms", fake_resolve_synonyms)
    monkeypatch.setattr(det, "hybrid_retrieve", fake_hybrid_retrieve)
    monkeypatch.setattr(det, "rerank", fake_rerank)
    monkeypatch.setattr(det, "resolve_skus", fake_resolve_skus)
    monkeypatch.setattr(det, "aggregate_facts", fake_aggregate_facts)
    monkeypatch.setattr(det, "generate", fake_generate)
    monkeypatch.setattr(det, "self_eval", fake_self_eval)
    monkeypatch.setattr(det, "reformulate", fake_reformulate)

    graph = det.build_graph(InMemorySaver())
    thread_id, config = _thread_config()
    result = await graph.ainvoke(_initial_state(thread_id, "hi"), config)

    # Initial attempt + MAX_RETRIEVAL_LOOPS retries, then it ships anyway.
    assert self_eval_calls == det.MAX_RETRIEVAL_LOOPS + 1
    assert result["answer"] == "always inadequate"


async def test_retrieval_loop_count_does_not_leak_into_a_fresh_turn(monkeypatch):
    """Regression test for a real bug: retrieval_loop_count has no reducer,
    so without the API layer explicitly resetting it per turn (see
    backend/api/routes/chat.py's inputs dict), a prior turn's exhausted
    retry budget would silently carry over via the checkpointer and shrink
    the next turn's retry allowance.
    """
    self_eval_calls = 0

    async def fake_detect_intent(state):
        return {"resolved_system": None, "resolved_system_id": None, "resolved_question_type": []}

    async def fake_resolve_synonyms(state):
        return {"resolved_synonyms": []}

    async def fake_hybrid_retrieve(state):
        return {"retrieved": []}

    async def fake_rerank(state):
        return {"reranked": []}

    async def fake_resolve_skus(state):
        return {"resolved_parts": []}

    async def fake_aggregate_facts(state):
        return {"aggregated_facts": ""}

    async def fake_generate(state):
        return {"answer": "always inadequate", "citations": []}

    async def fake_self_eval(state):
        nonlocal self_eval_calls
        self_eval_calls += 1
        return {"eval_scores": {"faithfulness": 0.0, "relevance": 0.0, "style": 0.0, "citation": 0.0}}

    async def fake_reformulate(state):
        return {
            "search_query": "still bad",
            "retrieval_loop_count": state.get("retrieval_loop_count", 0) + 1,
        }

    monkeypatch.setattr(det, "detect_intent", fake_detect_intent)
    monkeypatch.setattr(det, "resolve_synonyms", fake_resolve_synonyms)
    monkeypatch.setattr(det, "hybrid_retrieve", fake_hybrid_retrieve)
    monkeypatch.setattr(det, "rerank", fake_rerank)
    monkeypatch.setattr(det, "resolve_skus", fake_resolve_skus)
    monkeypatch.setattr(det, "aggregate_facts", fake_aggregate_facts)
    monkeypatch.setattr(det, "generate", fake_generate)
    monkeypatch.setattr(det, "self_eval", fake_self_eval)
    monkeypatch.setattr(det, "reformulate", fake_reformulate)

    graph = det.build_graph(InMemorySaver())
    thread_id, config = _thread_config()

    await graph.ainvoke(_initial_state(thread_id, "first turn"), config)
    assert self_eval_calls == det.MAX_RETRIEVAL_LOOPS + 1

    self_eval_calls = 0
    # Same thread, but a fresh call with retrieval_loop_count explicitly
    # reset to 0 -- exactly what chat.py always does for a new turn.
    await graph.ainvoke(_initial_state(thread_id, "second turn"), config)
    assert self_eval_calls == det.MAX_RETRIEVAL_LOOPS + 1
