"""Thread memory (issue #7, Ticket A): memory/budget.py's token budget,
memory/thread_memory.py's per-turn records and their rendered block, and
the deterministic workflow using them -- resolve_skus looking the previous
turn's SKUs up again, generate sending the block instead of the message
history, and the fact check accepting a remembered Part. No real model
call: the chat model is scripted and the graph lookups are faked.
"""
import uuid

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

import agents.workflows.deterministic as det
from memory.budget import count_tokens, fit_to_budget, truncate_to_tokens
from memory.thread_memory import (
    ANSWER_TOKEN_LIMIT,
    MAX_PARTS_PER_TURN,
    RECENT_TURNS_IN_FULL,
    THREAD_MEMORY_HEADER,
    THREAD_MEMORY_TOKEN_CAP,
    build_turn_record,
    remembered_parts,
    render_thread_memory,
)

# ---- memory/budget.py ----


def test_truncate_to_tokens_leaves_short_text_alone():
    assert truncate_to_tokens("a short answer", 50) == "a short answer"


def test_truncate_to_tokens_cuts_to_the_cap_with_an_ellipsis():
    text = "screw " * 500

    truncated = truncate_to_tokens(text, 40)

    assert count_tokens(truncated) <= 40
    assert truncated.endswith("…")


def test_fit_to_budget_drops_the_oldest_entries_first():
    entries = [f"entry {number} " + "word " * 40 for number in range(10)]

    block, tokens = fit_to_budget("Header", entries, 200)

    assert tokens == count_tokens(block) <= 200
    assert block.startswith("Header")
    assert "entry 9" in block
    assert "entry 0" not in block


def test_fit_to_budget_truncates_a_single_entry_over_the_cap():
    block, tokens = fit_to_budget("Header", ["word " * 1000], 100)

    assert tokens == count_tokens(block) <= 100


def test_fit_to_budget_with_nothing_to_say_costs_nothing():
    assert fit_to_budget("Header", [], 100) == ("", 0)


# ---- memory/thread_memory.py ----


def _part(number: int) -> dict:
    return {
        "sku": f"MSK4{number:04d}",
        "description": f"MIS HV CHAMFER FT 4.0 X {number} SCREW, CANNULATED, HEADED, TITANIUM ALLOY",
        "thread": "Full",
    }


def _long_turn_record(turn_number: int) -> dict:
    """A turn at every per-record cap: a long question and answer, more
    Parts and citations than are kept."""
    return build_turn_record(
        question=f"Turn {turn_number}: " + "which screw do I need for this bunion case " * 20,
        answer="The MIS screws come in many lengths and thread types. " * 100,
        parts=[_part(turn_number * 100 + index) for index in range(MAX_PARTS_PER_TURN + 5)],
        cited_chunk_ids=[f"{uuid.uuid4()}#{index}" for index in range(10)],
        family="MIS - Foot Recon System",
    )


def test_a_turn_record_is_cut_to_its_caps():
    record = _long_turn_record(1)

    assert count_tokens(record["answer"]) <= ANSWER_TOKEN_LIMIT
    assert len(record["parts"]) == MAX_PARTS_PER_TURN
    assert set(record["parts"][0]) == {"sku", "description"}


def test_the_block_never_exceeds_its_cap_over_a_twenty_turn_thread():
    records: list = []
    for turn_number in range(1, 21):
        records.append(_long_turn_record(turn_number))

        block, tokens = render_thread_memory(records)

        assert tokens == count_tokens(block)
        assert tokens <= THREAD_MEMORY_TOKEN_CAP
        # The latest turn always survives in full; the oldest go first.
        assert f"Rep asked: Turn {turn_number}:" in block
    assert "Turn 1 " not in block
    assert block.startswith(THREAD_MEMORY_HEADER)


def test_turns_past_the_recent_ones_collapse_to_skus_and_document_ids():
    document_id = str(uuid.uuid4())
    records = [
        build_turn_record(
            question="where is MSK40020?",
            answer="MSK40020 is in the MIS tray.",
            parts=[_part(20)],
            cited_chunk_ids=[f"{document_id}#3"],
            family="MIS - Foot Recon System",
        ),
        *(
            build_turn_record(
                question=f"follow-up {number}", answer="ok", parts=[], cited_chunk_ids=[], family=None
            )
            for number in range(RECENT_TURNS_IN_FULL)
        ),
    ]

    block, _ = render_thread_memory(records)

    assert f"Turn 1 (MIS - Foot Recon System): parts MSK40020; documents {document_id}" in block
    assert "where is MSK40020?" not in block
    assert "Rep asked: follow-up 0" in block


def test_a_thread_with_no_earlier_turns_has_no_block():
    assert render_thread_memory([]) == ("", 0)


def test_remembered_parts_prefers_the_parts_the_answer_names():
    named, unnamed, query_part = _part(1), _part(2), _part(3)

    assert remembered_parts(f"Use {named['sku']}.", [named, unnamed], [query_part]) == [named]
    assert remembered_parts("No SKU here.", [named, unnamed], [query_part]) == [query_part]


# ---- The deterministic workflow ----

_SCREW = {"sku": "MSK42020", "description": "MIS HV CHAMFER FT 4.0 X 20", "thread": "Full"}


class _FakePartLookup:
    def __init__(self, parts_by_term: dict[str, list[dict]] | None = None):
        self.parts_by_term = parts_by_term or {}
        self.terms: list[str] = []

    async def ainvoke(self, arguments: dict) -> list[dict]:
        self.terms.append(arguments["term"])
        return self.parts_by_term.get(arguments["term"], [])


class _ScriptedModel:
    def __init__(self, replies: list[str]):
        self._replies = list(replies)
        self.prompts: list[list] = []

    async def ainvoke(self, messages):
        self.prompts.append(messages)
        return AIMessage(content=self._replies.pop(0))


def _turn_inputs(thread_id: str, message: str) -> dict:
    return {
        "messages": [HumanMessage(content=message)],
        "query": message,
        "search_query": message,
        "clarification_rounds": 0,
        "clarification_reply": None,
        "correction_rounds": 0,
        "fact_check_issues": [],
        "memory_tokens": {},
        "user_id": "u1",
        "thread_id": thread_id,
    }


def _patch_retrieval(monkeypatch) -> _FakePartLookup:
    """Fakes everything up to rerank. The rep's own words match the screw
    only when they name its SKU; resolve_skus, aggregate_facts, generate,
    self_eval and finalize are the real ones."""

    async def fake_detect_intent(state):
        return {"resolved_family": None, "resolved_family_id": None, "detected_question_types": ["unclassified"]}

    async def fake_resolve_synonyms(state):
        return {"resolved_canonical_terms": [], "synonym_ambiguity": None}

    async def fake_hybrid_retrieve(state):
        return {"retrieved": [{"chunk_id": "doc-1#0", "document_id": "doc-1", "text": "a passage", "score": 1.0}]}

    async def fake_resolve_query_parts(state):
        return {"query_parts": [_SCREW] if _SCREW["sku"] in state["query"] else [], "turn_tray_ids": []}

    async def fake_rerank(state):
        return {"reranked": state["retrieved"]}

    lookup = _FakePartLookup({_SCREW["sku"]: [_SCREW]})
    monkeypatch.setattr(det, "detect_intent", fake_detect_intent)
    monkeypatch.setattr(det, "resolve_synonyms", fake_resolve_synonyms)
    monkeypatch.setattr(det, "hybrid_retrieve", fake_hybrid_retrieve)
    monkeypatch.setattr(det, "resolve_query_parts", fake_resolve_query_parts)
    monkeypatch.setattr(det, "rerank", fake_rerank)
    monkeypatch.setattr(det, "part_lookup", lookup)
    return lookup


async def test_a_follow_up_naming_no_sku_resolves_the_previous_turns_parts(monkeypatch):
    lookup = _patch_retrieval(monkeypatch)
    model = _ScriptedModel(["MSK42020 is a 4.0 x 20mm FT screw.", "- MSK42020: yes, it's FT, 20mm long."])
    monkeypatch.setattr(det, "get_chat_model", lambda: model)
    graph = det.build_graph(InMemorySaver())
    thread_id = f"test-{uuid.uuid4().hex[:10]}"
    config = {"configurable": {"thread_id": thread_id}}

    first = await graph.ainvoke(_turn_inputs(thread_id, "what is MSK42020?"), config)
    assert first["turn_memory"][0]["parts"] == [{"sku": "MSK42020", "description": _SCREW["description"]}]
    # Nothing earlier to remember on a thread's first turn.
    assert first["memory_tokens"] == {"thread_memory": 0}
    assert lookup.terms == []

    second = await graph.ainvoke(_turn_inputs(thread_id, "is it full thread?"), config)

    assert lookup.terms == ["MSK42020"]
    assert second["resolved_parts"] == [_SCREW]
    assert "MSK42020: MIS HV CHAMFER FT 4.0 X 20" in second["aggregated_facts"]
    assert second["fact_check_issues"] == []
    assert len(second["turn_memory"]) == 2
    # The prompt is the system prompt and one message, memory first and
    # the question last -- not the earlier turn's raw messages.
    system_message, question_message = model.prompts[1]
    assert system_message.content.startswith("You are OrthoMate")
    assert question_message.content.startswith(THREAD_MEMORY_HEADER)
    assert "Rep asked: what is MSK42020?" in question_message.content
    assert question_message.content.endswith("Question: is it full thread?")
    assert 0 < second["memory_tokens"]["thread_memory"] <= THREAD_MEMORY_TOKEN_CAP
    # messages stays the full transcript for the UI.
    assert [message.content for message in second["messages"]] == [
        "what is MSK42020?",
        "MSK42020 is a 4.0 x 20mm FT screw.",
        "is it full thread?",
        "- MSK42020: yes, it's FT, 20mm long.",
    ]


async def test_a_remembered_part_is_still_held_to_its_catalog_thread_type(monkeypatch):
    _patch_retrieval(monkeypatch)
    model = _ScriptedModel(
        ["MSK42020 is a 4.0 x 20mm FT screw.", "- MSK42020: it's PT.", "- MSK42020: it's FT."]
    )
    monkeypatch.setattr(det, "get_chat_model", lambda: model)
    graph = det.build_graph(InMemorySaver())
    thread_id = f"test-{uuid.uuid4().hex[:10]}"
    config = {"configurable": {"thread_id": thread_id}}

    await graph.ainvoke(_turn_inputs(thread_id, "what is MSK42020?"), config)
    second = await graph.ainvoke(_turn_inputs(thread_id, "what thread is it?"), config)

    assert second["correction_rounds"] == 1
    assert "MSK42020 is Full Thread (FT) in the catalog" in model.prompts[2][-1].content
    assert second["answer"] == "- MSK42020: it's FT."


async def test_resolve_skus_looks_up_the_previous_turns_skus_within_the_term_cap(monkeypatch):
    remembered = [_part(number) for number in range(1, 5)]
    lookup = _FakePartLookup({part["sku"]: [part] for part in remembered})
    monkeypatch.setattr(det, "part_lookup", lookup)
    state = {
        "query": "and the longer one?",
        "resolved_question_type": "system_contents",
        "query_parts": [remembered[0]],
        "resolved_canonical_terms": [],
        "turn_memory": [
            build_turn_record(question="q", answer="a", parts=remembered, cited_chunk_ids=[], family=None)
        ],
    }
    monkeypatch.setattr(det, "MAX_TERMS_PER_TURN", len(det._query_terms(state)) + 2)

    result = await det.resolve_skus(state)

    # The first remembered SKU was already a query Part; of the rest, only
    # two fit under the cap.
    assert lookup.terms == [remembered[1]["sku"], remembered[2]["sku"]]
    assert result["resolved_parts"] == remembered[:3]


async def test_resolve_skus_keeps_only_a_remembered_skus_exact_match(monkeypatch):
    lookalike = {"sku": "MSK42020-X", "description": "SIMILAR"}
    lookup = _FakePartLookup({_SCREW["sku"]: [lookalike, _SCREW]})
    monkeypatch.setattr(det, "part_lookup", lookup)

    result = await det.resolve_skus(
        {
            "query": "and that one?",
            "resolved_question_type": "unclassified",
            "query_parts": [],
            "turn_memory": [
                build_turn_record(question="q", answer="a", parts=[_SCREW], cited_chunk_ids=[], family=None)
            ],
        }
    )

    assert result["resolved_parts"] == [_SCREW]
