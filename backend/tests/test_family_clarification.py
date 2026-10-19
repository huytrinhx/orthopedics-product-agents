"""Product Family intent and clarification (issue #18), through the real
deterministic graph: detect_intent resolves a family (by display name) or
interrupts with at most MAX_FAMILY_CANDIDATES candidates -- every family
when there are none -- a clicked option or free text resolves back to a
real family on resume, and hybrid_retrieve hard-filters by it.

The classifier and the family list are faked (no LLM, no database), and so
is everything past hybrid_retrieve; hybrid_retrieve itself runs for real
against a recording vector_search, so what reaches retrieval is asserted
directly. The SQL side of the filter is test_vector_store.py's.

Multi-turn cases (issue #22) ask twice on the same thread: a follow-up
naming no family stays on the previous turn's family.
"""
import uuid

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

import agents.workflows.deterministic as det
from agents.workflows import react_agent
from catalog.repository import ProductFamilyRecord

# Display names differ from catalog names on purpose: the classifier and the
# rep only ever see display names, and both must resolve to the real family.
_HYBRID = ProductFamilyRecord(uuid.uuid4(), "REFLEX HYBRID", "Reflex Hybrid")
_FOOT_RECON = ProductFamilyRecord(uuid.uuid4(), "MIS - Foot Recon System", "MIS Foot Recon")
_POWER = ProductFamilyRecord(uuid.uuid4(), "MIS - Power System", "MIS Power")
_ANKLE = ProductFamilyRecord(uuid.uuid4(), "Ankle Fracture Plating Sytem", "Ankle Fracture Plating System")
_SCREWS = ProductFamilyRecord(uuid.uuid4(), "Cannulated Screws", "Cannulated Screws")
_TIBIA = ProductFamilyRecord(uuid.uuid4(), "Distal Tibia Plating System", "Distal Tibia Plating")
_FAMILIES = [_HYBRID, _FOOT_RECON, _POWER, _ANKLE, _SCREWS, _TIBIA]


class _RecordingVectorSearch:
    def __init__(self):
        self.calls: list[dict] = []

    async def ainvoke(self, arguments: dict) -> list[dict]:
        self.calls.append(arguments)
        return []


class _Harness:
    def __init__(
        self,
        monkeypatch,
        classification: det._IntentClassification | dict[str, det._IntentClassification],
        families=_FAMILIES,
    ):
        """A dict of classifications is keyed by message, for multi-turn
        tests (issue #22) where each turn classifies differently -- keyed
        rather than ordered because resuming a family clarification replays
        the classifier call. Every call's previous-turn hints are recorded.
        """
        self.classifier_family_names: list[list[str]] = []
        self.classifier_hints: list[tuple[str | None, str | None]] = []
        self.classifier_family_descriptions: list[dict[str, str] | None] = []
        self.search = _RecordingVectorSearch()

        async def fake_list_families_with_trays():
            return list(families)

        async def fake_load_family_descriptions(described_families):
            return {family.display_name: f"About {family.display_name}." for family in described_families}

        async def fake_classify_intent(
            query, family_names, previous_family=None, previous_message=None, family_descriptions=None
        ):
            self.classifier_family_names.append(family_names)
            self.classifier_hints.append((previous_family, previous_message))
            self.classifier_family_descriptions.append(family_descriptions)
            turn_classification = classification[query] if isinstance(classification, dict) else classification
            return turn_classification.model_copy(deep=True)

        async def fake_list_hierarchy():
            return {}

        async def fake_resolve_synonyms(state):
            return {"resolved_canonical_terms": [], "synonym_ambiguity": None}

        async def fake_resolve_query_parts(state):
            return {"query_parts": [], "turn_tray_ids": []}

        async def fake_resolve_skus(state):
            return {"resolved_parts": []}

        async def fake_aggregate_facts(state):
            return {"aggregated_facts": ""}

        async def fake_generate(state):
            return {"answer": "final answer", "citations": []}

        async def fake_self_eval(state):
            return {"fact_check_issues": []}

        monkeypatch.setattr(det, "list_families_with_trays", fake_list_families_with_trays)
        monkeypatch.setattr(det, "load_family_descriptions", fake_load_family_descriptions)
        monkeypatch.setattr(det, "_classify_intent", fake_classify_intent)
        monkeypatch.setattr(det, "list_hierarchy", fake_list_hierarchy)
        monkeypatch.setattr(det, "vector_search", self.search)
        monkeypatch.setattr(det, "resolve_synonyms", fake_resolve_synonyms)
        monkeypatch.setattr(det, "resolve_query_parts", fake_resolve_query_parts)
        monkeypatch.setattr(det, "resolve_skus", fake_resolve_skus)
        monkeypatch.setattr(det, "aggregate_facts", fake_aggregate_facts)
        monkeypatch.setattr(det, "generate", fake_generate)
        monkeypatch.setattr(det, "self_eval", fake_self_eval)

        self.graph = det.build_graph(InMemorySaver())
        self.config = {"configurable": {"thread_id": f"test-{uuid.uuid4().hex[:10]}"}}

    async def ask(self, message: str = "What screws are in the set?") -> dict:
        return await self.graph.ainvoke(
            {
                "messages": [HumanMessage(content=message)],
                "query": message,
                "search_query": message,
                "clarification_rounds": 0,
                "clarification_reply": None,
                "correction_rounds": 0,
                "fact_check_issues": [],
                "user_id": "u1",
                "thread_id": self.config["configurable"]["thread_id"],
            },
            self.config,
        )

    async def resume(self, human_input: str) -> dict:
        return await self.graph.ainvoke(Command(resume=human_input), self.config)


def _unsure(*candidate_families: str) -> det._IntentClassification:
    return det._IntentClassification(product_family=None, candidate_families=list(candidate_families))


async def test_a_confident_classification_never_interrupts_and_scopes_retrieval(monkeypatch):
    harness = _Harness(monkeypatch, det._IntentClassification(product_family="Reflex Hybrid"))

    result = await harness.ask("Torque for the REFLEX HYBRID locking screw?")

    assert "__interrupt__" not in result
    assert result["answer"] == "final answer"
    assert result["resolved_family"] == "Reflex Hybrid"
    assert result["resolved_family_id"] == str(_HYBRID.id)
    # The classifier chooses among display names, not catalog names, each
    # with its generated description (issue #30).
    assert harness.classifier_family_names == [[family.display_name for family in _FAMILIES]]
    assert harness.classifier_family_descriptions == [
        {family.display_name: f"About {family.display_name}." for family in _FAMILIES}
    ]
    assert [call["product_family_id"] for call in harness.search.calls] == [str(_HYBRID.id)]


async def test_a_confident_catalog_name_in_any_case_still_resolves(monkeypatch):
    harness = _Harness(monkeypatch, det._IntentClassification(product_family="mis - foot recon system"))

    result = await harness.ask()

    assert "__interrupt__" not in result
    assert result["resolved_family_id"] == str(_FOOT_RECON.id)
    assert result["resolved_family"] == "MIS Foot Recon"


async def test_ambiguity_offers_at_most_four_real_candidates_as_display_names(monkeypatch):
    harness = _Harness(
        monkeypatch,
        _unsure(
            "MIS Foot Recon",
            "Not A Real Family",
            "MIS Power",
            "mis foot recon",  # a duplicate, differently cased
            "Cannulated Screws",
            "Ankle Fracture Plating Sytem",  # catalog name -> shown as its display name
            "Reflex Hybrid",
        ),
    )

    result = await harness.ask()

    assert harness.search.calls == []  # paused before retrieval
    assert result["__interrupt__"][0].value == {
        "question": "Which product family is this about?",
        "options": ["MIS Foot Recon", "MIS Power", "Cannulated Screws", "Ankle Fracture Plating System"],
    }


async def test_no_candidates_offers_every_family(monkeypatch):
    harness = _Harness(monkeypatch, _unsure())

    result = await harness.ask()

    assert result["__interrupt__"][0].value["options"] == [family.display_name for family in _FAMILIES]


async def test_only_unreal_candidates_offers_every_family(monkeypatch):
    harness = _Harness(monkeypatch, _unsure("Made Up Family", "null"))

    result = await harness.ask()

    assert result["__interrupt__"][0].value["options"] == [family.display_name for family in _FAMILIES]


async def test_a_literal_null_family_is_treated_as_unsure(monkeypatch):
    harness = _Harness(
        monkeypatch, det._IntentClassification(product_family="null", candidate_families=["MIS Power"])
    )

    result = await harness.ask()

    assert result["__interrupt__"][0].value["options"] == ["MIS Power"]


async def test_resume_with_a_clicked_option_resolves_the_family_and_scopes_retrieval(monkeypatch):
    harness = _Harness(monkeypatch, _unsure("MIS Foot Recon", "MIS Power"))
    await harness.ask()

    result = await harness.resume("MIS Power")

    assert result["answer"] == "final answer"
    assert result["resolved_family"] == "MIS Power"
    assert result["resolved_family_id"] == str(_POWER.id)
    assert [call["product_family_id"] for call in harness.search.calls] == [str(_POWER.id)]


async def test_resume_with_free_text_resolves_any_real_family_not_just_the_offered_ones(monkeypatch):
    harness = _Harness(monkeypatch, _unsure("MIS Foot Recon", "MIS Power"))
    await harness.ask()

    result = await harness.resume("  distal tibia plating ")

    assert result["resolved_family_id"] == str(_TIBIA.id)
    assert result["resolved_family"] == "Distal Tibia Plating"


async def test_free_text_naming_no_family_runs_unscoped_without_asking_again(monkeypatch):
    harness = _Harness(monkeypatch, _unsure("MIS Foot Recon"))
    await harness.ask()

    result = await harness.resume("not sure, whichever")

    # At most one family clarification per turn.
    assert "__interrupt__" not in result
    assert result["answer"] == "final answer"
    assert result["resolved_family"] is None
    assert result["resolved_family_id"] is None
    assert [call["product_family_id"] for call in harness.search.calls] == [None]


async def test_no_families_mirrored_yet_skips_the_clarification(monkeypatch):
    harness = _Harness(monkeypatch, _unsure(), families=[])

    result = await harness.ask()

    assert "__interrupt__" not in result
    assert result["resolved_family_id"] is None
    assert [call["product_family_id"] for call in harness.search.calls] == [None]


# Issue #22: carrying the thread's family into a follow-up turn.
_FIRST_QUESTION = "What screws are in the MIS Foot Recon set?"
_FOLLOW_UP = "What about the 3.5 PT?"
_ON_FOOT_RECON = det._IntentClassification(product_family="MIS Foot Recon", question_type=["system_contents"])
_NAMES_NO_FAMILY = det._IntentClassification(
    product_family=None, candidate_families=["MIS Power"], question_type=["product_characteristics"]
)


async def test_a_follow_up_naming_no_family_stays_on_the_previous_family_without_asking(monkeypatch):
    harness = _Harness(monkeypatch, {_FIRST_QUESTION: _ON_FOOT_RECON, _FOLLOW_UP: _NAMES_NO_FAMILY})
    await harness.ask(_FIRST_QUESTION)

    result = await harness.ask(_FOLLOW_UP)

    assert "__interrupt__" not in result
    assert result["answer"] == "final answer"
    assert result["resolved_family"] == "MIS Foot Recon"
    assert result["resolved_family_id"] == str(_FOOT_RECON.id)
    # The question type is still classified fresh.
    assert result["resolved_question_type"] == "product_characteristics"
    assert [call["product_family_id"] for call in harness.search.calls] == [str(_FOOT_RECON.id)] * 2
    # The previous family and the previous user message go to the classifier as hints.
    assert harness.classifier_hints == [(None, None), ("MIS Foot Recon", _FIRST_QUESTION)]


async def test_a_follow_up_clearly_naming_another_family_switches_to_it(monkeypatch):
    switch = "And the REFLEX HYBRID plate?"
    harness = _Harness(
        monkeypatch,
        {
            _FIRST_QUESTION: _ON_FOOT_RECON,
            switch: det._IntentClassification(product_family="Reflex Hybrid", mentions_family=True),
        },
    )
    await harness.ask(_FIRST_QUESTION)

    result = await harness.ask(switch)

    assert "__interrupt__" not in result
    assert result["resolved_family_id"] == str(_HYBRID.id)
    assert [call["product_family_id"] for call in harness.search.calls] == [
        str(_FOOT_RECON.id),
        str(_HYBRID.id),
    ]


async def test_a_follow_up_naming_another_family_ambiguously_asks_offering_the_previous_one_first(monkeypatch):
    ambiguous = "What about the plating one?"
    harness = _Harness(
        monkeypatch,
        {
            _FIRST_QUESTION: _ON_FOOT_RECON,
            ambiguous: det._IntentClassification(
                product_family=None,
                mentions_family=True,
                candidate_families=["Ankle Fracture Plating System", "Distal Tibia Plating", "MIS Foot Recon", "MIS Power"],
            ),
        },
    )
    await harness.ask(_FIRST_QUESTION)

    paused = await harness.ask(ambiguous)

    # The previous family leads, deduped, and the list stays capped at four.
    assert paused["__interrupt__"][0].value == {
        "question": "Which product family is this about?",
        "options": ["MIS Foot Recon", "Ankle Fracture Plating System", "Distal Tibia Plating", "MIS Power"],
    }
    result = await harness.resume("Distal Tibia Plating")
    assert result["resolved_family_id"] == str(_TIBIA.id)


async def test_an_ambiguous_follow_up_with_no_candidates_offers_every_family_previous_first(monkeypatch):
    ambiguous = "What about the other system?"
    harness = _Harness(
        monkeypatch,
        {_FIRST_QUESTION: _ON_FOOT_RECON, ambiguous: det._IntentClassification(mentions_family=True)},
    )
    await harness.ask(_FIRST_QUESTION)

    paused = await harness.ask(ambiguous)

    assert paused["__interrupt__"][0].value["options"] == [
        "MIS Foot Recon",
        *(family.display_name for family in _FAMILIES if family is not _FOOT_RECON),
    ]


async def test_a_first_turn_naming_no_family_still_asks(monkeypatch):
    # mentions_family false means nothing with no previous family to stay on.
    harness = _Harness(monkeypatch, _NAMES_NO_FAMILY)

    result = await harness.ask(_FOLLOW_UP)

    assert result["__interrupt__"][0].value == {
        "question": "Which product family is this about?",
        "options": ["MIS Power"],
    }


async def test_a_family_chosen_through_clarification_carries_into_the_next_turn(monkeypatch):
    first_question = "What screws are in the set?"
    harness = _Harness(monkeypatch, {first_question: _unsure("MIS Power"), _FOLLOW_UP: _NAMES_NO_FAMILY})
    await harness.ask(first_question)
    await harness.resume("MIS Power")

    result = await harness.ask(_FOLLOW_UP)

    assert "__interrupt__" not in result
    assert result["resolved_family_id"] == str(_POWER.id)
    # Resuming replays the first turn's classifier call with the same (empty) hints.
    assert harness.classifier_hints == [(None, None), (None, None), ("MIS Power", first_question)]


async def test_a_follow_up_to_an_unscoped_turn_asks_again(monkeypatch):
    first_question = "What screws are in the set?"
    harness = _Harness(monkeypatch, {first_question: _unsure("MIS Power"), _FOLLOW_UP: _NAMES_NO_FAMILY})
    await harness.ask(first_question)
    await harness.resume("not sure, whichever")

    result = await harness.ask(_FOLLOW_UP)

    # Nothing to carry: the rep declined to pick a family last turn.
    assert result["__interrupt__"][0].value["question"] == "Which product family is this about?"
    assert harness.classifier_hints[-1] == (None, first_question)


def test_the_previous_family_is_looked_up_by_id_under_its_current_display_name():
    renamed = ProductFamilyRecord(_POWER.id, _POWER.name, "MIS Power (Renamed)")
    state = {"resolved_family_id": str(_POWER.id)}

    assert det._previous_family(state, [_HYBRID, renamed]) is renamed
    assert det._previous_family(state, [_HYBRID]) is None  # no longer mirrored
    assert det._previous_family({}, _FAMILIES) is None


def test_the_previous_user_message_is_the_last_one_before_this_turns():
    first_turn_only = {"messages": [HumanMessage(content="first")]}
    two_turns = {
        "messages": [
            HumanMessage(content="first"),
            AIMessage(content="answer"),
            HumanMessage(content="second"),
            AIMessage(content="answer"),
            HumanMessage(content="third"),
        ]
    }

    assert det._previous_user_message(first_turn_only) is None
    assert det._previous_user_message(two_turns) == "second"


def test_no_workflow_reads_the_legacy_system_tag_list():
    for module in (det, react_agent):
        assert not hasattr(module, "list_systems")
    assert "resolved_system" not in det.DeterministicState.__annotations__
    assert "resolved_system" not in react_agent.ReactAgentState.__annotations__
