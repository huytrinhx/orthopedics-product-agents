"""Product Family intent and clarification (issue #18), through the real
deterministic graph: detect_intent resolves a family (by display name) or
interrupts with at most MAX_FAMILY_CANDIDATES candidates -- every family
when there are none -- a clicked option or free text resolves back to a
real family on resume, and hybrid_retrieve hard-filters by it.

The classifier and the family list are faked (no LLM, no database), and so
is everything past hybrid_retrieve; hybrid_retrieve itself runs for real
against a recording vector_search, so what reaches retrieval is asserted
directly. The SQL side of the filter is test_vector_store.py's.
"""
import uuid

from langchain_core.messages import HumanMessage
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
    def __init__(self, monkeypatch, classification: det._IntentClassification, families=_FAMILIES):
        self.classifier_family_names: list[list[str]] = []
        self.search = _RecordingVectorSearch()

        async def fake_list_families_with_trays():
            return list(families)

        async def fake_classify_intent(query, family_names):
            self.classifier_family_names.append(family_names)
            return classification.model_copy(deep=True)

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
    # The classifier chooses among display names, not catalog names.
    assert harness.classifier_family_names == [[family.display_name for family in _FAMILIES]]
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


def test_no_workflow_reads_the_legacy_system_tag_list():
    for module in (det, react_agent):
        assert not hasattr(module, "list_systems")
    assert "resolved_system" not in det.DeterministicState.__annotations__
    assert "resolved_system" not in react_agent.ReactAgentState.__annotations__
