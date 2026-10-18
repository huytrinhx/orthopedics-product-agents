"""Question-type clarification (issue #19), through the real deterministic
graph: a single classified question type never interrupts; several pause
the turn offering their descriptions plus "Something else"; resuming with a
clicked option or free text resolves exactly one question type
("Something else" -> unclassified), and hybrid_retrieve filters by that one
type. A turn that also needs a family asks family first, then question
type, once each.

Faked like test_family_clarification.py: the classifier, the family list,
the hierarchy table, and everything past hybrid_retrieve; hybrid_retrieve
itself runs for real against a recording vector_search.
"""
import uuid

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

import agents.workflows.deterministic as det
from catalog.repository import ProductFamilyRecord
from doc_type_hierarchy.repository import RankedDocumentType

_HYBRID = ProductFamilyRecord(uuid.uuid4(), "REFLEX HYBRID", "Reflex Hybrid")
_POWER = ProductFamilyRecord(uuid.uuid4(), "MIS - Power System", "MIS Power")
_FAMILIES = [_HYBRID, _POWER]

_TECHNIQUE = RankedDocumentType(uuid.uuid4(), "Surgical Technique")
_INVENTORY = RankedDocumentType(uuid.uuid4(), "Inventory Control")
_HIERARCHY = {
    "technique_procedural": [_TECHNIQUE],
    "product_characteristics": [_INVENTORY],
}

_TECHNIQUE_OPTION = (
    "Step-by-step surgical technique, procedural walkthrough, applications of "
    "products or parts, etc."
)
_CHARACTERISTICS_OPTION = "Materials, dimensions, thread types, color, etc."
_QUESTION = "Which of these best describes what you're asking?"


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
        question_types: list[str],
        product_family: str | None = "Reflex Hybrid",
        reply_classification: str = "unclassified",
    ):
        self.search = _RecordingVectorSearch()
        self.classify_calls = 0
        self.reply_classifications: list[tuple[str, str]] = []
        self.generated_question_types: list[str] = []
        classification = det._IntentClassification(
            product_family=product_family, question_type=question_types
        )

        async def fake_list_families_with_trays():
            return list(_FAMILIES)

        async def fake_classify_intent(query, family_names, previous_family=None, previous_message=None):
            self.classify_calls += 1
            return classification.model_copy(deep=True)

        async def fake_classify_question_type_reply(query, reply):
            self.reply_classifications.append((query, reply))
            return reply_classification

        async def fake_list_hierarchy():
            return _HIERARCHY

        async def fake_resolve_synonyms(state):
            return {"resolved_canonical_terms": [], "synonym_ambiguity": None}

        async def fake_resolve_query_parts(state):
            return {"query_parts": [], "turn_tray_ids": []}

        async def fake_resolve_skus(state):
            return {"resolved_parts": []}

        async def fake_aggregate_facts(state):
            return {"aggregated_facts": ""}

        async def fake_generate(state):
            self.generated_question_types.append(state["resolved_question_type"])
            return {"answer": "final answer", "citations": []}

        async def fake_self_eval(state):
            return {"fact_check_issues": []}

        monkeypatch.setattr(det, "list_families_with_trays", fake_list_families_with_trays)
        monkeypatch.setattr(det, "_classify_intent", fake_classify_intent)
        monkeypatch.setattr(det, "_classify_question_type_reply", fake_classify_question_type_reply)
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

    async def ask(self, message: str = "How do I place the screw?") -> dict:
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

    def filtered_document_type_ids(self) -> list[list[str] | None]:
        return [call["document_type_ids"] for call in self.search.calls]


@pytest.mark.parametrize(
    "question_types, expected",
    [
        (["technique_procedural"], "technique_procedural"),
        (["unclassified"], "unclassified"),
        ([], "unclassified"),
        # unclassified alongside a real type means nothing; a duplicate is one type.
        (["technique_procedural", "unclassified", "technique_procedural"], "technique_procedural"),
    ],
)
async def test_a_single_question_type_never_interrupts(monkeypatch, question_types, expected):
    harness = _Harness(monkeypatch, question_types)

    result = await harness.ask()

    assert "__interrupt__" not in result
    assert result["answer"] == "final answer"
    assert result["resolved_question_type"] == expected
    assert harness.generated_question_types == [expected]


async def test_several_question_types_interrupt_with_their_descriptions_and_something_else(monkeypatch):
    harness = _Harness(monkeypatch, ["technique_procedural", "product_characteristics"])

    result = await harness.ask()

    assert harness.search.calls == []  # paused before retrieval
    assert result["__interrupt__"][0].value == {
        "question": _QUESTION,
        "options": [_TECHNIQUE_OPTION, _CHARACTERISTICS_OPTION, "Something else"],
    }


async def test_resume_with_a_clicked_option_resolves_one_type_and_filters_by_it(monkeypatch):
    harness = _Harness(monkeypatch, ["technique_procedural", "product_characteristics"])
    await harness.ask()

    result = await harness.resume(_CHARACTERISTICS_OPTION)

    assert "__interrupt__" not in result
    assert result["resolved_question_type"] == "product_characteristics"
    assert harness.generated_question_types == ["product_characteristics"]
    assert harness.filtered_document_type_ids() == [[str(_INVENTORY.id)]]
    assert harness.reply_classifications == []  # no model call for a clicked option
    # Resuming replays only clarify_question_type, not the intent classifier.
    assert harness.classify_calls == 1


async def test_something_else_resolves_to_unclassified_and_retrieves_unfiltered(monkeypatch):
    harness = _Harness(monkeypatch, ["technique_procedural", "product_characteristics"])
    await harness.ask()

    result = await harness.resume("something else")

    assert result["resolved_question_type"] == "unclassified"
    assert harness.filtered_document_type_ids() == [None]


async def test_free_text_naming_a_type_resolves_it_without_a_model_call(monkeypatch):
    harness = _Harness(monkeypatch, ["technique_procedural", "product_characteristics"])
    await harness.ask()

    result = await harness.resume("  Technique Procedural ")

    assert result["resolved_question_type"] == "technique_procedural"
    assert harness.reply_classifications == []


async def test_other_free_text_is_classified_to_exactly_one_type(monkeypatch):
    harness = _Harness(
        monkeypatch,
        ["technique_procedural", "product_characteristics"],
        reply_classification="compatibility_lookup",
    )
    await harness.ask("Does this screw work here?")

    result = await harness.resume("whether it fits the plate")

    # Any question type, not only the offered ones -- like a family reply.
    assert "__interrupt__" not in result
    assert harness.reply_classifications == [("Does this screw work here?", "whether it fits the plate")]
    assert result["resolved_question_type"] == "compatibility_lookup"
    assert harness.generated_question_types == ["compatibility_lookup"]


async def test_family_then_question_type_are_each_asked_once_in_that_order(monkeypatch):
    harness = _Harness(
        monkeypatch, ["technique_procedural", "product_characteristics"], product_family=None
    )

    first = await harness.ask()
    assert first["__interrupt__"][0].value["question"] == "Which product family is this about?"

    second = await harness.resume("MIS Power")
    assert second["__interrupt__"][0].value["question"] == _QUESTION
    assert harness.search.calls == []

    third = await harness.resume(_TECHNIQUE_OPTION)
    assert "__interrupt__" not in third
    assert third["answer"] == "final answer"
    assert third["resolved_family_id"] == str(_POWER.id)
    assert third["resolved_question_type"] == "technique_procedural"
    assert [call["product_family_id"] for call in harness.search.calls] == [str(_POWER.id)]
    assert harness.filtered_document_type_ids() == [[str(_TECHNIQUE.id)]]


async def test_question_type_clarification_does_not_spend_the_synonym_budget(monkeypatch):
    harness = _Harness(monkeypatch, ["technique_procedural", "product_characteristics"])
    await harness.ask()

    result = await harness.resume(_TECHNIQUE_OPTION)

    assert result["clarification_rounds"] == 0
