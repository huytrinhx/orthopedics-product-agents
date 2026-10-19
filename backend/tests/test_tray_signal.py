"""Issue #21: tray as a soft rerank and answer signal in the deterministic
workflow (backend/agents/workflows/deterministic.py).

The stated-diameter rule and rerank's bonus are pure functions, tested
directly -- the diameter rule against the committed master catalog itself,
so it's proven on the real Cannulated Screws trays rather than on made-up
ones. resolve_query_parts is also run against a real Neo4j and Postgres
(fresh uuid-suffixed families/trays/SKUs, same approach as
test_catalog.py), and once through the compiled graph with its lookups
faked, to show the trays reach rerank without any new interrupt.
"""
import uuid
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

import agents.workflows.deterministic as det
from catalog.repository import (
    ProductFamilyRecord,
    TrayRecord,
    list_families_with_trays,
    upsert_families_and_trays,
)
from ingestion.seed_master_catalog import parse_master_catalog
from retrieval.graph_client import get_graph_client

# Same fixture test_seed_master_catalog.py reads: the committed workbook.
_MASTER_CATALOG = Path(__file__).resolve().parents[1] / "evals" / "master-product-detail-oct-26.xlsx"


def _unique(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


def _passage(index: int, *, tray_ids=(), document_type_id: str | None = None) -> dict:
    # Identical text throughout, so BM25 ties and only the bonuses decide.
    return {
        "chunk_id": f"doc-{index}#0",
        "document_id": f"doc-{index}",
        "text": "identical text",
        "score": 0.0,
        "document_type": None,
        "document_type_id": document_type_id,
        "tray_ids": list(tray_ids),
    }


class _FakePartLookup:
    """Stands in for the part_lookup tool: answers every term with the
    parts given for it (none by default) and records each term asked."""

    def __init__(self, parts_by_term: dict[str, list[dict]] | None = None):
        self.parts_by_term = parts_by_term or {}
        self.terms: list[str] = []

    async def ainvoke(self, arguments: dict) -> list[dict]:
        self.terms.append(arguments["term"])
        return self.parts_by_term.get(arguments["term"], [])


def _no_model_call():
    raise AssertionError("the tray signal must never call the model")


# ---- The stated-diameter rule, against the real master catalog ----


@pytest.fixture(scope="module")
def cannulated_descriptions_by_tray() -> dict[str, list[str]]:
    descriptions_by_tray: dict[str, list[str]] = {}
    for row in parse_master_catalog(_MASTER_CATALOG):
        if row.family == det._DIAMETER_RULE_FAMILY:
            descriptions_by_tray.setdefault(row.tray, []).append(row.description)
    return descriptions_by_tray


@pytest.mark.parametrize(
    ("query", "tray_fragment"),
    [
        ("What lengths does the 2.7mm snap-off screw come in?", "Forefoot Fixation System"),
        ("What lengths does the 4.0mm cannulated screw come in?", "4.0/4.5mm Screw System"),
        ("Is there a 4.5 headed screw?", "4.0/4.5mm Screw System"),
        ("Do we have a 5.5 x 40 screw?", "5.5mm Screw System"),
        ("Which guidepin goes with the Ø7.0 screw?", "6.5/7.0mm Screw System"),
    ],
)
def test_a_stated_diameter_names_its_cannulated_screws_tray(
    cannulated_descriptions_by_tray, query, tray_fragment
):
    [expected] = [tray for tray in cannulated_descriptions_by_tray if tray_fragment in tray]
    assert det._trays_for_stated_diameters(query, cannulated_descriptions_by_tray) == [expected]


@pytest.mark.parametrize(
    "query",
    ["Which cannulated screws are headless?", "Where is MSD14020?", "Do we have a 40mm length?"],
)
def test_no_stated_diameter_names_no_tray(cannulated_descriptions_by_tray, query):
    # No size at all, a SKU's digits, and a length no screw's diameter is.
    assert det._trays_for_stated_diameters(query, cannulated_descriptions_by_tray) == []


def test_turn_tray_ids_maps_part_display_names_and_rule_catalog_names_to_ids():
    first = TrayRecord(uuid.uuid4(), "Catalog First", "Display First")
    second = TrayRecord(uuid.uuid4(), "Catalog Second", "Display Second")
    parts = [
        {"sku": "SKU-1", "trays": ["Display First", "Some Other Family's Tray"]},
        {"sku": "SKU-2", "trays": ["Display First"]},
    ]

    assert det._turn_tray_ids(parts, ["Catalog Second"], [first, second]) == [str(first.id), str(second.id)]


# ---- rerank's tray bonus ----


async def test_rerank_ranks_a_document_tagged_with_the_turns_tray_above_an_otherwise_equal_one():
    turn_tray, other_tray = str(uuid.uuid4()), str(uuid.uuid4())
    state = {
        "query": "identical text",
        "retrieved": [_passage(0), _passage(1, tray_ids=[other_tray]), _passage(2, tray_ids=[other_tray, turn_tray])],
        "turn_tray_ids": [turn_tray],
    }

    reranked = (await det.rerank(state))["reranked"]

    assert reranked[0]["chunk_id"] == "doc-2#0"
    assert reranked[0]["score"] == pytest.approx(det._TRAY_BONUS)
    assert [passage["score"] for passage in reranked[1:]] == [0.0, 0.0]


async def test_tray_bonus_is_flat_however_many_of_the_turns_trays_a_document_carries():
    first_tray, second_tray = str(uuid.uuid4()), str(uuid.uuid4())
    state = {
        "query": "identical text",
        "retrieved": [_passage(0, tray_ids=[first_tray, second_tray]), _passage(1, tray_ids=[first_tray])],
        "turn_tray_ids": [first_tray, second_tray],
    }

    reranked = (await det.rerank(state))["reranked"]

    assert [passage["score"] for passage in reranked] == pytest.approx([det._TRAY_BONUS, det._TRAY_BONUS])


async def test_tray_bonus_never_overtakes_the_doc_type_order():
    """The tray bonus is the lowest doc-type rank's bonus, so a tray match
    can lift a lower doc type level with the one above it, never past it."""
    assert det._TRAY_BONUS == det._DOCTYPE_BONUS_STEP
    top_type, second_type, turn_tray = str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4())
    state = {
        "query": "identical text",
        "retrieved": [
            _passage(0, document_type_id=second_type),
            _passage(1, tray_ids=[turn_tray], document_type_id=second_type),
            _passage(2, document_type_id=top_type),
        ],
        "document_type_bonuses": {top_type: 2 * det._DOCTYPE_BONUS_STEP, second_type: det._DOCTYPE_BONUS_STEP},
        "turn_tray_ids": [turn_tray],
    }

    reranked = (await det.rerank(state))["reranked"]

    assert [passage["chunk_id"] for passage in reranked] == ["doc-2#0", "doc-1#0", "doc-0#0"]


async def test_hybrid_retrieve_carries_tray_ids_and_never_filters_by_tray(monkeypatch):
    class _RecordingVectorSearch:
        def __init__(self):
            self.calls: list[dict] = []

        async def ainvoke(self, arguments: dict) -> list[dict]:
            self.calls.append(arguments)
            return [
                {
                    "citation": "doc-1#0",
                    "document_id": "doc-1",
                    "content": "text",
                    "score": 1.0,
                    "document_type": None,
                    "document_type_id": None,
                    "tray_ids": ["tray-1"],
                }
            ]

    async def fake_list_hierarchy():
        return {}

    search = _RecordingVectorSearch()
    monkeypatch.setattr(det, "vector_search", search)
    monkeypatch.setattr(det, "list_hierarchy", fake_list_hierarchy)

    result = await det.hybrid_retrieve({"query": "where is it", "turn_tray_ids": ["tray-1"]})

    assert result["retrieved"][0]["tray_ids"] == ["tray-1"]
    assert set(search.calls[0]) == {"query", "top_k", "document_type_ids", "product_family_id"}


# ---- resolve_query_parts / resolve_skus ----


async def _seed_family(trays_with_parts: dict[str, list[tuple[str, str]]]) -> tuple[str, ProductFamilyRecord]:
    """Writes a fresh family whose trays hold the given (sku, description)
    parts to the graph, mirrors it into Postgres, and returns its catalog
    name with its mirrored record."""
    family_name = _unique("Tray Signal Family")
    graph = get_graph_client()
    await graph.ensure_constraints()
    for tray, parts in trays_with_parts.items():
        await graph.upsert_tray(tray, family_name)
        for sku, description in parts:
            await graph.upsert_part(sku, tray, description=description)
    await upsert_families_and_trays({family_name: list(trays_with_parts)})
    [family] = [family for family in await list_families_with_trays() if family.name == family_name]
    return family_name, family


def _tray_id(family: ProductFamilyRecord, tray_name: str) -> str:
    return next(str(tray.id) for tray in family.trays if tray.name == tray_name)


async def test_a_query_naming_a_sku_ranks_its_trays_documents_first(monkeypatch):
    monkeypatch.setattr(det, "get_chat_model", _no_model_call)
    mini_tray, nitinol_tray = _unique("MINI Tray"), _unique("Nitinol Tray")
    blade_sku, staple_sku = _unique("BLADE").upper(), _unique("STAPLE").upper()
    _, family = await _seed_family(
        {mini_tray: [(blade_sku, "BLADE, AKIN, STEPPED")], nitinol_tray: [(staple_sku, "STAPLE, TETRA")]}
    )
    state = {
        "query": f"Where is {blade_sku}?",
        "resolved_family": family.display_name,
        "resolved_family_id": str(family.id),
        "resolved_canonical_terms": [],
        "retrieved": [
            _passage(0, tray_ids=[_tray_id(family, nitinol_tray)]),
            _passage(1, tray_ids=[_tray_id(family, mini_tray)]),
        ],
    }

    state.update(await det.resolve_query_parts(state))
    reranked = (await det.rerank(state))["reranked"]

    assert [part["sku"] for part in state["query_parts"]] == [blade_sku]
    assert state["query_parts"][0]["trays"] == [mini_tray]
    assert state["turn_tray_ids"] == [_tray_id(family, mini_tray)]
    assert reranked[0]["chunk_id"] == "doc-1#0"


async def test_a_stated_diameter_derives_the_tray_with_no_model_call(monkeypatch):
    four_tray, five_tray = _unique("4.0 Screw Tray"), _unique("5.5 Screw Tray")
    family_name, family = await _seed_family(
        {
            four_tray: [(_unique("SCREW").upper(), "SCREW, CANN, HEADED, 4.0 X 20MM")],
            five_tray: [(_unique("SCREW").upper(), "SCREW, CANN, HEADED, 5.5 X 40MM")],
        }
    )
    monkeypatch.setattr(det, "_DIAMETER_RULE_FAMILY", family_name)
    monkeypatch.setattr(det, "get_chat_model", _no_model_call)
    # No Part matches, so whatever tray comes back came from the rule alone.
    monkeypatch.setattr(det, "part_lookup", _FakePartLookup())
    state = {
        "query": "Do we have a 5.5mm screw in 40mm?",
        "resolved_family": family.display_name,
        "resolved_family_id": str(family.id),
        "resolved_canonical_terms": [],
    }

    assert (await det.resolve_query_parts(state))["turn_tray_ids"] == [_tray_id(family, five_tray)]


async def test_the_diameter_rule_only_applies_to_its_own_family(monkeypatch):
    four_tray = _unique("4.0 Screw Tray")
    _, family = await _seed_family({four_tray: [(_unique("SCREW").upper(), "SCREW, CANN, HEADED, 4.0 X 20MM")]})
    monkeypatch.setattr(det, "part_lookup", _FakePartLookup())
    state = {
        "query": "Do we have a 4.0mm screw?",
        "resolved_family": family.display_name,
        "resolved_family_id": str(family.id),
        "resolved_canonical_terms": [],
    }

    assert (await det.resolve_query_parts(state))["turn_tray_ids"] == []


async def test_resolve_skus_reuses_the_query_parts_outside_technique_procedural(monkeypatch):
    lookup = _FakePartLookup()
    monkeypatch.setattr(det, "part_lookup", lookup)
    query_parts = [{"sku": "SKU-1", "trays": ["Tray"]}]

    result = await det.resolve_skus(
        {"query": "where is sku-1", "resolved_question_type": "system_contents", "query_parts": query_parts}
    )

    assert result["resolved_parts"] == query_parts
    assert lookup.terms == []


async def test_resolve_skus_adds_only_passage_terms_for_technique_procedural(monkeypatch):
    passage_part = {"sku": "SKU-2", "trays": ["Tray"]}
    lookup = _FakePartLookup({"guidepin": [passage_part]})
    monkeypatch.setattr(det, "part_lookup", lookup)
    query_parts = [{"sku": "SKU-1", "trays": ["Tray"]}]

    result = await det.resolve_skus(
        {
            "query": "how do I place the screw",
            "resolved_question_type": "technique_procedural",
            "query_parts": query_parts,
            "reranked": [_passage(0) | {"text": "Insert the guidepin, then the screw."}],
        }
    )

    assert result["resolved_parts"] == [*query_parts, passage_part]
    # "screw" was a query term, already looked up by resolve_query_parts.
    assert "screw" not in lookup.terms
    assert "guidepin" in lookup.terms


# ---- Tray in the catalog facts, and in the answer ----

_MINI_TRAY = "REFLEX® MINI Implant System"
_NITINOL_TRAY = "REFLEX® Nitinol Implant System"
_AKIN_BLADE = {"sku": "RFX25001", "description": "BLADE, AKIN, STEPPED", "product_family": "REFLEX", "trays": [_MINI_TRAY]}
_TETRA_STAPLE = {"sku": "RFX18020", "description": "STAPLE, TETRA", "product_family": "REFLEX", "trays": [_NITINOL_TRAY]}


class _ScriptedModel:
    def __init__(self, replies: list[str]):
        self._replies = list(replies)
        self.prompts: list[list] = []

    async def ainvoke(self, messages):
        self.prompts.append(messages)
        return AIMessage(content=self._replies.pop(0))


def test_catalog_facts_name_each_parts_tray():
    assert det._format_part(_AKIN_BLADE) == f"- RFX25001: BLADE, AKIN, STEPPED (product_family=REFLEX, tray={_MINI_TRAY})"
    shared = {"sku": "MSN90003", "description": "PICKUPS", "trays": [_MINI_TRAY, _NITINOL_TRAY]}
    assert det._format_part(shared) == f"- MSN90003: PICKUPS (tray={_MINI_TRAY}; {_NITINOL_TRAY})"


async def test_generate_sees_the_tray_and_a_wrong_tray_earns_a_correction(monkeypatch):
    """Runs the compiled graph with retrieval and the Part lookups faked,
    but the real resolve_query_parts, rerank, resolve_skus,
    aggregate_facts, generate and self_eval: the tray reaches generate's
    catalog facts, a draft placing the blade in the wrong tray fails the
    fact check once, and the turn never pauses to ask anything."""
    mini_tray_id, nitinol_tray_id = uuid.uuid4(), uuid.uuid4()
    family = ProductFamilyRecord(
        uuid.uuid4(),
        "REFLEX",
        "REFLEX",
        [TrayRecord(mini_tray_id, _MINI_TRAY, _MINI_TRAY), TrayRecord(nitinol_tray_id, _NITINOL_TRAY, _NITINOL_TRAY)],
    )

    async def fake_list_families_with_trays():
        return [family]

    async def fake_classify_intent(
        query, family_names, previous_family=None, previous_message=None, family_descriptions=None
    ):
        return det._IntentClassification(product_family="REFLEX", question_type=["system_contents"])

    async def fake_resolve_synonyms(state):
        return {"resolved_canonical_terms": [], "synonym_ambiguity": None}

    async def fake_hybrid_retrieve(state):
        return {
            "retrieved": [
                _passage(0, tray_ids=[str(nitinol_tray_id)]),
                _passage(1, tray_ids=[str(mini_tray_id)]),
            ],
            "document_type_bonuses": {},
        }

    model = _ScriptedModel(
        [
            f"RFX25001 is in the {_NITINOL_TRAY} tray.",
            f"RFX25001 is in the {_MINI_TRAY} tray.",
        ]
    )
    monkeypatch.setattr(det, "list_families_with_trays", fake_list_families_with_trays)
    monkeypatch.setattr(det, "_classify_intent", fake_classify_intent)
    monkeypatch.setattr(det, "resolve_synonyms", fake_resolve_synonyms)
    monkeypatch.setattr(det, "hybrid_retrieve", fake_hybrid_retrieve)
    monkeypatch.setattr(det, "part_lookup", _FakePartLookup({"rfx25001": [_AKIN_BLADE]}))
    monkeypatch.setattr(det, "get_chat_model", lambda: model)

    graph = det.build_graph(InMemorySaver())
    thread_id = _unique("test")
    query = "Where is RFX25001?"
    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content=query)],
            "query": query,
            "search_query": query,
            "clarification_rounds": 0,
            "clarification_reply": None,
            "correction_rounds": 0,
            "fact_check_issues": [],
            "user_id": "u1",
            "thread_id": thread_id,
        },
        {"configurable": {"thread_id": thread_id}},
    )

    assert "__interrupt__" not in result
    assert result["turn_tray_ids"] == [str(mini_tray_id)]
    assert result["reranked"][0]["chunk_id"] == "doc-1#0"
    assert f"tray={_MINI_TRAY}" in model.prompts[0][-1].content
    assert f"RFX25001 is in {_MINI_TRAY} in the catalog" in model.prompts[1][-1].content
    assert result["answer"] == f"RFX25001 is in the {_MINI_TRAY} tray."
    assert result["fact_check_issues"] == []
