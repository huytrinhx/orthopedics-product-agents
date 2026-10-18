"""The full answer eval (issue #25 Slice 2, evals/answer_eval.py): scripted
replies to each pause kind, the resume cap, scoring and the report. Runs
the real deterministic graph with its retrieval and model-calling nodes
faked, as test_deterministic_workflow.py does -- clarify_question_type,
request_clarification and self_eval stay real, since they're what a
scripted reply and the fact check go through -- plus a fake judge, so no
OPENAI_API_KEY is needed. Only the checkpoint test touches Postgres.
"""
import json
import operator
import os
import uuid
from datetime import UTC, datetime
from typing import Annotated

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph
from langgraph.types import interrupt

import agents.workflows.deterministic as det
from agents import registry
from agents.citations import extract_citations
from agents.state import BaseAgentState
from config.db import get_connection
from evals import answer_eval, harness
from evals.answer_eval import (
    AnswerRowResult,
    Pause,
    evaluate_row,
    format_report,
    judge_context,
    load_qa_dataset,
    run_answer_eval,
    scripted_reply,
    summarize,
    write_results,
)
from evals.golden_datasets import build_dataset
from evals.intent_eval import TokenUsage
from memory.checkpointer import get_checkpointer

_INVENTORY_FORM_ID = uuid.uuid4()
_BROCHURE_ID = uuid.uuid4()
_FAMILY_OPTIONS = ["MIS Foot Recon", "Reflex Hybrid"]
_SYSTEM_CONTENTS_OPTION = det._question_type_option(
    next(question_type for question_type in det.QUESTION_TYPES if question_type.name == "system_contents")
)


def _document(document_id: uuid.UUID, filename: str):
    from documents.repository import DocumentRecord

    now = datetime.now(UTC)
    return DocumentRecord(
        id=document_id,
        filename=filename,
        storage_path="",
        status="done",
        error=None,
        uploaded_by=uuid.uuid4(),
        created_at=now,
        updated_at=now,
        document_type_id=None,
        document_type_name=None,
        product_family_name="MIS - Foot Recon System",
        product_family_display_name="MIS Foot Recon",
    )


_DOCUMENTS = [
    _document(_INVENTORY_FORM_ID, "MIS_Inventory-Control-Form.pdf"),
    _document(_BROCHURE_ID, "MIS Brochure and Surgical Technique.pdf"),
]


def _row(**overrides) -> dict:
    return {
        "query": "Which wire goes with the 3.5 PT?",
        "expected_answer": "The 1.4 x 150mm titanium guidepin.",
        "expected_citations": ["MIS Inventory Control Form"],
        "product_family": "MIS - Foot Recon System",
        "question_type": ["system_contents"],
        "dataset": "mis-foot-recon-system.jsonl",
        **overrides,
    }


class _FakeJudge:
    def __init__(self, scores: dict[str, float] | None = None):
        self.scores = scores or {"faithfulness": 0.9, "relevance": 0.8, "style": 0.7, "citation": 1.0}
        self.calls: list[dict] = []

    async def __call__(self, query, retrieved, answer, reference_answer=None):
        self.calls.append(
            {"query": query, "retrieved": retrieved, "answer": answer, "reference_answer": reference_answer}
        )
        return self.scores


def _patch_pausing_nodes(monkeypatch, *, cited_chunk: str | None = None) -> None:
    """Fakes every node of the deterministic graph that would call a model,
    the database or Neo4j. detect_intent asks for the family the way the
    real one does, and its classification reads as two question types, so
    the real clarify_question_type asks too; resolve_synonyms finds an
    ambiguous term until request_clarification (real) has spent the turn's
    round. generate writes what every clarification resolved to into the
    answer, so a test can read the replies' effect straight off it.
    """
    cited_chunk = cited_chunk or f"{_INVENTORY_FORM_ID}#0"

    async def fake_detect_intent(state):
        family = interrupt({"question": "Which product family is this about?", "options": _FAMILY_OPTIONS})
        return {
            "resolved_family": family,
            "resolved_family_id": None,
            "detected_question_types": ["technique_procedural", "system_contents"],
        }

    async def fake_resolve_synonyms(state):
        ambiguity = {"wire": ["Guidepin", "K-wire"]} if state.get("clarification_rounds", 0) == 0 else None
        return {"resolved_canonical_terms": [], "synonym_ambiguity": ambiguity}

    async def fake_hybrid_retrieve(state):
        return {
            "retrieved": [
                {"chunk_id": f"{_INVENTORY_FORM_ID}#0", "document_id": str(_INVENTORY_FORM_ID), "text": "MGT14150 guidepin", "score": 1.0}
            ]
        }

    async def fake_resolve_query_parts(state):
        return {"query_parts": [], "turn_tray_ids": []}

    async def fake_rerank(state):
        return {"reranked": state["retrieved"]}

    async def fake_resolve_skus(state):
        return {"resolved_parts": []}

    async def fake_aggregate_facts(state):
        return {"aggregated_facts": "MGT14150: 1.4 x 150mm guidepin, titanium"}

    async def fake_generate(state):
        answer = (
            f"family={state['resolved_family']}; type={state['resolved_question_type']}; "
            f"term={state.get('clarification_reply')} [{cited_chunk}]"
        )
        update = {"answer": answer, "citations": extract_citations(answer)}
        if state.get("fact_check_issues"):
            update["correction_rounds"] = state.get("correction_rounds", 0) + 1
        return update

    monkeypatch.setattr(det, "detect_intent", fake_detect_intent)
    monkeypatch.setattr(det, "resolve_synonyms", fake_resolve_synonyms)
    monkeypatch.setattr(det, "hybrid_retrieve", fake_hybrid_retrieve)
    monkeypatch.setattr(det, "resolve_query_parts", fake_resolve_query_parts)
    monkeypatch.setattr(det, "rerank", fake_rerank)
    monkeypatch.setattr(det, "resolve_skus", fake_resolve_skus)
    monkeypatch.setattr(det, "aggregate_facts", fake_aggregate_facts)
    monkeypatch.setattr(det, "generate", fake_generate)


async def test_each_pause_kind_resumes_with_the_rows_own_labels(monkeypatch):
    _patch_pausing_nodes(monkeypatch)

    result = await evaluate_row(_row(), "deterministic", _DOCUMENTS, _FakeJudge())

    assert result.failure_reason is None
    assert [(pause.kind, pause.node, pause.reply, pause.reply_source) for pause in result.pauses] == [
        # The catalog name, which _resolve_family accepts like a display name.
        ("family", "detect_intent", "MIS - Foot Recon System", "default"),
        # The offered option text for the row's first question type.
        ("question_type", "clarify_question_type", _SYSTEM_CONTENTS_OPTION, "default"),
        # The interrupt offers no options, so the first canonical term of
        # the ambiguous word -- what it would have offered.
        ("synonym", "request_clarification", "Guidepin", "default"),
    ]
    assert result.pauses[0].question == "Which product family is this about?"
    assert result.pauses[0].options == _FAMILY_OPTIONS
    assert result.pauses[2].question.startswith('Just to confirm, by "wire" do you mean Guidepin or K-wire')
    assert result.answer.startswith("family=MIS - Foot Recon System; type=system_contents; term=Guidepin")


async def test_a_rows_clarification_replies_override_the_defaults(monkeypatch):
    _patch_pausing_nodes(monkeypatch)
    row = _row(
        clarification_replies={"family": "Reflex Hybrid", "question_type": "technique_procedural", "synonym": "K-wire"}
    )

    result = await evaluate_row(row, "deterministic", _DOCUMENTS, _FakeJudge())

    assert [(pause.reply, pause.reply_source) for pause in result.pauses] == [
        ("Reflex Hybrid", "override"),
        ("technique_procedural", "override"),
        ("K-wire", "override"),
    ]
    assert result.answer.startswith("family=Reflex Hybrid; type=technique_procedural; term=K-wire")


def test_an_unoffered_question_type_is_answered_with_its_slug_and_none_with_something_else():
    options = [_SYSTEM_CONTENTS_OPTION, det.SOMETHING_ELSE]

    assert scripted_reply("question_type", _row(question_type=["documents_lookup"]), options, {}) == (
        "documents_lookup",
        "default",
    )
    assert scripted_reply("question_type", _row(question_type=[]), options, {}) == (det.SOMETHING_ELSE, "default")
    # An offered synonym option wins over the state's ambiguity.
    assert scripted_reply("synonym", _row(), ["Pin"], {"synonym_ambiguity": {"wire": ["Guidepin"]}}) == (
        "Pin",
        "default",
    )


class _AlwaysAsksState(BaseAgentState, total=False):
    replies: Annotated[list, operator.add]


def _build_always_asks(checkpointer):
    async def ask(state):
        reply = interrupt({"question": "Again?", "options": ["again", "never"]})
        return {"replies": [reply]}

    graph = StateGraph(_AlwaysAsksState)
    graph.add_node("ask", ask)
    graph.set_entry_point("ask")
    graph.add_edge("ask", "ask")
    return graph.compile(checkpointer=checkpointer)


async def test_a_row_still_paused_after_three_resumes_fails_with_the_reason(monkeypatch):
    monkeypatch.setitem(registry._REGISTRY, "always_asks", registry._Entry(_build_always_asks, True))
    judge = _FakeJudge()

    result = await evaluate_row(_row(), "always_asks", _DOCUMENTS, judge)

    assert result.answer is None
    assert result.failure_reason == "still paused after 3 resumes, at ask: 'Again?'"
    # Three answered (an unknown node gets its first option), and the fourth
    # recorded unanswered.
    assert [(pause.kind, pause.reply, pause.reply_source) for pause in result.pauses] == [
        ("unknown", "again", "default"),
        ("unknown", "again", "default"),
        ("unknown", "again", "default"),
        ("unknown", None, "none"),
    ]
    assert judge.calls == []
    assert result.judge_scores is None
    assert result.citation_recall is None


async def test_the_judge_gets_the_reference_answer_and_what_generate_saw(monkeypatch):
    _patch_pausing_nodes(monkeypatch)
    judge = _FakeJudge()

    result = await evaluate_row(_row(), "deterministic", _DOCUMENTS, judge)

    [call] = judge.calls
    assert call["query"] == "Which wire goes with the 3.5 PT?"
    assert call["answer"] == result.answer
    assert call["reference_answer"] == "The 1.4 x 150mm titanium guidepin."
    assert [passage["chunk_id"] for passage in call["retrieved"]] == [f"{_INVENTORY_FORM_ID}#0", "catalog-facts"]
    assert result.judge_scores == judge.scores


def test_a_react_agent_turns_tool_results_are_the_judges_context():
    from langchain_core.messages import ToolMessage

    context = judge_context(
        {"scratchpad": [ToolMessage(content='[{"sku": "MGT14150"}]', tool_call_id="call-1", name="part_lookup")]}
    )

    assert [(passage["chunk_id"], passage["text"]) for passage in context] == [
        ("tool-result:part_lookup", '[{"sku": "MGT14150"}]')
    ]


async def test_citation_recall_matches_expected_names_to_the_cited_documents(monkeypatch):
    _patch_pausing_nodes(monkeypatch)
    row = _row(
        expected_citations=["MIS Inventory Control Form", "MIS Brochure and Surgical Technique", "MIS Quick Card"]
    )

    result = await evaluate_row(row, "deterministic", _DOCUMENTS, _FakeJudge())

    assert result.cited_documents == ["MIS_Inventory-Control-Form.pdf"]
    # The brochure is ingested but uncited; no document is a "Quick Card".
    assert result.matched_expected_citations == ["MIS Inventory Control Form"]
    assert result.expected_citations_naming_no_document == ["MIS Quick Card"]
    assert result.citation_recall == pytest.approx(1 / 3)
    assert result.fact_check_issues == []


async def test_the_workflows_fact_check_on_the_final_answer_is_reported(monkeypatch):
    # Cites a passage that was never retrieved: self_eval (real) flags it,
    # the correction pass cites it again, and the issue ships with the answer.
    unretrieved_chunk = f"{_BROCHURE_ID}#4"
    _patch_pausing_nodes(monkeypatch, cited_chunk=unretrieved_chunk)

    result = await evaluate_row(_row(), "deterministic", _DOCUMENTS, _FakeJudge())

    assert result.fact_check_issues == [
        f"Cited [{unretrieved_chunk}], but no passage with that id was retrieved this turn."
    ]
    assert result.cited_documents == ["MIS Brochure and Surgical Technique.pdf"]


async def test_a_run_writes_no_checkpoints_or_chat_threads_to_postgres(monkeypatch):
    # Make sure the checkpointer's tables exist, so an empty count means
    # nothing was written rather than that there's nowhere to write.
    async with get_checkpointer(os.environ["DATABASE_URL"]):
        pass
    _patch_pausing_nodes(monkeypatch)
    checkpointers = []
    real_get_workflow = answer_eval.get_workflow

    def spying_get_workflow(name, checkpointer):
        checkpointers.append(checkpointer)
        return real_get_workflow(name, checkpointer)

    monkeypatch.setattr(answer_eval, "get_workflow", spying_get_workflow)

    results = await run_answer_eval([_row(), _row()], "deterministic", _DOCUMENTS, _FakeJudge())

    assert all(result.answer for result in results)
    assert len(checkpointers) == 2 and all(isinstance(saver, InMemorySaver) for saver in checkpointers)
    assert checkpointers[0] is not checkpointers[1]
    connection = await get_connection()
    try:
        async with connection.cursor() as cursor:
            for table in ("checkpoints", "checkpoint_writes", "checkpoint_blobs", "chat_threads"):
                await cursor.execute(f"SELECT count(*) FROM {table} WHERE thread_id LIKE 'eval:%'")
                assert (await cursor.fetchone())[0] == 0, table
    finally:
        await connection.close()


def _result(
    judge_scores: dict[str, float] | None = None,
    answer: str | None = "an answer",
    failure_reason: str | None = None,
    fact_check_issues: list[str] | None = None,
    citation_recall: float | None = 1.0,
    pause_kinds: list[str] | None = None,
    latency_seconds: float = 1.0,
    token_usage: dict[str, TokenUsage] | None = None,
    judge_token_usage: dict[str, TokenUsage] | None = None,
    query: str = "q",
) -> AnswerRowResult:
    return AnswerRowResult(
        dataset="mis-foot-recon-system.jsonl",
        query=query,
        product_family="MIS - Foot Recon System",
        question_types=["system_contents"],
        pauses=[Pause(kind, "node", "?", [], "reply", "default") for kind in pause_kinds or []],
        answer=answer,
        failure_reason=failure_reason,
        judge_scores=judge_scores,
        fact_check_issues=fact_check_issues or [],
        expected_citations=[],
        cited_documents=[],
        matched_expected_citations=[],
        expected_citations_naming_no_document=[],
        citation_recall=citation_recall,
        latency_seconds=latency_seconds,
        token_usage=token_usage or {},
        judge_token_usage=judge_token_usage or {},
    )


def _scores(faithfulness: float, relevance: float, style: float, citation: float) -> dict[str, float]:
    return {"faithfulness": faithfulness, "relevance": relevance, "style": style, "citation": citation}


def test_summarize_computes_score_averages_rates_recall_and_latency():
    results = [
        _result(_scores(1.0, 1.0, 1.0, 1.0), pause_kinds=["family"], latency_seconds=1.0),
        _result(
            _scores(0.5, 0.0, 0.5, 0.0),
            fact_check_issues=["wrong thread"],
            citation_recall=0.5,
            pause_kinds=["family", "question_type"],
            latency_seconds=2.0,
            token_usage={"model-a": TokenUsage(10, 2, 12)},
            judge_token_usage={"model-a": TokenUsage(4, 1, 5)},
        ),
        # Expects no citations: no recall to average in.
        _result(_scores(0.0, 0.5, 0.0, 0.5), citation_recall=None, latency_seconds=3.0,
                token_usage={"model-a": TokenUsage(5, 1, 6), "model-b": TokenUsage(1, 1, 2)}),
        # Still paused at the cap: no answer, no scores, counted as a failure.
        _result(None, answer=None, failure_reason="still paused", citation_recall=None,
                pause_kinds=["synonym"] * 4, latency_seconds=10.0),
    ]

    report = summarize("deterministic", results)

    assert (report.row_count, report.answered_count, report.failure_count) == (4, 3, 1)
    assert report.judge_averages == {
        "faithfulness": pytest.approx(0.5),
        "relevance": pytest.approx(0.5),
        "style": pytest.approx(0.5),
        "citation": pytest.approx(0.5),
    }
    assert report.fact_check_issue_rate == pytest.approx(1 / 3)
    assert report.citation_recall == pytest.approx(0.75)
    assert report.pause_rate == pytest.approx(3 / 4)
    assert report.pause_rate_by_kind == {
        "family": pytest.approx(2 / 4),
        "question_type": pytest.approx(1 / 4),
        "synonym": pytest.approx(1 / 4),
    }
    assert report.latency_p50_seconds == 2.0
    assert report.latency_p95_seconds == 10.0
    assert report.token_usage == {"model-a": TokenUsage(15, 3, 18), "model-b": TokenUsage(1, 1, 2)}
    assert report.judge_token_usage == {"model-a": TokenUsage(4, 1, 5)}


def test_an_empty_run_reports_no_numbers_rather_than_dividing_by_zero():
    report = summarize("deterministic", [])

    assert report.judge_averages["faithfulness"] is None
    assert report.fact_check_issue_rate is None
    assert report.citation_recall is None
    assert "n/a" in format_report(report, [])


def test_the_report_lists_failures_and_low_scores_only(tmp_path):
    results = [
        _result(_scores(0.9, 0.9, 0.9, 0.9), query="A fine answer"),
        _result(_scores(0.9, 0.3, 0.9, 0.9), query="A low relevance answer"),
        _result(None, answer=None, failure_reason="still paused after 3 resumes", query="A stuck question",
                pause_kinds=["family"]),
    ]
    report = summarize("deterministic", results)

    text = format_report(report, results)

    assert "Failures and low scores" in text
    assert "A low relevance answer" in text
    assert "A stuck question" in text
    assert "FAILED: still paused after 3 resumes" in text
    assert "paused (family): '?' -> 'reply'" in text
    assert "A fine answer" not in text

    output_path = tmp_path / "results.json"
    write_results(output_path, report, results)
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["report"]["failure_count"] == 1
    assert written["rows"][2]["pauses"][0]["kind"] == "family"


def test_the_default_dataset_is_every_per_family_qa_file():
    rows = load_qa_dataset(harness.DEFAULT_ANSWER_DATASET)

    assert {row["dataset"] for row in rows} == {
        "mis-foot-recon-system.jsonl",
        "reflex-foot-recon-systems.jsonl",
        "reflex-hybrid.jsonl",
    }
    for row in rows:
        assert {"query", "expected_answer", "expected_citations", "product_family", "question_type"} <= row.keys()


def test_a_single_dataset_file_loads_only_its_rows():
    rows = load_qa_dataset(harness.DEFAULT_ANSWER_DATASET / "reflex-hybrid.jsonl")

    assert rows and {row["dataset"] for row in rows} == {"reflex-hybrid.jsonl"}


def test_build_dataset_writes_clarification_replies_only_for_overridden_queries(monkeypatch):
    [first_row, second_row] = build_dataset.load_rows()[:2]
    first_prompt = first_row[build_dataset.COL_PROMPT].strip()
    monkeypatch.setattr(build_dataset, "CLARIFICATION_REPLY_OVERRIDES", {first_prompt: {"synonym": "Guidepin"}})

    assert build_dataset.build_qa_record(first_row)["clarification_replies"] == {"synonym": "Guidepin"}
    assert "clarification_replies" not in build_dataset.build_qa_record(second_row)
    build_dataset.check_clarification_reply_overrides({first_prompt})


def test_build_dataset_rejects_a_stale_or_unknown_override(monkeypatch):
    monkeypatch.setattr(build_dataset, "CLARIFICATION_REPLY_OVERRIDES", {"Gone?": {"family": "MIS"}})
    with pytest.raises(ValueError, match="not in the CSV"):
        build_dataset.check_clarification_reply_overrides({"Here?"})

    monkeypatch.setattr(build_dataset, "CLARIFICATION_REPLY_OVERRIDES", {"Here?": {"tray": "MIS"}})
    with pytest.raises(ValueError, match="Unknown pause kind"):
        build_dataset.check_clarification_reply_overrides({"Here?"})


def test_build_dataset_and_the_eval_agree_on_the_pause_kinds():
    assert build_dataset.CLARIFICATION_PAUSE_KINDS == answer_eval.PAUSE_KINDS


async def test_run_eval_returns_the_report_and_rows_as_dicts(monkeypatch, tmp_path):
    _patch_pausing_nodes(monkeypatch)
    dataset_path = tmp_path / "one-family.jsonl"
    dataset_path.write_text(json.dumps(_row()) + "\n", encoding="utf-8")

    async def fake_documents():
        return _DOCUMENTS

    async def fake_judge(query, retrieved, answer, reference_answer=None):
        return _scores(1.0, 1.0, 1.0, 1.0)

    monkeypatch.setattr(harness, "_ingested_documents", fake_documents)
    monkeypatch.setattr(answer_eval, "judge_answer", fake_judge)

    output = await harness.run_eval("deterministic", str(dataset_path))

    assert output["report"]["row_count"] == 1
    assert output["report"]["citation_recall"] == pytest.approx(1.0)
    assert output["rows"][0]["dataset"] == "one-family.jsonl"
    assert len(output["rows"][0]["pauses"]) == 3


def test_the_answers_command_refuses_without_an_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(SystemExit) as exit_info:
        harness.main(["answers", "--workflow", "deterministic"])

    assert exit_info.value.code == 2


@pytest.mark.parametrize("workflow_name", ["no_such_workflow", "supervisor"])
def test_the_answers_command_refuses_an_unknown_or_unbuilt_workflow(monkeypatch, workflow_name):
    monkeypatch.setenv("OPENAI_API_KEY", "placeholder")

    with pytest.raises(SystemExit) as exit_info:
        harness.main(["answers", "--workflow", workflow_name])

    assert exit_info.value.code == 2
