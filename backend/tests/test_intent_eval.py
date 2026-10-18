"""The intent eval (issue #25 Slice 1, evals/intent_eval.py): row comparison
and report metrics, with a faked classifier and an in-memory families list
-- no OPENAI_API_KEY, no database. The classifier's own behavior is
test_family_clarification.py's and test_question_type_clarification.py's;
this checks the eval scores whatever it returns the way the workflow would
act on it.
"""
import json
import uuid

import pytest
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from agents.workflows.deterministic import _IntentClassification
from catalog.repository import ProductFamilyRecord
from evals import harness
from evals.intent_eval import (
    IntentRowResult,
    TokenUsage,
    evaluate_row,
    format_report,
    last_user_turn,
    load_intent_dataset,
    run_intent_eval,
    summarize,
    write_results,
)

# Display names differ from catalog names, like test_family_clarification.py:
# the classifier answers in display names, the dataset labels in catalog names.
_FOOT_RECON = ProductFamilyRecord(uuid.uuid4(), "MIS - Foot Recon System", "MIS Foot Recon")
_HYBRID = ProductFamilyRecord(uuid.uuid4(), "REFLEX HYBRID", "Reflex Hybrid")
_FAMILIES = [_FOOT_RECON, _HYBRID]


def _row(
    query: str,
    family: str | None,
    question_types: list[str],
    expects_family_clarification: bool = False,
    expects_question_type_clarification: bool = False,
) -> dict:
    return {
        "turns": [{"role": "user", "content": query}],
        "expected_product_family": family,
        "expected_question_type": question_types,
        "expects_family_clarification": expects_family_clarification,
        "expects_question_type_clarification": expects_question_type_clarification,
    }


class _FakeClassifier:
    """Answers each query with a canned classification, recording what it
    was asked so tests can check it got the workflow's own inputs.
    """

    def __init__(self, classifications: dict[str, _IntentClassification]):
        self.classifications = classifications
        self.calls: list[tuple[str, list[str]]] = []

    async def __call__(self, query: str, family_names: list[str]) -> _IntentClassification:
        self.calls.append((query, family_names))
        return self.classifications[query]


async def test_a_confident_display_name_pick_matches_the_catalog_name_label():
    classify = _FakeClassifier(
        {"What screws?": _IntentClassification(product_family="MIS Foot Recon", question_type=["system_contents"])}
    )

    result = await evaluate_row(
        _row("What screws?", "MIS - Foot Recon System", ["system_contents"]), _FAMILIES, classify
    )

    assert classify.calls == [("What screws?", ["MIS Foot Recon", "Reflex Hybrid"])]
    assert result.resolved_family == "MIS Foot Recon"
    assert result.family_correct
    assert result.expected_family_in_catalog
    assert result.question_types_exact_match and result.question_types_overlap
    assert not result.would_clarify_family
    assert not result.would_clarify_question_type


async def test_the_wrong_family_scores_incorrect():
    classify = _FakeClassifier({"Torque?": _IntentClassification(product_family="Reflex Hybrid")})

    result = await evaluate_row(_row("Torque?", "MIS - Foot Recon System", []), _FAMILIES, classify)

    assert result.resolved_family == "Reflex Hybrid"
    assert not result.family_correct


async def test_no_family_where_none_is_expected_is_correct_and_clarifies():
    classify = _FakeClassifier({"What screws are in the set?": _IntentClassification(product_family="null")})

    result = await evaluate_row(
        _row("What screws are in the set?", None, ["system_contents"], expects_family_clarification=True),
        _FAMILIES,
        classify,
    )

    assert result.resolved_family is None
    assert result.family_correct
    assert result.would_clarify_family
    assert result.expects_family_clarification


async def test_no_family_mirrored_never_clarifies_like_detect_intent():
    classify = _FakeClassifier({"What screws?": _IntentClassification()})

    result = await evaluate_row(
        _row("What screws?", None, [], expects_family_clarification=True), [], classify
    )

    assert classify.calls == [("What screws?", [])]
    assert not result.would_clarify_family


async def test_a_label_naming_an_unmirrored_family_can_never_score_correct():
    classify = _FakeClassifier({"Screws?": _IntentClassification()})

    result = await evaluate_row(_row("Screws?", "Cannulated Screws", []), _FAMILIES, classify)

    assert not result.expected_family_in_catalog
    assert not result.family_correct


async def test_several_question_types_clarify_and_score_exact_and_overlap_separately():
    classify = _FakeClassifier(
        {
            "Setup?": _IntentClassification(
                product_family="MIS Foot Recon",
                # `unclassified` alongside a real type is dropped, as in the workflow.
                question_type=["technique_procedural", "unclassified", "system_contents"],
            )
        }
    )

    result = await evaluate_row(
        _row("Setup?", "MIS - Foot Recon System", ["technique_procedural"]), _FAMILIES, classify
    )

    assert result.detected_question_types == ["technique_procedural", "system_contents"]
    assert not result.question_types_exact_match
    assert result.question_types_overlap
    assert result.would_clarify_question_type
    assert not result.expects_question_type_clarification


async def test_no_question_type_comes_back_as_unclassified():
    classify = _FakeClassifier({"Hi": _IntentClassification(product_family="Reflex Hybrid")})

    result = await evaluate_row(_row("Hi", "REFLEX HYBRID", ["unclassified"]), _FAMILIES, classify)

    assert result.detected_question_types == ["unclassified"]
    assert result.question_types_exact_match
    assert not result.would_clarify_question_type


def test_the_last_user_turn_is_what_gets_classified():
    row = {
        "turns": [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "follow-up"},
            {"role": "assistant", "content": "another answer"},
        ]
    }

    assert last_user_turn(row) == "follow-up"


async def test_token_usage_is_read_from_langchain_usage_metadata():
    fake_model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="ok",
                    usage_metadata={"input_tokens": 120, "output_tokens": 30, "total_tokens": 150},
                    response_metadata={"model_name": "fake-model"},
                )
            ]
        )
    )

    async def classify(query: str, family_names: list[str]) -> _IntentClassification:
        await fake_model.ainvoke(query)
        return _IntentClassification(product_family="Reflex Hybrid")

    result = await evaluate_row(_row("Torque?", "REFLEX HYBRID", []), _FAMILIES, classify)

    assert result.token_usage == {"fake-model": TokenUsage(120, 30, 150)}


def _result(
    family_correct: bool = True,
    exact: bool = True,
    overlap: bool = True,
    would_clarify_family: bool = False,
    expects_family_clarification: bool = False,
    would_clarify_question_type: bool = False,
    expects_question_type_clarification: bool = False,
    latency_seconds: float = 1.0,
    token_usage: dict[str, TokenUsage] | None = None,
) -> IntentRowResult:
    return IntentRowResult(
        query="q",
        resolved_family=None,
        expected_product_family=None,
        expected_family_in_catalog=True,
        family_correct=family_correct,
        detected_question_types=[],
        expected_question_types=[],
        question_types_exact_match=exact,
        question_types_overlap=overlap,
        would_clarify_family=would_clarify_family,
        expects_family_clarification=expects_family_clarification,
        would_clarify_question_type=would_clarify_question_type,
        expects_question_type_clarification=expects_question_type_clarification,
        latency_seconds=latency_seconds,
        token_usage=token_usage or {},
    )


def test_summarize_computes_accuracy_precision_recall_and_rates():
    results = [
        # True positive family clarification.
        _result(would_clarify_family=True, expects_family_clarification=True, latency_seconds=1.0),
        # False positive: clarified a query that names its family.
        _result(family_correct=False, would_clarify_family=True, latency_seconds=2.0),
        # False negative: guessed where it should have asked.
        _result(family_correct=False, expects_family_clarification=True, latency_seconds=3.0),
        _result(exact=False, would_clarify_question_type=True, latency_seconds=4.0),
        _result(
            exact=False,
            overlap=False,
            expects_question_type_clarification=True,
            latency_seconds=10.0,
        ),
    ]

    report = summarize(results)

    assert report.row_count == 5
    assert report.family_accuracy == pytest.approx(3 / 5)
    assert report.question_type_exact_accuracy == pytest.approx(3 / 5)
    assert report.question_type_overlap_accuracy == pytest.approx(4 / 5)
    assert report.family_clarification_precision == pytest.approx(1 / 2)
    assert report.family_clarification_recall == pytest.approx(1 / 2)
    assert report.question_type_clarification_rate == pytest.approx(1 / 5)
    assert report.expected_question_type_clarification_rate == pytest.approx(1 / 5)
    assert report.question_type_clarification_agreement == pytest.approx(3 / 5)
    # Nearest rank: an observed latency, never an interpolated one.
    assert report.latency_p50_seconds == 3.0
    assert report.latency_p95_seconds == 10.0


def test_precision_and_recall_are_none_with_nothing_to_divide_by():
    report = summarize([_result(), _result()])

    assert report.family_clarification_precision is None
    assert report.family_clarification_recall is None
    assert "n/a" in format_report(report, [])


def test_an_empty_run_reports_no_numbers_rather_than_dividing_by_zero():
    report = summarize([])

    assert report.family_accuracy is None
    assert report.latency_p50_seconds is None


def test_token_usage_is_summed_per_model():
    report = summarize(
        [
            _result(token_usage={"model-a": TokenUsage(10, 2, 12)}),
            _result(token_usage={"model-a": TokenUsage(5, 1, 6), "model-b": TokenUsage(1, 1, 2)}),
        ]
    )

    assert report.token_usage == {"model-a": TokenUsage(15, 3, 18), "model-b": TokenUsage(1, 1, 2)}


async def test_run_writes_the_report_and_per_row_results_as_json(tmp_path):
    classify = _FakeClassifier(
        {
            "What screws?": _IntentClassification(product_family="MIS Foot Recon", question_type=["system_contents"]),
            "Torque?": _IntentClassification(product_family="Reflex Hybrid", question_type=["product_characteristics"]),
        }
    )
    rows = [
        _row("What screws?", "MIS - Foot Recon System", ["system_contents"]),
        _row("Torque?", "MIS - Foot Recon System", ["product_characteristics"]),
    ]

    results = await run_intent_eval(rows, _FAMILIES, classify)
    report = summarize(results)
    output_path = tmp_path / "results.json"
    write_results(output_path, report, results)

    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["report"]["family_accuracy"] == pytest.approx(0.5)
    assert [row["family_correct"] for row in written["rows"]] == [True, False]
    assert "'Torque?'" in format_report(report, results)  # the miss is listed
    assert "'What screws?'" not in format_report(report, results)


def test_the_real_intent_dataset_has_every_field_the_eval_reads():
    rows = load_intent_dataset(harness.DEFAULT_INTENT_DATASET)

    assert rows
    for row in rows:
        assert last_user_turn(row)
        assert {
            "expected_product_family",
            "expected_question_type",
            "expects_family_clarification",
            "expects_question_type_clarification",
        } <= row.keys()


async def test_the_full_answer_eval_is_still_not_built():
    with pytest.raises(NotImplementedError):
        await harness.run_eval("deterministic", "evals/golden_datasets/reflex-hybrid.jsonl")


def test_the_cli_refuses_to_run_without_an_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(SystemExit) as exit_info:
        harness.main(["intent"])

    assert exit_info.value.code == 2
