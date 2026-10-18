"""Intent-only eval (issue #25, Slice 1): scores what the deterministic
workflow's detect_intent and clarify_question_type decide for a query --
which Product Family, which question types, and whether each node would
pause to ask -- against intent_detection.jsonl's labels. No answer is
generated, so there's no pause to resolve and no documents or vector store
to stand up; the only data it needs is the families list (the catalog
mirror, catalog.repository), the same list detect_intent classifies among.

Every decision here is the workflow's own code, not a copy of it: the
classifier call is deterministic._classify_intent (same prompt, same
structured-output schema, same get_chat_model()), the family pick goes
through _resolve_family, the detected types through
_detected_question_types, and both pause conditions are the nodes' own
_needs_family_clarification/_needs_question_type_clarification. So a
prompt or schema change in the workflow moves this eval's numbers with no
change here.

The classifier is a parameter (defaulting to the real one) so the
comparison and metrics are unit-testable with a fake one and no
OPENAI_API_KEY. Run it through evals/harness.py's `intent` command.
"""
import json
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from langchain_core.callbacks import get_usage_metadata_callback

from agents.workflows.deterministic import (
    _classify_intent,
    _detected_question_types,
    _family_for_turn,
    _IntentClassification,
    _needs_family_clarification,
    _needs_question_type_clarification,
    _resolve_family,
)
from catalog.display_names import catalog_display_names_for
from catalog.repository import ProductFamilyRecord
from config.llm_clients import DecisionUsage, collect_decision_usage

# Called like deterministic._classify_intent: (query, family_names) plus
# the previous_family/previous_message keyword hints (issue #22).
IntentClassifier = Callable[..., Awaitable[_IntentClassification]]


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    # US dollars. Only a decision model (issue #23) reports one: its
    # response carries the cost, while LangChain's chat usage doesn't.
    cost: float = 0.0

    def add(self, other: "TokenUsage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_tokens += other.total_tokens
        self.cost += other.cost


@dataclass
class IntentRowResult:
    query: str
    # The classifier's pick before any clarification, as the display name
    # detect_intent would record -- None when it named no real family.
    resolved_family: str | None
    expected_product_family: str | None
    # False when the row expects a family the catalog mirror doesn't have
    # (renamed out from under the dataset, or not seeded): such a row can
    # never score correct, and the report calls the count out separately.
    expected_family_in_catalog: bool
    family_correct: bool
    detected_question_types: list[str]
    expected_question_types: list[str]
    question_types_exact_match: bool
    question_types_overlap: bool
    would_clarify_family: bool
    expects_family_clarification: bool
    would_clarify_question_type: bool
    expects_question_type_clarification: bool
    latency_seconds: float
    # Keyed by model name, as LangChain's usage metadata reports it. Empty
    # when the provider reported none (or a fake classifier ran).
    token_usage: dict[str, TokenUsage] = field(default_factory=dict)


@dataclass
class IntentReport:
    row_count: int
    family_accuracy: float | None
    rows_expecting_unmirrored_family: int
    question_type_exact_accuracy: float | None
    question_type_overlap_accuracy: float | None
    # None when there's nothing to divide by: no row was predicted to
    # clarify (precision) or none is expected to (recall).
    family_clarification_precision: float | None
    family_clarification_recall: float | None
    question_type_clarification_rate: float | None
    expected_question_type_clarification_rate: float | None
    question_type_clarification_agreement: float | None
    latency_p50_seconds: float | None
    latency_p95_seconds: float | None
    token_usage: dict[str, TokenUsage]


def load_intent_dataset(dataset_path: str | Path) -> list[dict]:
    with open(dataset_path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def last_user_turn(row: dict) -> str:
    """The query detect_intent would classify: a turn's `query` is always
    the latest user message (backend/api/routes/chat.py).
    """
    return next(turn["content"] for turn in reversed(row["turns"]) if turn["role"] == "user")


def _display_name(family: ProductFamilyRecord | None) -> str | None:
    return family.display_name if family else None


def _token_usage_by_model(usage_metadata: dict) -> dict[str, TokenUsage]:
    return {
        model_name: TokenUsage(
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        )
        for model_name, usage in usage_metadata.items()
    }


def _add_decision_usage(
    token_usage: dict[str, TokenUsage], decision_usage: dict[str, list[DecisionUsage]]
) -> dict[str, TokenUsage]:
    for model_name, usages in decision_usage.items():
        for usage in usages:
            token_usage.setdefault(model_name, TokenUsage()).add(
                TokenUsage(
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    total_tokens=usage.input_tokens + usage.output_tokens,
                    cost=usage.cost,
                )
            )
    return token_usage


async def evaluate_row(
    row: dict, families: list[ProductFamilyRecord], classify: IntentClassifier = _classify_intent
) -> IntentRowResult:
    display_names = catalog_display_names_for(families)
    family_names = [family.display_name for family in families]
    user_messages = [turn["content"] for turn in row["turns"] if turn["role"] == "user"]
    query = user_messages[-1]

    # A multi-turn row (issue #22) replays its earlier turns first, so the
    # scored turn gets the same hints detect_intent would: the previous
    # family and the previous user message. An earlier turn that would have
    # asked for its family carries none forward -- the eval can't answer a
    # clarification, the same as a reply naming no real family. Only the
    # scored turn counts toward latency and token usage.
    previous_family: ProductFamilyRecord | None = None
    previous_message: str | None = None
    for earlier_message in user_messages[:-1]:
        earlier_classification = await classify(
            earlier_message,
            family_names,
            previous_family=_display_name(previous_family),
            previous_message=previous_message,
        )
        previous_family = _family_for_turn(
            earlier_classification, families, display_names, previous_family
        )
        previous_message = earlier_message

    started = time.perf_counter()
    with get_usage_metadata_callback() as usage_callback, collect_decision_usage() as decision_usage:
        classification = await classify(
            query,
            family_names,
            previous_family=_display_name(previous_family),
            previous_message=previous_message,
        )
    latency_seconds = time.perf_counter() - started

    family = _family_for_turn(classification, families, display_names, previous_family)
    detected_question_types = _detected_question_types(classification.question_type)

    expected_family_name = row["expected_product_family"]
    expected_family = _resolve_family(expected_family_name, families, display_names)
    if expected_family_name is None:
        family_correct = family is None
    else:
        family_correct = (
            expected_family is not None and family is not None and family.id == expected_family.id
        )

    expected_question_types = row["expected_question_type"]
    return IntentRowResult(
        query=query,
        resolved_family=family.display_name if family else None,
        expected_product_family=expected_family_name,
        expected_family_in_catalog=expected_family_name is None or expected_family is not None,
        family_correct=family_correct,
        detected_question_types=detected_question_types,
        expected_question_types=expected_question_types,
        question_types_exact_match=set(detected_question_types) == set(expected_question_types),
        question_types_overlap=bool(set(detected_question_types) & set(expected_question_types)),
        would_clarify_family=_needs_family_clarification(family, families),
        expects_family_clarification=row["expects_family_clarification"],
        would_clarify_question_type=_needs_question_type_clarification(detected_question_types),
        expects_question_type_clarification=row["expects_question_type_clarification"],
        latency_seconds=latency_seconds,
        token_usage=_add_decision_usage(
            _token_usage_by_model(usage_callback.usage_metadata), decision_usage
        ),
    )


async def run_intent_eval(
    rows: list[dict],
    families: list[ProductFamilyRecord],
    classify: IntentClassifier = _classify_intent,
) -> list[IntentRowResult]:
    # One row at a time, not gathered: latency is part of the report, and
    # concurrent calls would measure rate limiting as much as the model.
    return [await evaluate_row(row, families, classify) for row in rows]


def _fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _percentile(values: list[float], fraction: float) -> float | None:
    """Nearest-rank percentile: always a latency that was actually observed,
    which reads more honestly than an interpolated one on ~35 rows.
    """
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def summarize(results: list[IntentRowResult]) -> IntentReport:
    row_count = len(results)
    predicted_family_clarifications = [result for result in results if result.would_clarify_family]
    expected_family_clarifications = [
        result for result in results if result.expects_family_clarification
    ]
    true_family_clarifications = [
        result for result in predicted_family_clarifications if result.expects_family_clarification
    ]

    token_usage: dict[str, TokenUsage] = {}
    for result in results:
        for model_name, usage in result.token_usage.items():
            token_usage.setdefault(model_name, TokenUsage()).add(usage)

    latencies = [result.latency_seconds for result in results]
    return IntentReport(
        row_count=row_count,
        family_accuracy=_fraction(sum(result.family_correct for result in results), row_count),
        rows_expecting_unmirrored_family=sum(
            not result.expected_family_in_catalog for result in results
        ),
        question_type_exact_accuracy=_fraction(
            sum(result.question_types_exact_match for result in results), row_count
        ),
        question_type_overlap_accuracy=_fraction(
            sum(result.question_types_overlap for result in results), row_count
        ),
        family_clarification_precision=_fraction(
            len(true_family_clarifications), len(predicted_family_clarifications)
        ),
        family_clarification_recall=_fraction(
            len(true_family_clarifications), len(expected_family_clarifications)
        ),
        question_type_clarification_rate=_fraction(
            sum(result.would_clarify_question_type for result in results), row_count
        ),
        expected_question_type_clarification_rate=_fraction(
            sum(result.expects_question_type_clarification for result in results), row_count
        ),
        question_type_clarification_agreement=_fraction(
            sum(
                result.would_clarify_question_type == result.expects_question_type_clarification
                for result in results
            ),
            row_count,
        ),
        latency_p50_seconds=_percentile(latencies, 0.50),
        latency_p95_seconds=_percentile(latencies, 0.95),
        token_usage=token_usage,
    )


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _seconds(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}s"


def format_report(report: IntentReport, results: list[IntentRowResult]) -> str:
    lines = [
        f"Intent eval: {report.row_count} rows",
        f"  Family accuracy:                    {_percent(report.family_accuracy)}",
        f"  Question type accuracy (exact set): {_percent(report.question_type_exact_accuracy)}",
        f"  Question type accuracy (overlap):   {_percent(report.question_type_overlap_accuracy)}",
        f"  Family clarification precision:     {_percent(report.family_clarification_precision)}",
        f"  Family clarification recall:        {_percent(report.family_clarification_recall)}",
        (
            "  Question type clarification rate:   "
            f"{_percent(report.question_type_clarification_rate)} "
            f"(expected {_percent(report.expected_question_type_clarification_rate)}, "
            f"rows agreeing {_percent(report.question_type_clarification_agreement)})"
        ),
        (
            f"  Latency p50 / p95:                  {_seconds(report.latency_p50_seconds)} / "
            f"{_seconds(report.latency_p95_seconds)}"
        ),
    ]
    if report.token_usage:
        for model_name, usage in report.token_usage.items():
            cost = (
                f", ${usage.cost:.4f} (${usage.cost / report.row_count:.5f} per turn)"
                if usage.cost and report.row_count
                else ""
            )
            lines.append(
                f"  Tokens ({model_name}): {usage.input_tokens} in / "
                f"{usage.output_tokens} out / {usage.total_tokens} total{cost}"
            )
    else:
        lines.append("  Tokens: not reported by the provider")
    # A chat model's cost isn't computed: LangChain's usage metadata carries
    # token counts but no prices, and this repo keeps no price table to
    # multiply by. A decision model's response reports its own cost.
    if report.rows_expecting_unmirrored_family:
        lines.append(
            f"  WARNING: {report.rows_expecting_unmirrored_family} row(s) expect a family the "
            "catalog mirror doesn't have -- they can't score correct until it's seeded or the "
            "dataset is relabelled"
        )

    misses = [
        result
        for result in results
        if not result.family_correct
        or not result.question_types_exact_match
        or result.would_clarify_family != result.expects_family_clarification
        or result.would_clarify_question_type != result.expects_question_type_clarification
    ]
    if misses:
        lines.append("")
        lines.append(f"Rows with any mismatch ({len(misses)}):")
        for result in misses:
            lines.append(f"  {result.query!r}")
            lines.append(
                f"    family: {result.resolved_family!r} (expected "
                f"{result.expected_product_family!r})"
            )
            lines.append(
                f"    question types: {result.detected_question_types} (expected "
                f"{result.expected_question_types})"
            )
            lines.append(
                f"    clarifies family/question type: {result.would_clarify_family}/"
                f"{result.would_clarify_question_type} (expected "
                f"{result.expects_family_clarification}/"
                f"{result.expects_question_type_clarification})"
            )
    return "\n".join(lines)


def write_results(
    output_path: str | Path,
    report: IntentReport,
    results: list[IntentRowResult],
    models: dict[str, dict] | None = None,
) -> None:
    """`models` is the harness's slot -> {model_id, reasoning_effort,
    source} for the run (issue #24), so a results file says what it scored.
    """
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "models": models or {},
                "report": asdict(report),
                "rows": [asdict(result) for result in results],
            },
            file,
            indent=2,
            ensure_ascii=False,
        )
        file.write("\n")
