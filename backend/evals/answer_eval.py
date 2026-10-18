"""Full answer eval (issue #25, Slice 2): runs the per-Product-Family QA
golden datasets through a registered workflow, by registry name, and scores
each final answer -- the shared judge (agents/judge.py, the rubric human
feedback uses) against the row's expected_answer, the workflow's own fact
check, and citation recall against the row's expected_citations.

Each row is one fresh thread on an in-memory checkpointer (LangGraph's
InMemorySaver), never memory/checkpointer.py's Postgres one, so a run
writes no chat threads or checkpoints into the database. Retrieval and the
catalog still read the configured DATABASE_URL and Neo4j, read-only.

Pause policy (b), settled on the issue: a turn that pauses on a
clarification interrupt() resumes with Command(resume=<reply>), the same
way POST /chat/resume does, with a reply scripted from the row's own
labels. Which pause it is comes from the node that paused, not the
question's wording:

- detect_intent (family): the row's product_family. _resolve_family takes
  the catalog name as readily as a clicked display name.
- clarify_question_type (question type): the option text of the row's
  first question_type when it's offered, else the slug itself, which
  _match_question_type_reply also accepts with no model call. A row with
  no question type answers "Something else" (unclassified).
- request_clarification (synonym/term): the first offered option. The
  interrupt offers none today -- its question is built from
  synonym_ambiguity -- so the first canonical term of the first ambiguous
  word in that state is the option it would have offered.

A row's optional clarification_replies ({"family": ..., "question_type":
..., "synonym": ...}, written by build_dataset.py) overrides any of these.
A row resumes at most MAX_RESUMES_PER_ROW times; one still paused after
that scores as a failure, with the question it stopped on as the reason.
Every pause is recorded on the row's result with its question and the
reply used.

The judge is a parameter (defaulting to the real one) so the run, the
scripted replies and the metrics are unit-testable with a fake one and no
OPENAI_API_KEY. Run it through evals/harness.py's `answers` command.
"""
import json
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

import agents.workflows  # noqa: F401 - registers every workflow with agents.registry
from agents.citations import extract_citations
from agents.judge import judge_answer
from agents.question_types import QUESTION_TYPES
from agents.registry import get_workflow
from agents.state import EvalScores, RetrievedPassage
from agents.tools.document_lookup import names_document
from agents.workflows.deterministic import SOMETHING_ELSE, _question_type_option
from documents.repository import DocumentRecord
from evals.intent_eval import (
    TokenUsage,
    _fraction,
    _percent,
    _percentile,
    _seconds,
    _token_usage_by_model,
)

# Called like agents/judge.py's judge_answer: (query, retrieved, answer)
# plus the reference_answer keyword.
AnswerJudge = Callable[..., Awaitable[EvalScores]]

INTENT_DATASET_NAME = "intent_detection.jsonl"

# The deterministic workflow can pause at most three times in a turn (family,
# question type, synonym -- each once), so three resumes is enough for any
# real row; a fourth pause means something is looping.
MAX_RESUMES_PER_ROW = 3

# Same value and reason as api/routes/chat.py's _RECURSION_LIMIT: LangGraph's
# default of 25 is a library constant, not a turn budget.
_RECURSION_LIMIT = 50

# A row with any judge dimension under this is listed in the report's table
# of failures and low scores.
LOW_SCORE_THRESHOLD = 0.6

JUDGE_DIMENSIONS = ("faithfulness", "relevance", "style", "citation")

# The node that paused -> which kind of pause it is, and so which scripted
# reply it gets. Keyed by node rather than by the question's wording, so a
# reworded question still gets the right reply.
FAMILY_PAUSE = "family"
QUESTION_TYPE_PAUSE = "question_type"
SYNONYM_PAUSE = "synonym"
PAUSE_KINDS = (FAMILY_PAUSE, QUESTION_TYPE_PAUSE, SYNONYM_PAUSE)
_PAUSE_KIND_BY_NODE = {
    "detect_intent": FAMILY_PAUSE,
    "clarify_question_type": QUESTION_TYPE_PAUSE,
    "request_clarification": SYNONYM_PAUSE,
}


@dataclass
class Pause:
    # One of PAUSE_KINDS, or "unknown" for a node this eval has no
    # scripted reply for (it then gets the first offered option).
    kind: str
    node: str
    question: str
    options: list[str]
    # None only for the pause that hit MAX_RESUMES_PER_ROW, which nothing
    # answered.
    reply: str | None
    # "default" (scripted from the row's labels), "override" (the row's
    # clarification_replies), or "none" (unanswered).
    reply_source: str


@dataclass
class AnswerRowResult:
    dataset: str
    query: str
    product_family: str | None
    question_types: list[str]
    pauses: list[Pause]
    # None when the row produced no answer: it errored, or was still paused
    # after MAX_RESUMES_PER_ROW resumes. failure_reason then says which.
    answer: str | None
    failure_reason: str | None
    # None when the judge didn't run (no answer, or the judge call failed).
    judge_scores: dict[str, float] | None
    # The workflow's own self_eval result on the final answer
    # (agents/fact_check.py, BaseAgentState.fact_check_issues).
    fact_check_issues: list[str]
    expected_citations: list[str]
    # The filenames of the documents the answer cited.
    cited_documents: list[str]
    matched_expected_citations: list[str]
    # Expected citations that name no ingested document at all: they can
    # never be matched until that document is ingested or the row
    # relabelled, so the report calls the count out separately.
    expected_citations_naming_no_document: list[str]
    # None when the row expects no citations.
    citation_recall: float | None
    # The workflow's time across every resume, not the judge's.
    latency_seconds: float
    # Keyed by model name, as LangChain's usage metadata reports it. The
    # workflow's calls and the judge's are kept apart, so a model shared by
    # both isn't double-counted against the workflow.
    token_usage: dict[str, TokenUsage] = field(default_factory=dict)
    judge_token_usage: dict[str, TokenUsage] = field(default_factory=dict)


@dataclass
class AnswerReport:
    workflow_name: str
    row_count: int
    answered_count: int
    failure_count: int
    # Averaged over judged rows; None for a dimension with none.
    judge_averages: dict[str, float | None]
    # Of answered rows, the share whose final answer still had fact-check
    # issues after the workflow's own correction pass.
    fact_check_issue_rate: float | None
    # Mean per-row recall over answered rows that expect any citation.
    citation_recall: float | None
    expected_citations_naming_no_document: int
    # Share of all rows that paused at least once on each kind.
    pause_rate_by_kind: dict[str, float | None]
    pause_rate: float | None
    latency_p50_seconds: float | None
    latency_p95_seconds: float | None
    token_usage: dict[str, TokenUsage]
    judge_token_usage: dict[str, TokenUsage]


def qa_dataset_paths(dataset_path: str | Path) -> list[Path]:
    """A single file as given, or every per-family QA file in a directory
    -- every *.jsonl there except intent_detection.jsonl, which has no
    expected answers.
    """
    path = Path(dataset_path)
    if path.is_dir():
        return sorted(
            candidate for candidate in path.glob("*.jsonl") if candidate.name != INTENT_DATASET_NAME
        )
    return [path]


def load_qa_dataset(dataset_path: str | Path) -> list[dict]:
    """Every row of every QA file at `dataset_path`, each tagged with the
    file it came from as `dataset`.
    """
    rows: list[dict] = []
    for path in qa_dataset_paths(dataset_path):
        with open(path, encoding="utf-8") as file:
            rows.extend({**json.loads(line), "dataset": path.name} for line in file if line.strip())
    return rows


def _question_type_reply(row: dict, options: list[str]) -> str:
    question_types = row.get("question_type") or []
    if not question_types:
        return SOMETHING_ELSE
    slug = question_types[0]
    question_type = next((candidate for candidate in QUESTION_TYPES if candidate.name == slug), None)
    if question_type is not None and _question_type_option(question_type) in options:
        return _question_type_option(question_type)
    return slug


def _synonym_reply(options: list[str], state_values: dict) -> str:
    if options:
        return options[0]
    ambiguity = state_values.get("synonym_ambiguity") or {}
    for canonical_terms in ambiguity.values():
        if canonical_terms:
            return canonical_terms[0]
    return ""


def scripted_reply(kind: str, row: dict, options: list[str], state_values: dict) -> tuple[str, str]:
    """The reply to a pause of `kind` and where it came from ("override" or
    "default") -- see the module docstring for each kind's default.
    """
    override = (row.get("clarification_replies") or {}).get(kind)
    if override is not None:
        return override, "override"
    if kind == FAMILY_PAUSE:
        return row.get("product_family") or "", "default"
    if kind == QUESTION_TYPE_PAUSE:
        return _question_type_reply(row, options), "default"
    if kind == SYNONYM_PAUSE:
        return _synonym_reply(options, state_values), "default"
    return (options[0] if options else ""), "default"


def _pending_interrupt(snapshot) -> tuple[str, dict] | None:
    """(node, interrupt payload) for the graph's pending interrupt(), or
    None when the run finished.
    """
    for task in snapshot.tasks:
        if task.interrupts:
            return task.name, task.interrupts[0].value or {}
    return None


def _turn_inputs(query: str, thread_id: str) -> dict:
    """A fresh turn's inputs, mirroring api/routes/chat.py's _stream_chat --
    including the per-turn resets, though on a fresh thread there's nothing
    to reset.
    """
    return {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "user_id": "eval",
        "thread_id": thread_id,
        "search_query": query,
        "clarification_rounds": 0,
        "clarification_reply": None,
        "correction_rounds": 0,
        "fact_check_issues": [],
    }


def _context_passage(chunk_id: str, text: str) -> RetrievedPassage:
    return RetrievedPassage(
        chunk_id=chunk_id,
        document_id="",
        text=text,
        score=0.0,
        document_type=None,
        document_type_id=None,
        tray_ids=[],
    )


def judge_context(state_values: dict) -> list[RetrievedPassage]:
    """What the workflow's final answer was written from, as the judge's
    context: the deterministic workflow's reranked passages (what generate
    saw, not the wider retrieved pool) plus its catalog facts, or a
    react_agent turn's tool results. The judge's prompt names
    "catalog-facts"/"tool-result" ids as where a SKU-named fact comes from.
    """
    if "reranked" in state_values:
        passages = list(state_values.get("reranked") or [])
    else:
        passages = list(state_values.get("retrieved") or [])
    if state_values.get("aggregated_facts"):
        passages.append(_context_passage("catalog-facts", state_values["aggregated_facts"]))
    for message in state_values.get("scratchpad") or []:
        if isinstance(message, ToolMessage):
            passages.append(_context_passage(f"tool-result:{message.name}", str(message.content)))
    return passages


def _cited_documents(answer: str, documents_by_id: dict[str, DocumentRecord]) -> list[DocumentRecord]:
    cited: dict[str, DocumentRecord] = {}
    for citation in extract_citations(answer):
        document_id = citation.rpartition("#")[0]
        if document_id in documents_by_id:
            cited.setdefault(document_id, documents_by_id[document_id])
    return list(cited.values())


async def evaluate_row(
    row: dict,
    workflow_name: str,
    documents: list[DocumentRecord],
    judge: AnswerJudge | None = None,
) -> AnswerRowResult:
    # Looked up at call time rather than bound as the default, so a test can
    # swap the module's judge_answer under harness.run_eval.
    judge = judge or judge_answer
    # A fresh checkpointer per row, not only a fresh thread: nothing a row
    # leaves behind can reach the next one.
    graph = get_workflow(workflow_name, InMemorySaver())
    thread_id = f"eval:{uuid.uuid4().hex}"
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": _RECURSION_LIMIT}
    query = row["query"]

    pauses: list[Pause] = []
    failure_reason: str | None = None
    state_values: dict = {}
    run_input: dict | Command = _turn_inputs(query, thread_id)
    started = time.perf_counter()
    with get_usage_metadata_callback() as usage_callback:
        try:
            while True:
                await graph.ainvoke(run_input, config)
                snapshot = await graph.aget_state(config)
                state_values = snapshot.values
                pending = _pending_interrupt(snapshot)
                if pending is None:
                    break
                node, payload = pending
                kind = _PAUSE_KIND_BY_NODE.get(node, "unknown")
                question = payload.get("question", "")
                options = list(payload.get("options") or [])
                if len(pauses) >= MAX_RESUMES_PER_ROW:
                    pauses.append(Pause(kind, node, question, options, None, "none"))
                    failure_reason = (
                        f"still paused after {MAX_RESUMES_PER_ROW} resumes, at {node}: {question!r}"
                    )
                    break
                reply, reply_source = scripted_reply(kind, row, options, state_values)
                pauses.append(Pause(kind, node, question, options, reply, reply_source))
                run_input = Command(resume=reply)
        except Exception as error:  # noqa: BLE001 - one row's failure is a result, not the end of a paid run
            failure_reason = f"workflow raised {type(error).__name__}: {error}"
    latency_seconds = time.perf_counter() - started

    answer = None if failure_reason else state_values.get("answer")
    if failure_reason is None and not answer:
        failure_reason = "workflow finished without an answer"

    expected_citations = row.get("expected_citations") or []
    naming_no_document = [
        expected
        for expected in expected_citations
        if not any(names_document(expected, document) for document in documents)
    ]
    cited = _cited_documents(answer or "", {str(document.id): document for document in documents})
    matched = [
        expected
        for expected in expected_citations
        if any(names_document(expected, document) for document in cited)
    ]

    judge_scores: dict[str, float] | None = None
    judge_token_usage: dict[str, TokenUsage] = {}
    if answer:
        try:
            with get_usage_metadata_callback() as judge_usage_callback:
                judge_scores = dict(
                    await judge(
                        query,
                        judge_context(state_values),
                        answer,
                        reference_answer=row.get("expected_answer") or None,
                    )
                )
            judge_token_usage = _token_usage_by_model(judge_usage_callback.usage_metadata)
        except Exception as error:  # noqa: BLE001 - keep the answer and its other scores
            failure_reason = f"judge raised {type(error).__name__}: {error}"

    return AnswerRowResult(
        dataset=row.get("dataset", ""),
        query=query,
        product_family=row.get("product_family"),
        question_types=row.get("question_type") or [],
        pauses=pauses,
        answer=answer,
        failure_reason=failure_reason,
        judge_scores=judge_scores,
        fact_check_issues=list(state_values.get("fact_check_issues") or []) if answer else [],
        expected_citations=expected_citations,
        cited_documents=[document.filename for document in cited],
        matched_expected_citations=matched,
        expected_citations_naming_no_document=naming_no_document,
        citation_recall=_fraction(len(matched), len(expected_citations)) if answer else None,
        latency_seconds=latency_seconds,
        token_usage=_token_usage_by_model(usage_callback.usage_metadata),
        judge_token_usage=judge_token_usage,
    )


async def run_answer_eval(
    rows: list[dict],
    workflow_name: str,
    documents: list[DocumentRecord],
    judge: AnswerJudge | None = None,
) -> list[AnswerRowResult]:
    # One row at a time, like the intent eval: latency is part of the
    # report, and concurrent turns would measure rate limiting too.
    return [await evaluate_row(row, workflow_name, documents, judge) for row in rows]


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _summed_usage(usages: list[dict[str, TokenUsage]]) -> dict[str, TokenUsage]:
    total: dict[str, TokenUsage] = {}
    for usage_by_model in usages:
        for model_name, usage in usage_by_model.items():
            total.setdefault(model_name, TokenUsage()).add(usage)
    return total


def summarize(workflow_name: str, results: list[AnswerRowResult]) -> AnswerReport:
    row_count = len(results)
    answered = [result for result in results if result.answer]
    judged = [result for result in results if result.judge_scores is not None]
    recalls = [result.citation_recall for result in answered if result.citation_recall is not None]
    latencies = [result.latency_seconds for result in results]
    return AnswerReport(
        workflow_name=workflow_name,
        row_count=row_count,
        answered_count=len(answered),
        failure_count=sum(result.failure_reason is not None for result in results),
        judge_averages={
            dimension: _mean(
                [result.judge_scores[dimension] for result in judged if dimension in result.judge_scores]
            )
            for dimension in JUDGE_DIMENSIONS
        },
        fact_check_issue_rate=_fraction(
            sum(bool(result.fact_check_issues) for result in answered), len(answered)
        ),
        citation_recall=_mean(recalls),
        expected_citations_naming_no_document=len(
            {
                expected
                for result in results
                for expected in result.expected_citations_naming_no_document
            }
        ),
        pause_rate_by_kind={
            kind: _fraction(
                sum(any(pause.kind == kind for pause in result.pauses) for result in results), row_count
            )
            for kind in PAUSE_KINDS
        },
        pause_rate=_fraction(sum(bool(result.pauses) for result in results), row_count),
        latency_p50_seconds=_percentile(latencies, 0.50),
        latency_p95_seconds=_percentile(latencies, 0.95),
        token_usage=_summed_usage([result.token_usage for result in results]),
        judge_token_usage=_summed_usage([result.judge_token_usage for result in results]),
    )


def _is_low_score(result: AnswerRowResult) -> bool:
    return result.judge_scores is not None and any(
        score < LOW_SCORE_THRESHOLD for score in result.judge_scores.values()
    )


def _score(scores: dict[str, float] | None, dimension: str) -> str:
    if scores is None or dimension not in scores:
        return "-"
    return f"{scores[dimension]:.2f}"


def _truncated(text: str, length: int) -> str:
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= length else collapsed[: length - 1] + "…"


def _token_lines(label: str, usage_by_model: dict[str, TokenUsage]) -> list[str]:
    if not usage_by_model:
        return [f"  {label} tokens: not reported by the provider"]
    return [
        f"  {label} tokens ({model_name}): {usage.input_tokens} in / "
        f"{usage.output_tokens} out / {usage.total_tokens} total"
        for model_name, usage in usage_by_model.items()
    ]


def format_report(report: AnswerReport, results: list[AnswerRowResult]) -> str:
    averages = report.judge_averages
    lines = [
        (
            f"Answer eval ({report.workflow_name}): {report.row_count} rows, "
            f"{report.answered_count} answered, {report.failure_count} failed"
        ),
        "  Judge averages:",
        *(
            f"    {dimension:<13} "
            + ("n/a" if averages[dimension] is None else f"{averages[dimension]:.2f}")
            for dimension in JUDGE_DIMENSIONS
        ),
        f"  Fact-check issue rate:  {_percent(report.fact_check_issue_rate)}",
        f"  Citation recall:        {_percent(report.citation_recall)}",
        (
            f"  Pause rate:             {_percent(report.pause_rate)} ("
            + ", ".join(
                f"{kind} {_percent(rate)}" for kind, rate in report.pause_rate_by_kind.items()
            )
            + ")"
        ),
        (
            f"  Latency p50 / p95:      {_seconds(report.latency_p50_seconds)} / "
            f"{_seconds(report.latency_p95_seconds)}"
        ),
        *_token_lines("Workflow", report.token_usage),
        *_token_lines("Judge", report.judge_token_usage),
    ]
    # Cost isn't computed, for the same reason as the intent eval: usage
    # metadata carries token counts but no prices.
    if report.expected_citations_naming_no_document:
        lines.append(
            f"  WARNING: {report.expected_citations_naming_no_document} expected citation(s) name "
            "no ingested document -- they can't be matched until it's ingested or the dataset is "
            "relabelled (see expected_citations_naming_no_document in --output)"
        )

    flagged = [result for result in results if result.failure_reason or _is_low_score(result)]
    if flagged:
        lines.append("")
        lines.append(
            f"Failures and low scores (any judge score under {LOW_SCORE_THRESHOLD}) ({len(flagged)}):"
        )
        header = (
            f"  {'Row':<52} {'Faith':>5} {'Relev':>5} {'Style':>5} {'Cite':>5} "
            f"{'Facts':>5} {'Recall':>6} Pauses"
        )
        lines.append(header)
        for result in flagged:
            recall = "-" if result.citation_recall is None else f"{result.citation_recall:.0%}"
            pause_kinds = ",".join(pause.kind for pause in result.pauses) or "-"
            lines.append(
                f"  {_truncated(result.query, 52):<52} "
                f"{_score(result.judge_scores, 'faithfulness'):>5} "
                f"{_score(result.judge_scores, 'relevance'):>5} "
                f"{_score(result.judge_scores, 'style'):>5} "
                f"{_score(result.judge_scores, 'citation'):>5} "
                f"{len(result.fact_check_issues):>5} {recall:>6} {pause_kinds}"
            )
            if result.failure_reason:
                lines.append(f"    FAILED: {result.failure_reason}")
            for pause in result.pauses:
                lines.append(f"    paused ({pause.kind}): {pause.question!r} -> {pause.reply!r}")
    return "\n".join(lines)


def write_results(
    output_path: str | Path, report: AnswerReport, results: list[AnswerRowResult]
) -> None:
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            {"report": asdict(report), "rows": [asdict(result) for result in results]},
            file,
            indent=2,
            ensure_ascii=False,
        )
        file.write("\n")
