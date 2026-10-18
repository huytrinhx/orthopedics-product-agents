"""Offline eval entry point (issue #25). Two slices:

- `intent` (Slice 1, evals/intent_eval.py): runs intent_detection.jsonl's
  queries through the deterministic workflow's own intent classifier and
  scores the family, the question types, and whether each clarification
  would fire. Needs OPENAI_API_KEY and a DATABASE_URL whose catalog mirror
  is seeded (ingestion.seed_master_catalog); no documents or vector store.
- `answers` / run_eval (Slice 2, evals/answer_eval.py): a full answer eval
  of a registered workflow over the per-family QA files, scored by the
  shared judge (agents/judge.py), the workflow's fact check, and citation
  recall. A row that pauses on a clarification resumes with a reply
  scripted from its own labels. Needs OPENAI_API_KEY, a DATABASE_URL with
  ingested documents and the catalog mirror, and Neo4j; it only reads
  them, and keeps its threads in memory.

Not run in CI: a real model call costs money, so an admin runs it by hand
(issue #25 comment; agents.md). Results are a CLI report, optionally also
written as JSON with --output; nothing is persisted.

Usage (from backend/):
    python -m evals.harness intent
    python -m evals.harness intent --dataset evals/golden_datasets/intent_detection.jsonl \\
        --output intent-results.json
    python -m evals.harness answers --workflow deterministic
    python -m evals.harness answers --workflow react_agent \\
        --dataset evals/golden_datasets/reflex-hybrid.jsonl --output answer-results.json
"""
import argparse
import asyncio
import os
from dataclasses import asdict
from pathlib import Path

from agents.registry import is_functional, list_workflows
from catalog.repository import list_families_with_trays
from config.llm_clients import configured_chat_model, get_chat_model
from documents.repository import list_documents
from evals import answer_eval
from evals.intent_eval import (
    format_report,
    load_intent_dataset,
    run_intent_eval,
    summarize,
    write_results,
)

DEFAULT_INTENT_DATASET = Path(__file__).parent / "golden_datasets" / "intent_detection.jsonl"
# A directory: every per-family QA file in it (answer_eval.qa_dataset_paths).
DEFAULT_ANSWER_DATASET = Path(__file__).parent / "golden_datasets"


async def _ingested_documents() -> list:
    # Only documents an answer could have cited: an unfinished one has no
    # chunks to retrieve.
    return [document for document in await list_documents() if document.status == "done"]


async def run_eval(workflow_name: str, dataset_path: str) -> dict:
    """The full answer eval (issue #25 Slice 2) of `workflow_name` over the
    QA file or directory at `dataset_path`: the report and per-row results,
    as JSON-ready dicts.
    """
    rows = answer_eval.load_qa_dataset(dataset_path)
    results = await answer_eval.run_answer_eval(rows, workflow_name, await _ingested_documents())
    report = answer_eval.summarize(workflow_name, results)
    return {"report": asdict(report), "rows": [asdict(result) for result in results]}


async def _run_intent(dataset_path: Path, output_path: Path | None) -> None:
    rows = load_intent_dataset(dataset_path)
    families = await list_families_with_trays()
    if not families:
        # Same situation detect_intent logs and carries on through: with
        # nothing to pick between, the family clarification can never fire.
        print(
            "WARNING: no product families mirrored -- run ingestion.seed_master_catalog first; "
            "every family will resolve to None and never clarify"
        )

    # Extension point for `--model <slot>=<model_id>` (issue #25 Slice 1):
    # the slot names come from #24's per-node models, which aren't built, so
    # every call goes through get_chat_model() (OPENAI_CHAT_MODEL) for now.
    # Once #24 lands, parse the overrides here, apply them to the classifier
    # call, and print each slot's model and effort below.
    chat_model = get_chat_model()
    print(
        f"Model: {chat_model.model_name} "
        f"(reasoning effort: {chat_model.reasoning_effort or 'provider default'})"
    )
    print(f"Dataset: {dataset_path} ({len(rows)} rows), {len(families)} families mirrored")
    print()

    results = await run_intent_eval(rows, families)
    report = summarize(results)
    print(format_report(report, results))
    if output_path is not None:
        write_results(output_path, report, results)
        print()
        print(f"Wrote per-row results to {output_path}")


async def _run_answers(workflow_name: str, dataset_path: Path, output_path: Path | None) -> None:
    rows = answer_eval.load_qa_dataset(dataset_path)
    documents = await _ingested_documents()
    if not documents:
        print(
            "WARNING: no ingested documents -- retrieval will find nothing and no expected "
            "citation can match"
        )

    # Extension point for `--model <slot>=<model_id>` (issue #24): once the
    # intent command's overrides land, accept the same flags here, apply
    # them to the workflow's nodes and the judge, and print each slot's
    # model and effort below. Until then every call uses CHAT_MODEL.
    print(f"Workflow: {workflow_name}, model: {configured_chat_model()}")
    dataset_names = ", ".join(path.name for path in answer_eval.qa_dataset_paths(dataset_path))
    print(f"Dataset: {dataset_names} ({len(rows)} rows), {len(documents)} documents ingested")
    print()

    results = await answer_eval.run_answer_eval(rows, workflow_name, documents)
    report = answer_eval.summarize(workflow_name, results)
    print(answer_eval.format_report(report, results))
    if output_path is not None:
        answer_eval.write_results(output_path, report, results)
        print()
        print(f"Wrote per-row results to {output_path}")


def _add_answers_command(commands) -> None:
    answers_parser = commands.add_parser(
        "answers", help="Run the QA golden datasets through a workflow and score its answers"
    )
    answers_parser.add_argument(
        "--workflow", required=True, help="A registered workflow name (agents/registry.py)"
    )
    answers_parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_ANSWER_DATASET,
        help="A QA .jsonl file, or a directory of them (default: every per-family file)",
    )
    answers_parser.add_argument(
        "--output", type=Path, help="Also write the report and per-row results as JSON here"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m evals.harness", description="Offline evals")
    commands = parser.add_subparsers(dest="command", required=True)
    _add_answers_command(commands)
    intent_parser = commands.add_parser(
        "intent", help="Score detect_intent/clarify_question_type against intent_detection.jsonl"
    )
    intent_parser.add_argument("--dataset", type=Path, default=DEFAULT_INTENT_DATASET)
    intent_parser.add_argument(
        "--output", type=Path, help="Also write the report and per-row results as JSON here"
    )
    arguments = parser.parse_args(argv)

    if not os.environ.get("OPENAI_API_KEY"):
        parser.exit(2, "OPENAI_API_KEY is not set -- the evals make real model calls\n")
    if arguments.command == "answers":
        if arguments.workflow not in list_workflows() or not is_functional(arguments.workflow):
            functional = ", ".join(name for name in list_workflows() if is_functional(name))
            parser.exit(2, f"No runnable workflow {arguments.workflow!r} (choose from: {functional})\n")
        asyncio.run(_run_answers(arguments.workflow, arguments.dataset, arguments.output))
        return
    asyncio.run(_run_intent(arguments.dataset, arguments.output))


if __name__ == "__main__":
    main()
