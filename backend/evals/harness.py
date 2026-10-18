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

Each slot (model_assignments/slots.py, issue #24) runs what production
runs -- its saved setting, or CHAT_MODEL when unset -- unless `--model
<slot>=<model_id>[@<reasoning_effort>]` overrides it for this run. The
report starts with every slot's model and effort, and `--output` records
them too. A non-OpenAI model id needs OPENROUTER_API_KEY.

Usage (from backend/):
    python -m evals.harness intent
    python -m evals.harness intent --dataset evals/golden_datasets/intent_detection.jsonl \\
        --output intent-results.json
    python -m evals.harness answers --workflow deterministic
    python -m evals.harness answers --workflow react_agent \\
        --dataset evals/golden_datasets/reflex-hybrid.jsonl --output answer-results.json
    python -m evals.harness intent --model intent_classification=openai/gpt-5-mini@low
    python -m evals.harness answers --workflow deterministic --model deterministic_answer=openai/gpt-5-mini
"""
import argparse
import asyncio
import os
from dataclasses import asdict
from functools import partial
from pathlib import Path

from agents.registry import is_functional, list_workflows
from agents.workflows.deterministic import _classify_intent
from catalog.repository import list_families_with_trays
from config.llm_clients import OPENAI_MODEL_PREFIX
from documents.repository import list_documents
from evals import answer_eval
from evals.intent_eval import (
    format_report,
    load_intent_dataset,
    run_intent_eval,
    summarize,
    write_results,
)
from model_assignments.runtime import SlotModel, configured_slot_model, gateway_configured
from model_assignments.slots import REASONING_EFFORTS, SLOT_DEFINITIONS, Slot

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


def parse_model_override(value: str) -> tuple[Slot, str, str | None]:
    """`<slot>=<model_id>[@<reasoning_effort>]`, for argparse's `type=`."""
    slot_name, separator, choice = value.partition("=")
    if not separator or not choice:
        raise argparse.ArgumentTypeError(f"expected <slot>=<model_id>[@<effort>], got {value!r}")
    try:
        slot = Slot(slot_name)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"unknown slot {slot_name!r}; slots: {', '.join(slot.value for slot in Slot)}"
        )
    model_id, _, reasoning_effort = choice.partition("@")
    if not model_id:
        raise argparse.ArgumentTypeError(f"no model id in {value!r}")
    if reasoning_effort and reasoning_effort not in REASONING_EFFORTS:
        raise argparse.ArgumentTypeError(
            f"reasoning effort must be one of {', '.join(REASONING_EFFORTS)}, got {reasoning_effort!r}"
        )
    return slot, model_id, reasoning_effort or None


async def resolve_eval_slot_models(
    overrides: list[tuple[Slot, str, str | None]],
) -> dict[Slot, SlotModel]:
    """Every slot's model for this run: production's, then `--model`."""
    slot_models = {slot: await configured_slot_model(slot) for slot in SLOT_DEFINITIONS}
    for slot, model_id, reasoning_effort in overrides:
        slot_models[slot] = SlotModel(slot, model_id, reasoning_effort, "override")
    return slot_models


def format_slot_models(slot_models: dict[Slot, SlotModel]) -> str:
    lines = ["Models:"]
    for slot, slot_model in slot_models.items():
        description = slot_model.describe()
        lines.append(
            f"  {slot.value}: {description['model_id']} "
            f"(reasoning effort: {description['reasoning_effort'] or 'provider default'}; "
            f"{description['source']})"
        )
    return "\n".join(lines)


async def _run_intent(
    dataset_path: Path, output_path: Path | None, overrides: list[tuple[Slot, str, str | None]]
) -> None:
    rows = load_intent_dataset(dataset_path)
    families = await list_families_with_trays()
    if not families:
        # Same situation detect_intent logs and carries on through: with
        # nothing to pick between, the family clarification can never fire.
        print(
            "WARNING: no product families mirrored -- run ingestion.seed_master_catalog first; "
            "every family will resolve to None and never clarify"
        )

    slot_models = await resolve_eval_slot_models(overrides)
    print(format_slot_models(slot_models))
    print(f"Dataset: {dataset_path} ({len(rows)} rows), {len(families)} families mirrored")
    print()

    classify = partial(_classify_intent, slot_model=slot_models[Slot.INTENT_CLASSIFICATION])
    results = await run_intent_eval(rows, families, classify)
    report = summarize(results)
    print(format_report(report, results))
    if output_path is not None:
        described = {slot.value: slot_model.describe() for slot, slot_model in slot_models.items()}
        write_results(output_path, report, results, described)
        print()
        print(f"Wrote per-row results to {output_path}")


async def _run_answers(
    workflow_name: str,
    dataset_path: Path,
    output_path: Path | None,
    overrides: list[tuple[Slot, str, str | None]],
) -> None:
    rows = answer_eval.load_qa_dataset(dataset_path)
    documents = await _ingested_documents()
    if not documents:
        print(
            "WARNING: no ingested documents -- retrieval will find nothing and no expected "
            "citation can match"
        )

    slot_models = await resolve_eval_slot_models(overrides)
    print(f"Workflow: {workflow_name}")
    print(format_slot_models(slot_models))
    dataset_names = ", ".join(path.name for path in answer_eval.qa_dataset_paths(dataset_path))
    print(f"Dataset: {dataset_names} ({len(rows)} rows), {len(documents)} documents ingested")
    print()

    results = await answer_eval.run_answer_eval(
        rows, workflow_name, documents, slot_models=slot_models
    )
    report = answer_eval.summarize(workflow_name, results)
    print(answer_eval.format_report(report, results))
    if output_path is not None:
        described = {slot.value: slot_model.describe() for slot, slot_model in slot_models.items()}
        answer_eval.write_results(output_path, report, results, described)
        print()
        print(f"Wrote per-row results to {output_path}")


def _add_model_argument(command_parser) -> None:
    command_parser.add_argument(
        "--model",
        dest="model_overrides",
        type=parse_model_override,
        action="append",
        default=[],
        metavar="SLOT=MODEL_ID[@EFFORT]",
        help="Run a slot on this model for this eval (repeatable), e.g. "
        "intent_classification=openai/gpt-5-mini@low",
    )


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
    _add_model_argument(answers_parser)


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
    _add_model_argument(intent_parser)
    arguments = parser.parse_args(argv)

    if not os.environ.get("OPENAI_API_KEY"):
        parser.exit(2, "OPENAI_API_KEY is not set -- the evals make real model calls\n")
    gateway_only = [
        model_id
        for _, model_id, _ in arguments.model_overrides
        if not model_id.startswith(OPENAI_MODEL_PREFIX)
    ]
    if gateway_only and not gateway_configured():
        parser.exit(
            2,
            f"{', '.join(gateway_only)} can only run through OpenRouter -- set OPENROUTER_API_KEY\n",
        )
    if arguments.command == "answers":
        if arguments.workflow not in list_workflows() or not is_functional(arguments.workflow):
            functional = ", ".join(name for name in list_workflows() if is_functional(name))
            parser.exit(2, f"No runnable workflow {arguments.workflow!r} (choose from: {functional})\n")
        asyncio.run(
            _run_answers(
                arguments.workflow, arguments.dataset, arguments.output, arguments.model_overrides
            )
        )
        return
    asyncio.run(_run_intent(arguments.dataset, arguments.output, arguments.model_overrides))


if __name__ == "__main__":
    main()
