"""Offline eval entry point (issue #25). Two slices, only the first built:

- `intent` (Slice 1, evals/intent_eval.py): runs intent_detection.jsonl's
  queries through the deterministic workflow's own intent classifier and
  scores the family, the question types, and whether each clarification
  would fire. Needs OPENAI_API_KEY and a DATABASE_URL whose catalog mirror
  is seeded (ingestion.seed_master_catalog); no documents or vector store.
- run_eval (Slice 2, not built): a full answer eval of a registered
  workflow over the per-family QA files, scored by the shared judge
  (agents/judge.py). Blocked on a pause policy for turns that stop on a
  clarification interrupt() -- see the issue.

Not run in CI: a real model call costs money, so an admin runs it by hand
(issue #25 comment; agents.md). Slice 2's pause policy is settled: a row
that pauses resumes with a scripted reply stored on the row.

Each slot (model_assignments/slots.py, issue #24) runs what production
runs -- its saved setting, or CHAT_MODEL when unset -- unless `--model
<slot>=<model_id>[@<reasoning_effort>]` overrides it for this run. The
report starts with every slot's model and effort, and `--output` records
them too. A non-OpenAI model id needs OPENROUTER_API_KEY.

Usage (from backend/):
    python -m evals.harness intent
    python -m evals.harness intent --dataset evals/golden_datasets/intent_detection.jsonl \\
        --output intent-results.json
    python -m evals.harness intent --model intent_classification=openai/gpt-5-mini@low
"""
import argparse
import asyncio
import os
from functools import partial
from pathlib import Path

from agents.workflows.deterministic import _classify_intent
from catalog.repository import list_families_with_trays
from config.llm_clients import OPENAI_MODEL_PREFIX
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


async def run_eval(workflow_name: str, dataset_path: str) -> dict:
    # Issue #25 Slice 2 (full answer eval with the shared judge) isn't built.
    raise NotImplementedError("Full answer eval is issue #25 Slice 2; only `intent` exists")


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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m evals.harness", description="Offline evals")
    commands = parser.add_subparsers(dest="command", required=True)
    intent_parser = commands.add_parser(
        "intent", help="Score detect_intent/clarify_question_type against intent_detection.jsonl"
    )
    intent_parser.add_argument("--dataset", type=Path, default=DEFAULT_INTENT_DATASET)
    intent_parser.add_argument(
        "--output", type=Path, help="Also write the report and per-row results as JSON here"
    )
    intent_parser.add_argument(
        "--model",
        dest="model_overrides",
        type=parse_model_override,
        action="append",
        default=[],
        metavar="SLOT=MODEL_ID[@EFFORT]",
        help="Run a slot on this model for this eval (repeatable), e.g. "
        "intent_classification=openai/gpt-5-mini@low",
    )
    arguments = parser.parse_args(argv)

    if not os.environ.get("OPENAI_API_KEY"):
        parser.exit(2, "OPENAI_API_KEY is not set -- the intent eval makes real classifier calls\n")
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
    asyncio.run(_run_intent(arguments.dataset, arguments.output, arguments.model_overrides))


if __name__ == "__main__":
    main()
