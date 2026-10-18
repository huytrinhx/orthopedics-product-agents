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

Not run in CI: a real model call costs money, and whether a CI job should
spend it is one of #25's open questions.

Usage (from backend/):
    python -m evals.harness intent
    python -m evals.harness intent --dataset evals/golden_datasets/intent_detection.jsonl \\
        --output intent-results.json
"""
import argparse
import asyncio
import os
from pathlib import Path

from catalog.repository import list_families_with_trays
from config.llm_clients import get_chat_model
from evals.intent_eval import (
    format_report,
    load_intent_dataset,
    run_intent_eval,
    summarize,
    write_results,
)

DEFAULT_INTENT_DATASET = Path(__file__).parent / "golden_datasets" / "intent_detection.jsonl"


async def run_eval(workflow_name: str, dataset_path: str) -> dict:
    # Issue #25 Slice 2 (full answer eval with the shared judge) isn't built.
    raise NotImplementedError("Full answer eval is issue #25 Slice 2; only `intent` exists")


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
    arguments = parser.parse_args(argv)

    if not os.environ.get("OPENAI_API_KEY"):
        parser.exit(2, "OPENAI_API_KEY is not set -- the intent eval makes real classifier calls\n")
    asyncio.run(_run_intent(arguments.dataset, arguments.output))


if __name__ == "__main__":
    main()
