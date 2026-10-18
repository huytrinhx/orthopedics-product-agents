"""Issue #8's spike: does every chat node's call shape work through
OpenRouter, under the Responses API and under Chat Completions? Its result
decides config/llm_clients.py's USE_RESPONSES_API and gets recorded in
docs/adr/0007-openrouter-chat-gateway.md.

Each check sends one node's real request shape to OpenRouter, built by the
same `openrouter_chat_model()` production uses, with only the API switched:

- with_structured_output with each node's own pydantic schema: detect_intent,
  the question-type reply classifier, the judge, entity extraction, and
  tray-layout extraction (with the probe's tray image fixture as input);
- bind_tools with react_agent's real tools, then a second round replaying the
  tool call and its result, the way react_agent's loop does. That replay is
  where a stateless gateway would reject item ids from the earlier response;
- streamed text, the way deterministic `generate` streams through LangGraph.

The checks themselves live in model_assignments/probe.py, which runs one
slot's worth of them whenever an admin saves or tests a model in Settings
(issue #24). No database, graph or tool execution: the tool result is
canned. Needs only OPENROUTER_API_KEY, and CHAT_MODEL for anything but the
default model.

Usage:
    python -m evals.openrouter_spike
    python -m evals.openrouter_spike --reasoning-effort low
    python -m evals.openrouter_spike --api chat_completions
"""
import argparse
import asyncio
import os
import sys

from config.llm_clients import configured_chat_model, openrouter_chat_model
from model_assignments.probe import SLOT_CHECKS, run_check

APIS = {"responses": True, "chat_completions": False}

# The same checks the Settings probe runs per slot (model_assignments/
# probe.py), flattened into one table.
CHECKS = {
    check_name: check
    for slot_checks in SLOT_CHECKS.values()
    for check_name, check in slot_checks.items()
}


async def run_spike(model_id: str, api_names: list[str], reasoning_effort: str | None) -> bool:
    results: dict[str, dict[str, tuple[bool, str]]] = {name: {} for name in CHECKS}
    for api_name in api_names:
        model = openrouter_chat_model(
            model_id, reasoning_effort=reasoning_effort, use_responses_api=APIS[api_name]
        )
        for check_name, check in CHECKS.items():
            results[check_name][api_name] = await run_check(check, model)

    print(f"model={model_id} reasoning_effort={reasoning_effort or '(model default)'}\n")
    name_width = max(len(name) for name in CHECKS)
    print(" | ".join([f"{'check':<{name_width}}", *(f"{api_name:<16}" for api_name in api_names)]))
    print("-" * (name_width + 19 * len(api_names)))
    for check_name, by_api in results.items():
        cells = [f"{'PASS' if by_api[api_name][0] else 'FAIL':<16}" for api_name in api_names]
        print(" | ".join([f"{check_name:<{name_width}}", *cells]))

    print("\nDetails:")
    for check_name, by_api in results.items():
        for api_name, (passed, detail) in by_api.items():
            print(f"  [{'PASS' if passed else 'FAIL'}] {api_name} / {check_name}: {detail}")
    return all(passed for by_api in results.values() for passed, _ in by_api.values())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run every chat node's call shape through OpenRouter under both APIs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", help="OpenRouter model id (defaults to CHAT_MODEL, e.g. openai/gpt-5.6-luna)"
    )
    parser.add_argument("--api", choices=sorted(APIS), action="append", help="repeatable; default both")
    parser.add_argument("--reasoning-effort", help="e.g. low/medium/high; default is the model's own")
    args = parser.parse_args()
    if not os.environ.get("OPENROUTER_API_KEY"):
        sys.exit("OPENROUTER_API_KEY isn't set -- this spike only talks to OpenRouter.")
    all_passed = asyncio.run(
        run_spike(args.model or configured_chat_model(), args.api or list(APIS), args.reasoning_effort)
    )
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
