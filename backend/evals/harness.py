"""Runs a golden dataset against any registered workflow, scoring each
example with the shared judge (backend/agents/judge.py). Not run from CI:
its job there passed without evaluating anything, since run_eval is still a
stub, so issue #8 deleted it and #25 brings it back with a real harness.
"""
from agents.judge import judge_answer  # noqa: F401
from agents.registry import get_workflow  # noqa: F401


async def run_eval(workflow_name: str, dataset_path: str) -> dict:
    raise NotImplementedError
