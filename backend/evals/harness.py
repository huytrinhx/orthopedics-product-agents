"""Runs a golden dataset against any registered workflow, scoring each
example with the shared judge (backend/agents/judge.py). Triggered from CI
(.github/workflows/ci.yml) on PRs that touch backend/agents/.
"""
from agents.judge import judge_answer  # noqa: F401
from agents.registry import get_workflow  # noqa: F401


async def run_eval(workflow_name: str, dataset_path: str) -> dict:
    raise NotImplementedError
