"""Confirms the package imports cleanly and every workflow registers itself."""
from agents import workflows  # noqa: F401
from agents.registry import list_workflows


def test_workflows_registered():
    assert set(list_workflows()) == {"deterministic", "react_agent", "supervisor"}
