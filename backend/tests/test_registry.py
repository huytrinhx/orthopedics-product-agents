"""Confirms the package imports cleanly and every workflow registers itself."""
from agents import workflows  # noqa: F401
from agents.registry import is_functional, list_workflows


def test_workflows_registered():
    assert set(list_workflows()) == {"deterministic", "react_agent", "supervisor"}


def test_only_deterministic_is_functional_today():
    assert is_functional("deterministic") is True
    assert is_functional("react_agent") is False
    assert is_functional("supervisor") is False
