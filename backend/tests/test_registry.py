"""Confirms the package imports cleanly and every workflow registers itself."""
from agents import workflows  # noqa: F401
from agents.registry import is_functional, list_workflows


def test_workflows_registered():
    assert set(list_workflows()) == {"deterministic", "react_agent", "supervisor"}


def test_functional_workflows_today():
    # Ticket 23 (2026-09-03): react_agent is a real, working agentic
    # workflow now, not a stub -- only supervisor remains unimplemented.
    assert is_functional("deterministic") is True
    assert is_functional("react_agent") is True
    assert is_functional("supervisor") is False
