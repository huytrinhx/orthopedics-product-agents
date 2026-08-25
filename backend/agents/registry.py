"""Maps a workflow name to a compiled LangGraph graph factory.

The API layer and eval harness select a workflow by name at request/run
time, so multiple agent architectures can be compared without changing
callers. Register new workflows here as they're built out in
backend/agents/workflows/.
"""
from collections.abc import Callable

from langgraph.graph.state import CompiledStateGraph

_REGISTRY: dict[str, Callable[[], CompiledStateGraph]] = {}


def register(name: str, factory: Callable[[], CompiledStateGraph]) -> None:
    _REGISTRY[name] = factory


def get_workflow(name: str) -> CompiledStateGraph:
    return _REGISTRY[name]()


def list_workflows() -> list[str]:
    return list(_REGISTRY)
