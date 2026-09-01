"""Maps a workflow name to a compiled LangGraph graph factory.

The API layer and eval harness select a workflow by name at request/run
time, so multiple agent architectures can be compared without changing
callers. Register new workflows here as they're built out in
backend/agents/workflows/.

Factories take the checkpointer rather than closing over one at import time:
AsyncPostgresSaver's connection pool is scoped to an `async with` block (see
backend/memory/checkpointer.py), so the checkpointer only exists once the
caller (backend/api/main.py's lifespan, or the eval harness) has opened one
-- it can't be a module-level singleton the registry constructs itself.
"""
from collections.abc import Callable

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

_REGISTRY: dict[str, Callable[[BaseCheckpointSaver], CompiledStateGraph]] = {}


def register(name: str, factory: Callable[[BaseCheckpointSaver], CompiledStateGraph]) -> None:
    _REGISTRY[name] = factory


def get_workflow(name: str, checkpointer: BaseCheckpointSaver) -> CompiledStateGraph:
    return _REGISTRY[name](checkpointer)


def list_workflows() -> list[str]:
    return list(_REGISTRY)
