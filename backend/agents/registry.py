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
from dataclasses import dataclass

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph


@dataclass
class _Entry:
    factory: Callable[[BaseCheckpointSaver], CompiledStateGraph]
    # False for a workflow that's registered (so it's visible/comparable)
    # but whose build_graph is still a stub that raises NotImplementedError
    # -- see workflows/react_agent.py and workflows/supervisor.py. The
    # admin workflow-selector (ticket 14) is the one place this matters:
    # it's what lets the picker show those as disabled instead of letting
    # an admin pick a workflow that can't actually run.
    functional: bool


_REGISTRY: dict[str, _Entry] = {}


def register(
    name: str,
    factory: Callable[[BaseCheckpointSaver], CompiledStateGraph],
    *,
    functional: bool = True,
) -> None:
    _REGISTRY[name] = _Entry(factory, functional)


def get_workflow(name: str, checkpointer: BaseCheckpointSaver) -> CompiledStateGraph:
    return _REGISTRY[name].factory(checkpointer)


def list_workflows() -> list[str]:
    return list(_REGISTRY)


def is_functional(name: str) -> bool:
    return _REGISTRY[name].functional
