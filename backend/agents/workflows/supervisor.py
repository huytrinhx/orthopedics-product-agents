"""Supervisor/multi-agent workflow: a router agent dispatches the query to
specialist retrieval sub-agents (e.g. clinical-guidelines, product-catalog)
and composes their results into a single answer.
"""
from langgraph.graph import StateGraph  # noqa: F401

from agents.registry import register
from agents.state import BaseAgentState  # noqa: F401


def build_graph(checkpointer):
    # TODO: StateGraph(BaseAgentState), define specialist sub-graphs and a
    # supervisor node that routes between them, then
    # graph.compile(checkpointer=checkpointer)
    raise NotImplementedError("Wire up supervisor multi-agent graph")


register("supervisor", build_graph)
