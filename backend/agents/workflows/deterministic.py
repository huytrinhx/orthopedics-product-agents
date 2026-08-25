"""Deterministic retrieval workflow: fixed pipeline, no agentic tool choice.

query -> synonym_resolve -> hybrid_retrieve -> rerank -> generate -> self_eval
       -> (loop: reformulate + retry if self_eval faithfulness/relevance low)

This is the baseline workflow every other architecture is compared against
in the eval harness (backend/evals/harness.py).
"""
from langgraph.graph import StateGraph  # noqa: F401

from agents.registry import register
from agents.state import BaseAgentState  # noqa: F401

MAX_RETRIEVAL_LOOPS = 2


def build_graph():
    # TODO: StateGraph(BaseAgentState), add nodes (synonym_resolve,
    # hybrid_retrieve, rerank, generate, self_eval) and the conditional
    # retry edge, then graph.compile()
    raise NotImplementedError("Wire up deterministic retrieval graph nodes")


register("deterministic", build_graph)
