"""ReAct-style agentic workflow: the model chooses which retrieval tools to
call (vector_search, graph_query, synonym_resolve) and when to stop,
looping until its own self-eval is satisfied or a step budget is hit.
"""
from langgraph.graph import StateGraph  # noqa: F401

from agents.registry import register
from agents.state import BaseAgentState  # noqa: F401


def build_graph(checkpointer):
    # TODO: StateGraph(BaseAgentState), bind tools from backend/agents/tools,
    # add a ReAct loop with interrupt() for clarification, then
    # graph.compile(checkpointer=checkpointer)
    raise NotImplementedError("Wire up ReAct agentic graph nodes")


register("react_agent", build_graph)
