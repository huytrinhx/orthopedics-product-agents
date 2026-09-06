"""Supervisor/multi-agent workflow: deliberately unbuilt.

The original stub premise here (a router dispatching to domain specialists
like "clinical-guidelines" vs "product-catalog") was never grounded in this
codebase -- there is one knowledge domain and four shared tools
(vector_search, part_lookup, graph_query, synonym_resolve), not several.

A concrete alternative (Inventory specialist + Surgeon specialist +
Synthesizer + a citation-verifying Product Manager gate) was proposed and
grilled on 2026-09-04, then deferred -- checked directly against
artifacts/evals-react-agent.html's real miss/partial cases, neither of
react_agent's actual failures looks like "needed two framings and only got
one" (its miss is same-agent phrasing-dependent routing; its worst partial
is a tool/data-layer grounding bug `deterministic` hits identically through
a totally different architecture). Full reasoning, what was ruled out, and
the revisit condition: ticket 24
(.scratch/chat-documents-evals/issues/24-supervisor-multi-agent-deferred.md).

Do not build against the "clinical-guidelines/product-catalog" split -- it
was the original ungrounded placeholder, not a real design decision.
"""
from langgraph.graph import StateGraph  # noqa: F401

from agents.registry import register
from agents.state import BaseAgentState  # noqa: F401


def build_graph(checkpointer):
    # TODO: StateGraph(BaseAgentState), define specialist sub-graphs and a
    # supervisor node that routes between them, then
    # graph.compile(checkpointer=checkpointer)
    raise NotImplementedError("Wire up supervisor multi-agent graph")


register("supervisor", build_graph, functional=False)
