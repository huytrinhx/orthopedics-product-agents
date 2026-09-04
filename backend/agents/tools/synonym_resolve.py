"""LangGraph tool that expands a query against the canonical synonym graph
in AuraDB before it's used for retrieval (e.g. 'ACL repair' -> also match
'anterior cruciate ligament reconstruction').
"""
from langchain_core.tools import tool

from agents.tools.term_extraction import with_singular_variants
from retrieval.graph_client import get_graph_client


@tool
async def synonym_resolve(term: str) -> list[str]:
    """Return canonical synonyms/aliases for a term from the entity graph."""
    # Graph Term nodes are singular ("wire" -[:ALIAS_OF]-> "guidepin"); a
    # plural term ("wires", "guidepins") as typed never matches one exactly.
    # deterministic.py's resolve_synonyms node already singularizes before
    # calling this tool, but react_agent.py calls it directly with whatever
    # term the model chose to pass -- seen for real in ticket 23's eval run
    # (the model tried "guidepins" and got nothing back, though "guidepin"
    # resolves fine). Tried here too so every caller benefits, not just the
    # one pipeline node that happened to add its own preprocessing.
    candidates = with_singular_variants([term.strip().lower()])
    groups = await get_graph_client().get_synonym_groups()
    for candidate in candidates:
        for group in groups:
            if any(variant.lower() == candidate for variant in group):
                return [variant for variant in group if variant.lower() != candidate]
    return []
