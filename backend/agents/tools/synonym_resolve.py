"""LangGraph tool that expands a query against the canonical synonym graph
in AuraDB before it's used for retrieval (e.g. 'ACL repair' -> also match
'anterior cruciate ligament reconstruction').
"""
from langchain_core.tools import tool

from retrieval.graph_client import get_graph_client


@tool
async def synonym_resolve(term: str) -> list[str]:
    """Return canonical synonyms/aliases for a term from the entity graph."""
    term_lower = term.strip().lower()
    for group in await get_graph_client().get_synonym_groups():
        if any(variant.lower() == term_lower for variant in group):
            return [variant for variant in group if variant.lower() != term_lower]
    return []
