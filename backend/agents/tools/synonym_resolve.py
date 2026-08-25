"""LangGraph tool that expands a query against the canonical synonym graph
in AuraDB before it's used for retrieval (e.g. 'ACL repair' -> also match
'anterior cruciate ligament reconstruction').
"""
from langchain_core.tools import tool


@tool
async def synonym_resolve(term: str) -> list[str]:
    """Return canonical synonyms/aliases for a term from the entity graph."""
    raise NotImplementedError
