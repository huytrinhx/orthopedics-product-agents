"""LangGraph tool wrapping Neo4j/AuraDB multi-hop entity queries.

See backend/retrieval/graph_client.py for the underlying client and the
schema (Part/Tray/ProductFamily/Procedure/Document, COMPATIBLE_WITH/
REQUIRES_TOOL/DIFFERENTIATES_FROM/REQUIRES/SOURCED_FROM edges).
"""
from langchain_core.tools import tool

from retrieval.graph_client import get_graph_client


@tool
async def graph_query(entity: str, relationship: str | None = None) -> list[dict]:
    """Query the product/clinical entity graph for related entities.

    `entity` matches a Part by SKU, or a Tray/ProductFamily/Procedure/
    CanonicalTerm by name (case-insensitive, falling back to a substring
    match on Part descriptions). `relationship` optionally restricts to one
    edge type, e.g. "COMPATIBLE_WITH", "DIFFERENTIATES_FROM", "REQUIRES_TOOL",
    "REQUIRES" (procedure -> tray), "SOURCED_FROM".
    """
    return await get_graph_client().query_related_entities(entity, relationship)
