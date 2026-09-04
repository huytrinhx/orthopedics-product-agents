"""LangGraph tool resolving a free-text term directly to candidate Part
nodes and their own properties (sku, description, thread, guidewire_spec,
driver_spec, pre_drill_spec, head_style, construct, ...) -- ticket 20's
graph-grounding mechanism for spec/SKU/pairing questions, as opposed to
graph_query.py's relationship traversal (empty COMPATIBLE_WITH for MIS
parts; this data lives as plain Part properties instead, see
backend/retrieval/graph_client.py's find_parts docstring).
"""
from langchain_core.tools import tool

from retrieval.graph_client import get_graph_client


@tool
async def part_lookup(term: str, product_family: str | None = None) -> list[dict]:
    """Resolve a term (a SKU, or a word from a part's description) to
    candidate Part nodes, each with its own catalog properties. Optionally
    scope to one product system to avoid cross-system noise -- either a
    broad family name ("MIS") or a specific one ("MIS - Foot Recon") both
    work, so pass whatever system name you already have.
    """
    return await get_graph_client().find_parts(term, product_family=product_family)
