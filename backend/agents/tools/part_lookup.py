"""LangGraph tool resolving a free-text term directly to candidate Part
nodes and their own properties (sku, description, thread, guidewire_spec,
driver_spec, pre_drill_spec, head_style, construct, ...) -- ticket 20's
graph-grounding mechanism for spec/SKU/pairing questions, as opposed to
graph_query.py's relationship traversal (empty COMPATIBLE_WITH for MIS
parts; this data lives as plain Part properties instead, see
backend/retrieval/graph_client.py's find_parts docstring).
"""
from langchain_core.tools import tool

from catalog.display_names import CatalogDisplayNames, load_catalog_display_names
from retrieval.graph_client import get_graph_client


@tool
async def part_lookup(term: str, product_family: str | None = None) -> list[dict]:
    """Resolve a term (a SKU, or a word from a part's description) to
    candidate Part nodes, each with its own catalog properties. Optionally
    scope to one product family (by name, e.g. "MIS - Foot Recon System")
    to avoid cross-family noise.
    """
    # Each result's product_family is its admin-edited display name (issue
    # #15), and that same display name is accepted back as product_family.
    # Its trays (issue #21) are display names too, for the same reason.
    display_names = await load_catalog_display_names()
    family = display_names.family_catalog_name(product_family) if product_family else None
    parts = await get_graph_client().find_parts(term, product_family=family)
    return [_with_display_names(part, display_names) for part in parts]


def _with_display_names(part: dict, display_names: CatalogDisplayNames) -> dict:
    translated = dict(part)
    if part.get("product_family"):
        translated["product_family"] = display_names.family_display_name(part["product_family"])
    if part.get("trays"):
        translated["trays"] = [display_names.tray_display_name(tray) for tray in part["trays"]]
    return translated
