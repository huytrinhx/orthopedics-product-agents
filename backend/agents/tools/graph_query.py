"""LangGraph tool wrapping Neo4j/AuraDB multi-hop entity queries.

See backend/retrieval/graph_client.py for the underlying client and the
schema (Part/Tray/ProductFamily/Procedure/Document, COMPATIBLE_WITH/
REQUIRES_TOOL/DIFFERENTIATES_FROM/REQUIRES/SOURCED_FROM edges).
"""
from langchain_core.tools import tool

from catalog.display_names import CatalogDisplayNames, load_catalog_display_names
from retrieval.graph_client import get_graph_client


def _with_display_name(row: dict, display_names: CatalogDisplayNames) -> dict:
    """A related Tray/ProductFamily is shown by its admin-edited display
    name (issue #15) -- both its `related_entity` and its `name` property."""
    labels = row.get("related_labels") or []
    if "Tray" in labels:
        to_display = display_names.tray_display_name
    elif "ProductFamily" in labels:
        to_display = display_names.family_display_name
    else:
        return row
    properties = dict(row.get("related_properties") or {})
    if "name" in properties:
        properties["name"] = to_display(properties["name"])
    return {**row, "related_entity": to_display(row["related_entity"]), "related_properties": properties}


@tool
async def graph_query(entity: str, relationship: str | None = None) -> list[dict]:
    """Query the product/clinical entity graph for related entities.

    `entity` matches a Part by SKU, or a Tray/ProductFamily/Procedure/
    CanonicalTerm by name (case-insensitive, falling back to a substring
    match on Part descriptions). `relationship` optionally restricts to one
    edge type, e.g. "COMPATIBLE_WITH", "DIFFERENTIATES_FROM", "REQUIRES_TOOL",
    "REQUIRES" (procedure -> tray), "SOURCED_FROM".
    """
    display_names = await load_catalog_display_names()
    rows = await get_graph_client().query_related_entities(
        display_names.catalog_name(entity), relationship
    )
    return [_with_display_name(row, display_names) for row in rows]
