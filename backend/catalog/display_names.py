"""Catalog name <-> display name translation for the chat tools (issue #15).

The graph (Neo4j) only knows Product Families and Trays by their catalog
names -- straight from the "Do Not Edit" master file, typos included
(`Ankle Fracture Plating Sytem`). Admins fix how those read through the
`display_name` columns on the Postgres mirror (catalog.repository), and
users should only ever see the display name. So the graph-facing chat tools
(agents/tools/part_lookup.py, graph_query.py) translate at their boundary:
names coming out of the graph become display names, and a display name the
model passes back in (having just read it in an earlier tool result) is
mapped back to the catalog name the graph needs. Anything not recognised
passes through unchanged -- a legacy system tag, a SKU, a free-text term.
The deterministic workflow's intent classifier (issue #18) also chooses
among family display names and maps its pick back to a real family here.
"""
from dataclasses import dataclass

from catalog.repository import ProductFamilyRecord, list_families_with_trays


@dataclass
class CatalogDisplayNames:
    family_display_names: dict[str, str]
    tray_display_names: dict[str, str]

    def family_display_name(self, catalog_name: str) -> str:
        return self.family_display_names.get(catalog_name, catalog_name)

    def tray_display_name(self, catalog_name: str) -> str:
        return self.tray_display_names.get(catalog_name, catalog_name)

    def family_catalog_name(self, name: str) -> str:
        return _catalog_name_for(name, self.family_display_names)

    def catalog_name(self, name: str) -> str:
        """A family or tray display name back to its catalog name."""
        return _catalog_name_for(
            name, {**self.tray_display_names, **self.family_display_names}
        )


def _catalog_name_for(name: str, display_names: dict[str, str]) -> str:
    # Case-insensitive like the graph lookups themselves; an exact catalog
    # name wins over a display name that happens to match it.
    if name in display_names:
        return name
    lowered = name.strip().lower()
    for catalog_name, display_name in display_names.items():
        if display_name.lower() == lowered:
            return catalog_name
    return name


async def load_catalog_display_names() -> CatalogDisplayNames:
    return catalog_display_names_for(await list_families_with_trays())


def catalog_display_names_for(families: list[ProductFamilyRecord]) -> CatalogDisplayNames:
    """For a caller that already loaded the families for its own use (the
    deterministic workflow's detect_intent needs their ids too), so it
    doesn't read them twice."""
    return CatalogDisplayNames(
        family_display_names={family.name: family.display_name for family in families},
        tray_display_names={
            tray.name: tray.display_name for family in families for tray in family.trays
        },
    )
