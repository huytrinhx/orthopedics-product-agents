"""Legacy System tag -> Product Family map. Delete this module (and its two
call sites) in the legacy-removal ticket.

Documents are still tagged with the old Postgres `systems` tags (e.g.
`MIS - Foot Recon`), which predate the master catalog's Family column.
Family-scoped graph lookups (backend/retrieval/graph_client.py) used to
derive a family from such a tag by stripping everything after " - ", which
only worked while the graph's families were the bare brand names `MIS` /
`REFLEX`. The Oct-26 catalog retired those for 18 real families, one of
which (`MIS - Foot Recon System`) itself contains " - " -- so the mapping
is spelled out here instead, one entry per legacy tag. Its two callers:
chat's family-scoped part lookup (graph_client.find_parts, which is still
handed detect_intent's system tag) and the one-off script moving document
tags from systems onto families (catalog/migrate_legacy_system_tags.py).
Ingestion no longer goes through it -- it uses the document's own family.

Entries cover production's `systems` tags as of 2026-10-16 that aren't
already a family name verbatim (`Ancillary Foot Recon System`, `Core Foot
Plating System`, `Distal Tibia Plating System` and `MIS - Power System`
pass straight through). `Ankle Fracture Plating Sytem` is the catalog's own
spelling, typo included -- the display name fixes it, not the key.
"""

LEGACY_SYSTEM_FAMILIES: dict[str, str] = {
    "Ankle Fracture Plating System": "Ankle Fracture Plating Sytem",
    "MIS - Foot Recon": "MIS - Foot Recon System",
    "REFLEX Nitinol Implants": "REFLEX Foot Recon Systems",
}

_FAMILIES_BY_LOWERCASE_TAG = {
    tag.lower(): family for tag, family in LEGACY_SYSTEM_FAMILIES.items()
}


def resolve_family_name(name: str) -> str:
    """Maps a legacy system tag to its Product Family name, matching
    case-insensitively the way the `systems` table's own uniqueness index
    does. Anything else -- a real family name already -- passes through
    unchanged.
    """
    return _FAMILIES_BY_LOWERCASE_TAG.get(name.strip().lower(), name)
