"""Legacy System tag -> Product Family map. Delete this module (and its two
callers) in the legacy-removal ticket.

Documents are still tagged with the old Postgres `systems` tags (e.g.
`MIS - Foot Recon`), which predate the master catalog's Family column.
Family-scoped graph lookups (backend/retrieval/graph_client.py) used to
derive a family from such a tag by stripping everything after " - ", which
only worked while the graph's families were the bare brand names `MIS` /
`REFLEX`. The Oct-26 catalog retired those for 18 real families, one of
which (`MIS - Foot Recon System`) itself contains " - " -- so the mapping
is spelled out here instead, one entry per legacy tag. Shared with the
migration that moves document tags from systems onto families.
"""

LEGACY_SYSTEM_FAMILIES: dict[str, str] = {
    "MIS - Foot Recon": "MIS - Foot Recon System",
    "MIS - Power Drive": "MIS - Power System",
    "REFLEX - Hybrid": "REFLEX HYBRID",
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
