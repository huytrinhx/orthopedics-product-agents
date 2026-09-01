"""One-off deterministic seed script for backend/evals/unite-master-csv.txt
(the real master item file / inventory control form) into the Neo4j graph
(see backend/retrieval/graph_client.py's schema docstring). Run manually —
this is NOT part of per-document ingestion (backend/ingestion/pipeline.py);
it's a batch job seeding the authoritative Part/Tray/ProductFamily catalog
that per-document prose extraction (backend/ingestion/entity_extraction.py)
then attaches facts to. Run it (and backend/ingestion/seed_synonyms.py)
before uploading/indexing real documents, or prose extraction has nothing to
attach to:

    python -m ingestion.seed_master_catalog [path/to/unite-master-csv.txt]

`Item No.` is a Part's only identity — rows without one are skipped. Two
kinds of relationship are resolved deterministically here, both acceptable
only inside this one-off, reviewable script (never in the LLM prose-
extraction path, which only merges by exact SKU):

  - COMPATIBLE_WITH (plate <-> screw family): each compatibility-matrix
    column header names a literal SKU prefix, e.g. "2.7mm Polyaxial Locking
    (MPSL27xx)" means any screw whose Item No. starts with "MPSL27" is
    compatible with a plate row marked "X" in that column. Exact prefix
    match, no fuzzy text involved.
  - REQUIRES_TOOL (screw <-> guidewire/drill-bit/driver): the Guidewire/
    Pre-Drill Diameter/Driver columns are free text with no SKU reference,
    even though matching catalog rows exist. Resolved by matching numeric
    tokens (diameter/length) or driver-bit tokens against candidate rows'
    Description, restricted to the same tray to avoid cross-tray false
    positives, and only written when exactly one candidate matches —
    ambiguous cases are left as the flat spec property on the Part instead
    of guessed into an edge.
"""
import asyncio
import csv
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from retrieval.graph_client import GraphClient, get_graph_client

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "backend" / "evals" / "unite-master-csv.txt"

_FIXED_COLUMNS = {
    "System",
    "Item Type",
    "Item No.",
    "Description",
    "Qty per Set",
    "Head Style",
    "Construct",
    "Thread",
    "Color",
    "Guidewire",
    "Pre-Drill Diameter",
    "Driver",
}
_COMPAT_PREFIX_RE = re.compile(r"\(([A-Za-z0-9]+)xx\)\s*$")
_NUMERIC_RE = re.compile(r"\d+(?:\.\d+)?")


@dataclass
class CatalogRow:
    tray: str
    item_type: str
    sku: str
    description: str
    qty_per_set: str
    head_style: str
    construct: str
    thread: str
    color: str
    guidewire_spec: str
    pre_drill_spec: str
    driver_spec: str
    compat_columns: dict[str, str]


def _clean(value: str | None) -> str:
    value = (value or "").strip()
    return "" if value.upper() == "N/A" else value


def _infer_product_family(tray: str) -> str | None:
    """Best-effort bucketing into the two product families the golden
    datasets/ticket-05 tags actually use today — most trays in this file
    (Ankle Fracture Plating, DEX Soft Tissue, ...) belong to neither and
    are seeded as standalone Trays with no ProductFamily edge, which is
    fine; ticket 05's System tag is admin-defined, not a fixed enum.
    """
    if re.search(r"\bMIS\b", tray):
        return "MIS"
    if re.search(r"\bREFLEX\b", tray, re.IGNORECASE):
        return "REFLEX"
    return None


def parse_master_catalog(path: Path) -> list[CatalogRow]:
    with path.open(newline="", encoding="latin-1") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = []
        for raw in reader:
            sku = _clean(raw.get("Item No."))
            if not sku:
                continue
            compat_columns = {
                header: value.strip()
                for header, value in raw.items()
                if header not in _FIXED_COLUMNS and value and value.strip()
            }
            rows.append(
                CatalogRow(
                    tray=_clean(raw.get("System")),
                    item_type=_clean(raw.get("Item Type")),
                    sku=sku,
                    description=_clean(raw.get("Description")),
                    qty_per_set=_clean(raw.get("Qty per Set")),
                    head_style=_clean(raw.get("Head Style")),
                    construct=_clean(raw.get("Construct")),
                    thread=_clean(raw.get("Thread")),
                    color=_clean(raw.get("Color")),
                    guidewire_spec=_clean(raw.get("Guidewire")),
                    pre_drill_spec=_clean(raw.get("Pre-Drill Diameter")),
                    driver_spec=_clean(raw.get("Driver")),
                    compat_columns=compat_columns,
                )
            )
        return rows


def _numeric_tokens(text: str) -> set[str]:
    return set(_NUMERIC_RE.findall(text))


def _resolve_by_numeric_tokens(spec: str, keyword: str, candidates: list[CatalogRow]) -> str | None:
    tokens = _numeric_tokens(spec)
    if not tokens:
        return None
    matches = [
        row
        for row in candidates
        if keyword in row.description.upper() and tokens.issubset(_numeric_tokens(row.description))
    ]
    return matches[0].sku if len(matches) == 1 else None


def _resolve_driver_sku(spec: str, candidates: list[CatalogRow]) -> str | None:
    token = spec.strip().upper()
    if not token:
        return None
    pattern = re.compile(rf"\b{re.escape(token)}\b")
    matches = [
        row
        for row in candidates
        if "DRIVER" in row.description.upper() and pattern.search(row.description.upper())
    ]
    return matches[0].sku if len(matches) == 1 else None


def resolve_compatible_skus(header: str, all_skus: list[str], own_sku: str) -> list[str]:
    match = _COMPAT_PREFIX_RE.search(header)
    if not match:
        return []
    prefix = match.group(1)
    return [sku for sku in all_skus if sku != own_sku and sku.startswith(prefix)]


def resolve_tool_sku(kind: str, spec: str, tray_rows: list[CatalogRow]) -> str | None:
    if kind == "guidewire":
        return _resolve_by_numeric_tokens(spec, "GUIDEWIRE", tray_rows)
    if kind == "pre_drill":
        return _resolve_by_numeric_tokens(spec, "DRILL", tray_rows)
    if kind == "driver":
        return _resolve_driver_sku(spec, tray_rows)
    raise ValueError(kind)


async def seed_master_catalog(client: GraphClient, path: Path = DEFAULT_PATH) -> None:
    rows = parse_master_catalog(path)
    if not rows:
        print(f"No rows with an Item No. found in {path}")
        return

    tray_family: dict[str, str | None] = {}
    for row in rows:
        tray_family.setdefault(row.tray, _infer_product_family(row.tray))

    for family in sorted({f for f in tray_family.values() if f}):
        await client.upsert_product_family(family)
    for tray, family in tray_family.items():
        await client.upsert_tray(tray, family)

    for row in rows:
        await client.upsert_part(
            row.sku,
            row.tray,
            description=row.description or None,
            item_type=row.item_type or None,
            head_style=row.head_style or None,
            construct=row.construct or None,
            thread=row.thread or None,
            color=row.color or None,
            guidewire_spec=row.guidewire_spec or None,
            pre_drill_spec=row.pre_drill_spec or None,
            driver_spec=row.driver_spec or None,
            qty_per_set=row.qty_per_set or None,
        )

    all_skus = [row.sku for row in rows]
    compat_count = 0
    for row in rows:
        for header, value in row.compat_columns.items():
            if value.upper() != "X":
                continue
            for other_sku in resolve_compatible_skus(header, all_skus, row.sku):
                await client.upsert_compatible_with(row.sku, other_sku)
                compat_count += 1

    rows_by_tray: dict[str, list[CatalogRow]] = defaultdict(list)
    for row in rows:
        rows_by_tray[row.tray].append(row)

    tool_count = 0
    for row in rows:
        tray_rows = rows_by_tray[row.tray]
        for kind, spec in (
            ("guidewire", row.guidewire_spec),
            ("pre_drill", row.pre_drill_spec),
            ("driver", row.driver_spec),
        ):
            if not spec:
                continue
            tool_sku = resolve_tool_sku(kind, spec, tray_rows)
            if tool_sku and tool_sku != row.sku:
                await client.upsert_requires_tool(row.sku, tool_sku)
                tool_count += 1

    print(
        f"Seeded {len(rows)} parts across {len(tray_family)} trays "
        f"({len([f for f in tray_family.values() if f])} mapped to a ProductFamily), "
        f"{compat_count} COMPATIBLE_WITH edges, {tool_count} REQUIRES_TOOL edges"
    )


async def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH
    client = get_graph_client()
    await client.ensure_constraints()
    await seed_master_catalog(client, path)


if __name__ == "__main__":
    asyncio.run(main())
