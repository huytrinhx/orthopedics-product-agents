"""Parsing/resolution logic for backend/ingestion/seed_master_catalog.py
against the real fixture (backend/evals/unite-master-csv.txt) -- these are
plain functions, no Neo4j needed. The end-to-end seed_master_catalog write
path is exercised separately below against a small synthetic file, against
real Neo4j -- no mocking, and small/fast rather than re-seeding the whole
1886-row real file on every test run.
"""
import csv
import uuid
from pathlib import Path

from ingestion.seed_master_catalog import (
    _infer_product_family,
    parse_master_catalog,
    resolve_compatible_skus,
    resolve_tool_sku,
    seed_master_catalog,
)
from retrieval.graph_client import get_graph_client

FIXTURE = Path(__file__).resolve().parents[1] / "evals" / "unite-master-csv.txt"


def test_parses_real_master_file():
    rows = parse_master_catalog(FIXTURE)
    assert len(rows) > 1000
    skus = {row.sku for row in rows}
    assert "MPPA100L" in skus
    # Item No. is a Part's identity, but the *same* SKU legitimately repeats
    # across multiple tray rows (a shared screw/driver cataloged under more
    # than one system) -- that's why upsert_part MERGEs by sku and adds a
    # BELONGS_TO_TRAY edge per tray it appears under, rather than assuming
    # one row per SKU.
    assert len(skus) < len(rows)


def test_rows_without_item_no_are_skipped():
    rows = parse_master_catalog(FIXTURE)
    assert all(row.sku for row in rows)


def test_na_specs_are_normalized_to_empty():
    rows = parse_master_catalog(FIXTURE)
    assert all(row.guidewire_spec.upper() != "N/A" for row in rows)


def test_family_inference_matches_word_boundaries_not_substrings():
    assert _infer_product_family("UNITE® Foot & Ankle | MIS Foot Recon System") == "MIS"
    assert _infer_product_family("UNITE® Foot & Ankle | REFLEX® HYBRID Implant System") == "REFLEX"
    # "Miscellaneous" contains "MIS" as a raw substring but isn't the MIS system.
    assert _infer_product_family("UNITE® Foot & Ankle | Miscellaneous Surgical Accessories") is None
    assert _infer_product_family("UNITE® Foot & Ankle | Ankle Fusion Plating System") is None


def test_compatibility_matrix_resolves_via_sku_prefix():
    rows = parse_master_catalog(FIXTURE)
    plate = next(r for r in rows if r.description.startswith("PLATE, LATERAL FIBULA, XSML, LEFT"))
    all_skus = [r.sku for r in rows]
    matches = resolve_compatible_skus("2.7mm Polyaxial Locking (MPSL27xx)", all_skus, plate.sku)
    assert matches
    assert all(sku.startswith("MPSL27") for sku in matches)
    assert plate.sku not in matches


def test_compatibility_column_without_resolvable_prefix_is_skipped():
    all_skus = ["MPSL2710", "SOMETHING"]
    assert resolve_compatible_skus("SYNDEX", all_skus, "anything") == []


def test_guidewire_tool_resolution_matches_by_diameter_and_length_within_tray():
    rows = parse_master_catalog(FIXTURE)
    screw = next(r for r in rows if r.sku == "MSD14030")
    tray_rows = [r for r in rows if r.tray == screw.tray]
    assert resolve_tool_sku("guidewire", screw.guidewire_spec, tray_rows) == "MSG14150"


def test_unresolvable_tool_spec_returns_none():
    assert resolve_tool_sku("guidewire", "", []) is None
    assert resolve_tool_sku("driver", "", []) is None


async def test_seed_master_catalog_writes_parts_family_and_compatibility(tmp_path):
    plate_sku = f"MPPA-{uuid.uuid4().hex[:8]}"
    screw_sku = f"MPSL27-{uuid.uuid4().hex[:6]}"
    tray = f"UNITE® Foot & Ankle | MIS Test Tray {uuid.uuid4().hex[:6]}"
    header = [
        "System", "Item Type", "Item No.", "Description", "Qty per Set", "Head Style",
        "Construct", "Thread", "Color", "Guidewire", "Pre-Drill Diameter", "Driver",
        "2.7mm Polyaxial Locking (MPSL27xx)",
    ]
    plate_row = [tray, "Implant", plate_sku, "PLATE, TEST", "1", "", "", "", "", "N/A", "N/A", "", "X"]
    screw_row = [
        tray, "Implant", screw_sku, "SCREW, TEST", "1", "Polyaxial Locking", "Cannulated",
        "Full", "Green", "N/A", "N/A", "T15", "",
    ]
    csv_path = tmp_path / "mini-master.csv"
    with csv_path.open("w", newline="", encoding="latin-1") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows([header, plate_row, screw_row])

    client = get_graph_client()
    await client.ensure_constraints()
    await seed_master_catalog(client, csv_path)

    related = await client.query_related_entities(plate_sku)
    families = [r for r in related if r["relationship"] == "BELONGS_TO_TRAY"]
    assert families and families[0]["related_entity"] == tray

    compat = await client.query_related_entities(plate_sku, "COMPATIBLE_WITH")
    assert any(r["related_entity"] == screw_sku for r in compat)
