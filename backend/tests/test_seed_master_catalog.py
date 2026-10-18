"""Parsing/resolution logic for backend/ingestion/seed_master_catalog.py
against the real master catalog (backend/evals/master-product-detail-oct-26.xlsx)
-- these are plain functions, no Neo4j needed. The end-to-end
seed_master_catalog write path is exercised separately below against small
synthetic workbooks, against real Neo4j and Postgres -- no mocking, and
small/fast rather than re-seeding the whole ~2000-row real file on every
test run.
"""
import uuid
from functools import cache
from pathlib import Path

import pytest
from openpyxl import Workbook

from catalog.repository import list_families_with_trays
from config.db import get_connection
from ingestion.seed_master_catalog import (
    CatalogRow,
    MasterCatalogError,
    normalize_tray_name,
    parse_master_catalog,
    resolve_compatible_skus,
    resolve_tool_sku,
    seed_master_catalog,
)
from retrieval.graph_client import get_graph_client

FIXTURE = Path(__file__).resolve().parents[1] / "evals" / "master-product-detail-oct-26.xlsx"
BRAND_PREFIX = "UNITE® Foot & Ankle | "


@cache
def _real_rows() -> tuple[CatalogRow, ...]:
    return tuple(parse_master_catalog(FIXTURE))


def test_parses_real_master_file():
    rows = _real_rows()
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
    assert all(row.sku for row in _real_rows())


def test_na_specs_are_normalized_to_empty():
    assert all(row.guidewire_spec.upper() != "N/A" for row in _real_rows())


def test_real_catalog_has_18_families_and_27_trays_each_under_one_family():
    rows = _real_rows()
    families_by_tray: dict[str, set[str]] = {}
    for row in rows:
        families_by_tray.setdefault(row.tray, set()).add(row.family)
    assert len({row.family for row in rows}) == 18
    assert len(families_by_tray) == 27
    assert all(len(families) == 1 for families in families_by_tray.values())
    assert {"MIS - Foot Recon System", "REFLEX HYBRID", "Cannulated Screws"} <= {
        row.family for row in rows
    }
    assert not {"MIS", "REFLEX"} & {row.family for row in rows}


def test_real_tray_names_drop_the_brand_segment_and_are_trimmed():
    trays = {row.tray for row in _real_rows()}
    assert not any(tray.startswith("UNITE") for tray in trays)
    assert all(tray == tray.strip() for tray in trays)
    # The one tray with a trailing space in the file.
    assert "REFLEX® Foot Recon Nitinol Implant System" in trays
    # Inner " | " segments are part of the name, not more brand prefix.
    assert "Soft Tissue | Comprehensive Instrument Tray" in trays


def test_normalize_tray_name_strips_only_the_exact_brand_segment():
    assert normalize_tray_name(f"{BRAND_PREFIX}Soft Tissue | Comprehensive Tray ") == (
        "Soft Tissue | Comprehensive Tray"
    )
    assert normalize_tray_name("Other Brand | Some Tray") == "Other Brand | Some Tray"
    assert normalize_tray_name("  Already Bare Tray  ") == "Already Bare Tray"


def test_mis_foot_recon_indication_enrichment_applies_to_the_renamed_tray():
    tray_rows = [row for row in _real_rows() if row.tray == "MIS Foot Recon System"]
    assert tray_rows
    hallux_valgus_rows = [row for row in tray_rows if "HV" in row.description.upper()]
    compression_rows = [
        row
        for row in tray_rows
        if "COMPRESSION" in row.description.upper() and "HV" not in row.description.upper()
    ]
    assert hallux_valgus_rows and compression_rows
    assert all(row.indication == "Bunion (Hallux Valgus) correction" for row in hallux_valgus_rows)
    assert all(
        row.indication == "Fusion / Akin osteotomy (compression fixation)"
        for row in compression_rows
    )
    # Still scoped to that one tray.
    assert all(row.indication is None for row in _real_rows() if row.tray != "MIS Foot Recon System")


def test_compatibility_matrix_resolves_via_sku_prefix():
    rows = _real_rows()
    plate = next(row for row in rows if row.description.startswith("PLATE, LATERAL FIBULA, XSML, LEFT"))
    all_skus = [row.sku for row in rows]
    matches = resolve_compatible_skus("2.7mm Polyaxial Locking (MPSL27xx)", all_skus, plate.sku)
    assert matches
    assert all(sku.startswith("MPSL27") for sku in matches)
    assert plate.sku not in matches


def test_compatibility_header_missing_its_closing_paren_still_resolves():
    # Verbatim from the Oct-26 header row.
    assert resolve_compatible_skus("2.0mm Non-Locking (MMSN20xx", ["MMSN2010"], "plate") == [
        "MMSN2010"
    ]


def test_compatibility_column_without_resolvable_prefix_is_skipped():
    all_skus = ["MPSL2710", "SOMETHING"]
    assert resolve_compatible_skus("SYNDEX", all_skus, "anything") == []


def test_guidewire_tool_resolution_matches_by_diameter_and_length_within_tray():
    rows = _real_rows()
    screw = next(row for row in rows if row.sku == "MSD14030")
    tray_rows = [row for row in rows if row.tray == screw.tray]
    assert resolve_tool_sku("guidewire", screw.guidewire_spec, tray_rows) == "MSG14150"


def test_unresolvable_tool_spec_returns_none():
    assert resolve_tool_sku("guidewire", "", []) is None
    assert resolve_tool_sku("driver", "", []) is None


# ---- Synthetic workbooks ----

_MINI_HEADER = [
    "Family", "Tray", "Item Type", "Item No.", "Description", "Qty per Set", "Head Style",
    "Construct", "Thread", "Color", "Guidewire", "Pre-Drill Diameter", "Driver",
]


def _write_workbook(path: Path, header: list[str], rows: list[list]) -> Path:
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(header)
    for row in rows:
        sheet.append(row)
    workbook.save(path)
    return path


def _write_mini_master(path: Path, family_tray_sku_rows: list[tuple[str, str, str]]) -> Path:
    return _write_workbook(
        path,
        _MINI_HEADER,
        [
            [family, tray, "Implant", sku, "TEST PART", 1, None, None, None, None, None, None, None]
            for family, tray, sku in family_tray_sku_rows
        ],
    )


def test_blank_family_fails_with_the_offending_row_numbers(tmp_path):
    path = _write_mini_master(
        tmp_path / "blank-family.xlsx",
        [("Family A", "Tray A", "SKU-1"), ("", "Tray A", "SKU-2"), (None, "Tray B", "SKU-3")],
    )
    with pytest.raises(MasterCatalogError, match=r"every row needs a Family.*row\(s\) 3, 4"):
        parse_master_catalog(path)


def test_tray_under_two_families_fails(tmp_path):
    path = _write_mini_master(
        tmp_path / "split-tray.xlsx",
        [("Family A", "Shared Tray", "SKU-1"), ("Family B", "Shared Tray", "SKU-2")],
    )
    with pytest.raises(MasterCatalogError, match="exactly one Family.*'Shared Tray'"):
        parse_master_catalog(path)


def test_missing_family_column_fails(tmp_path):
    header = [column for column in _MINI_HEADER if column != "Family"]
    path = _write_workbook(tmp_path / "no-family.xlsx", header, [["Tray", "Implant", "SKU-1"]])
    with pytest.raises(MasterCatalogError, match="no Family column"):
        parse_master_catalog(path)


def test_columns_are_looked_up_by_header_name_not_position(tmp_path):
    header = ["Item No.", "Description", "Tray", "Family", "Thread"]
    path = _write_workbook(
        tmp_path / "reordered.xlsx",
        header,
        [["SKU-1", "SCREW, TEST", f"{BRAND_PREFIX}Some Tray ", "Some Family", "Full"]],
    )
    [row] = parse_master_catalog(path)
    assert (row.sku, row.description, row.tray, row.family, row.thread) == (
        "SKU-1", "SCREW, TEST", "Some Tray", "Some Family", "Full",
    )


# ---- Seeding against real Neo4j/Postgres ----


async def _tray_names_for(client, sku: str) -> set[str]:
    related = await client.query_related_entities(sku, "BELONGS_TO_TRAY")
    return {related_entry["related_entity"] for related_entry in related}


async def _tray_exists(client, tray: str) -> bool:
    async with client._driver.session() as session:
        result = await session.run("MATCH (t:Tray {name: $name}) RETURN t.name AS name", name=tray)
        return await result.single() is not None


async def _family_names_for_tray(client, tray: str) -> list[str]:
    async with client._driver.session() as session:
        result = await session.run(
            "MATCH (:Tray {name: $name})-[:BELONGS_TO_FAMILY]->(f:ProductFamily) RETURN f.name AS name",
            name=tray,
        )
        return [record["name"] async for record in result]


async def test_seed_master_catalog_writes_parts_family_and_compatibility(tmp_path):
    suffix = uuid.uuid4().hex[:8]
    plate_sku = f"MPPA-{suffix}"
    screw_sku = f"MPSL27-{suffix[:6]}"
    family, tray = f"Test Family {suffix}", f"Test Tray {suffix}"
    header = [*_MINI_HEADER, "2.7mm Polyaxial Locking (MPSL27xx)"]
    path = _write_workbook(
        tmp_path / "mini-master.xlsx",
        header,
        [
            [family, f"{BRAND_PREFIX}{tray}", "Implant", plate_sku, "PLATE, TEST", 1,
             None, None, None, None, "N/A", "N/A", None, "X"],
            [family, f"{BRAND_PREFIX}{tray}", "Implant", screw_sku, "SCREW, TEST", 1,
             "Polyaxial Locking", "Cannulated", "Full", "Green", "N/A", "N/A", "T15", None],
        ],
    )

    client = get_graph_client()
    await client.ensure_constraints()
    await seed_master_catalog(client, path)

    assert await _tray_names_for(client, plate_sku) == {tray}
    assert await _family_names_for_tray(client, tray) == [family]

    compat = await client.query_related_entities(plate_sku, "COMPATIBLE_WITH")
    assert any(related_entry["related_entity"] == screw_sku for related_entry in compat)


async def test_seed_gives_each_tray_exactly_one_family(tmp_path):
    suffix = uuid.uuid4().hex[:8]
    tray, family, sku = f"Tray {suffix}", f"Family {suffix}", f"SKU-{suffix}"
    old_family, other_family = f"Old Family {suffix}", f"Other Family {suffix}"
    client = get_graph_client()
    await client.ensure_constraints()
    # A tray left sitting under two other families -- the seed's family
    # replaces both rather than being added alongside.
    await client.upsert_tray(tray, old_family)
    async with client._driver.session() as session:
        await session.run(
            "MATCH (t:Tray {name: $tray}) MERGE (f:ProductFamily {name: $family}) "
            "MERGE (t)-[:BELONGS_TO_FAMILY]->(f)",
            tray=tray,
            family=other_family,
        )

    await seed_master_catalog(
        client, _write_mini_master(tmp_path / "families.xlsx", [(family, tray, sku)])
    )

    assert await _family_names_for_tray(client, tray) == [family]


async def test_reseed_after_tray_rename_moves_parts_and_deletes_emptied_tray(tmp_path):
    suffix = uuid.uuid4().hex[:8]
    family = f"Family {suffix}"
    old_tray, new_tray, other_tray = (f"Old Tray {suffix}", f"New Tray {suffix}", f"Other Tray {suffix}")
    renamed_sku, shared_sku, untouched_sku = (f"REN-{suffix}", f"SHR-{suffix}", f"UNT-{suffix}")
    client = get_graph_client()
    await client.ensure_constraints()

    # A SKU outside the seeded file must survive the reconcile untouched.
    await client.upsert_part(untouched_sku, old_tray)
    await seed_master_catalog(
        client,
        _write_mini_master(
            tmp_path / "before.xlsx",
            [(family, old_tray, renamed_sku), (family, old_tray, shared_sku),
             (family, other_tray, shared_sku)],
        ),
    )
    await seed_master_catalog(
        client,
        _write_mini_master(
            tmp_path / "after.xlsx",
            [(family, new_tray, renamed_sku), (family, new_tray, shared_sku),
             (family, other_tray, shared_sku)],
        ),
    )

    assert await _tray_names_for(client, renamed_sku) == {new_tray}
    assert await _tray_names_for(client, shared_sku) == {new_tray, other_tray}
    assert await _tray_names_for(client, untouched_sku) == {old_tray}
    assert await _tray_exists(client, old_tray)  # still holds untouched_sku

    await seed_master_catalog(
        client, _write_mini_master(tmp_path / "again.xlsx", [(family, new_tray, untouched_sku)])
    )
    assert not await _tray_exists(client, old_tray)


async def test_emptied_tray_still_referenced_by_a_procedure_is_kept(tmp_path):
    suffix = uuid.uuid4().hex[:8]
    family = f"Family {suffix}"
    old_tray, new_tray = f"Old Tray {suffix}", f"New Tray {suffix}"
    sku, document_id = f"PRC-{suffix}", f"DOC-{suffix}"
    client = get_graph_client()
    await client.ensure_constraints()

    await seed_master_catalog(
        client, _write_mini_master(tmp_path / "before.xlsx", [(family, old_tray, sku)])
    )
    await client.upsert_document(document_id, "test.pdf", None, None)
    assert await client.attach_procedure(f"Procedure {suffix}", old_tray, document_id)

    await seed_master_catalog(
        client, _write_mini_master(tmp_path / "after.xlsx", [(family, new_tray, sku)])
    )

    assert await _tray_names_for(client, sku) == {new_tray}
    assert await _tray_exists(client, old_tray)


async def test_seed_mirrors_families_and_trays_into_postgres_and_keeps_display_names(tmp_path):
    suffix = uuid.uuid4().hex[:8]
    family, tray, other_tray = f"Family {suffix}", f"Tray {suffix}", f"Other Tray {suffix}"
    client = get_graph_client()
    await client.ensure_constraints()
    path = _write_mini_master(
        tmp_path / "mirror.xlsx",
        [(family, tray, f"SKU-A-{suffix}"), (family, other_tray, f"SKU-B-{suffix}")],
    )
    await seed_master_catalog(client, path)

    mirrored = {listed.name: listed for listed in await list_families_with_trays()}
    assert mirrored[family].display_name == family
    assert sorted(listed_tray.name for listed_tray in mirrored[family].trays) == sorted(
        [tray, other_tray]
    )
    assert all(listed_tray.display_name == listed_tray.name for listed_tray in mirrored[family].trays)

    connection = await get_connection()
    try:
        async with connection.cursor() as cursor:
            await cursor.execute(
                "UPDATE product_families SET display_name = %s WHERE name = %s",
                ("Edited Family", family),
            )
            await cursor.execute(
                "UPDATE trays SET display_name = %s WHERE name = %s", ("Edited Tray", tray)
            )
        await connection.commit()
    finally:
        await connection.close()

    await seed_master_catalog(client, path)

    mirrored = {listed.name: listed for listed in await list_families_with_trays()}
    assert mirrored[family].display_name == "Edited Family"
    display_names = {listed_tray.name: listed_tray.display_name for listed_tray in mirrored[family].trays}
    assert display_names == {tray: "Edited Tray", other_tray: other_tray}
