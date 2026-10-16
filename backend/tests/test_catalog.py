"""The families-with-trays endpoints (api/routes/catalog.py) -- the tree and
its display-name edits -- the chat tools' display-name translation
(catalog/display_names.py), and the legacy system-tag -> family map
family-scoped graph lookups go through (catalog/legacy_systems.py) --
against real Postgres/Neo4j, no mocking.
"""
import uuid

import pytest
from fastapi.testclient import TestClient

from agents.tools.graph_query import graph_query
from agents.tools.part_lookup import part_lookup
from api.main import app
from catalog.legacy_systems import LEGACY_SYSTEM_FAMILIES, resolve_family_name
from catalog.repository import upsert_families_and_trays
from ingestion.seed_master_catalog import seed_master_catalog
from retrieval.graph_client import get_graph_client
from tests.test_seed_master_catalog import _write_mini_master

client = TestClient(app)


def _signup_token(monkeypatch, tmp_path, *, admin: bool) -> str:
    monkeypatch.setenv("INGEST_DATA_DIR", str(tmp_path))
    email = f"test-{uuid.uuid4().hex[:12]}@example.com"
    if admin:
        monkeypatch.setenv("ADMIN_EMAILS", email)
    response = client.post("/auth/signup", json={"email": email, "password": "correct horse battery"})
    return response.json()["access_token"]


async def test_product_families_endpoint_returns_the_family_tray_tree(monkeypatch, tmp_path):
    suffix = uuid.uuid4().hex[:8]
    family, tray_a, tray_b = f"Family {suffix}", f"Tray A {suffix}", f"Tray B {suffix}"
    await upsert_families_and_trays({family: [tray_a, tray_b]})
    headers = {"Authorization": f"Bearer {_signup_token(monkeypatch, tmp_path, admin=True)}"}

    response = client.get("/catalog/product-families", headers=headers)

    assert response.status_code == 200
    listed = next(entry for entry in response.json() if entry["name"] == family)
    assert listed["display_name"] == family
    assert sorted((tray["name"], tray["display_name"]) for tray in listed["trays"]) == [
        (tray_a, tray_a),
        (tray_b, tray_b),
    ]


def test_product_families_endpoint_is_admin_only(monkeypatch, tmp_path):
    headers = {"Authorization": f"Bearer {_signup_token(monkeypatch, tmp_path, admin=False)}"}
    assert client.get("/catalog/product-families", headers=headers).status_code == 403
    assert client.get("/catalog/product-families").status_code == 401


def _listed_family(name: str, headers: dict) -> dict:
    response = client.get("/catalog/product-families", headers=headers)
    return next(entry for entry in response.json() if entry["name"] == name)


async def test_admin_edits_family_and_tray_display_names_and_blank_resets(monkeypatch, tmp_path):
    suffix = uuid.uuid4().hex[:8]
    family, tray = f"Ankle Fracture Plating Sytem {suffix}", f"Edit Tray {suffix}"
    await upsert_families_and_trays({family: [tray]})
    headers = {"Authorization": f"Bearer {_signup_token(monkeypatch, tmp_path, admin=True)}"}
    listed = _listed_family(family, headers)

    family_response = client.patch(
        f"/catalog/product-families/{listed['id']}",
        json={"display_name": f"  Ankle Fracture Plating System {suffix}  "},
        headers=headers,
    )
    tray_response = client.patch(
        f"/catalog/trays/{listed['trays'][0]['id']}",
        json={"display_name": f"Edited Tray {suffix}"},
        headers=headers,
    )

    assert family_response.status_code == 200
    assert family_response.json() == {
        "id": listed["id"],
        "name": family,
        "display_name": f"Ankle Fracture Plating System {suffix}",
    }
    assert tray_response.status_code == 200
    relisted = _listed_family(family, headers)
    assert relisted["display_name"] == f"Ankle Fracture Plating System {suffix}"
    assert relisted["trays"][0]["display_name"] == f"Edited Tray {suffix}"

    # Clearing a display name falls back to the catalog name.
    for blank in ("", "   ", None):
        reset = client.patch(
            f"/catalog/trays/{listed['trays'][0]['id']}", json={"display_name": blank}, headers=headers
        )
        assert reset.json()["display_name"] == tray
    reset_family = client.patch(f"/catalog/product-families/{listed['id']}", json={}, headers=headers)
    assert reset_family.json()["display_name"] == family


def test_display_name_edits_404_for_unknown_ids(monkeypatch, tmp_path):
    headers = {"Authorization": f"Bearer {_signup_token(monkeypatch, tmp_path, admin=True)}"}
    unknown_id = uuid.uuid4()
    body = {"display_name": "Anything"}
    assert client.patch(f"/catalog/product-families/{unknown_id}", json=body, headers=headers).status_code == 404
    assert client.patch(f"/catalog/trays/{unknown_id}", json=body, headers=headers).status_code == 404


def test_display_name_edits_are_admin_only(monkeypatch, tmp_path):
    headers = {"Authorization": f"Bearer {_signup_token(monkeypatch, tmp_path, admin=False)}"}
    some_id = uuid.uuid4()
    body = {"display_name": "Anything"}
    assert client.patch(f"/catalog/product-families/{some_id}", json=body, headers=headers).status_code == 403
    assert client.patch(f"/catalog/trays/{some_id}", json=body, headers=headers).status_code == 403
    assert client.patch(f"/catalog/trays/{some_id}", json=body).status_code == 401


async def test_reseed_after_an_admin_edit_keeps_the_edited_display_names(monkeypatch, tmp_path):
    suffix = uuid.uuid4().hex[:8]
    family, tray = f"Reseed Family {suffix}", f"Reseed Tray {suffix}"
    graph = get_graph_client()
    await graph.ensure_constraints()
    path = _write_mini_master(tmp_path / "reseed.xlsx", [(family, tray, f"RSD-{suffix}")])
    await seed_master_catalog(graph, path)
    headers = {"Authorization": f"Bearer {_signup_token(monkeypatch, tmp_path, admin=True)}"}
    listed = _listed_family(family, headers)
    client.patch(
        f"/catalog/product-families/{listed['id']}", json={"display_name": "Edited Family"}, headers=headers
    )
    client.patch(
        f"/catalog/trays/{listed['trays'][0]['id']}", json={"display_name": "Edited Tray"}, headers=headers
    )

    await seed_master_catalog(graph, path)

    relisted = _listed_family(family, headers)
    assert relisted["id"] == listed["id"]
    assert relisted["display_name"] == "Edited Family"
    assert [(listed_tray["name"], listed_tray["display_name"]) for listed_tray in relisted["trays"]] == [
        (tray, "Edited Tray")
    ]


async def test_chat_tools_show_display_names_and_accept_them_back(monkeypatch, tmp_path):
    suffix = uuid.uuid4().hex[:8]
    family, tray, sku = f"Typo Sytem {suffix}", f"Typo Tray {suffix}", f"TYP-{suffix}"
    graph = get_graph_client()
    await graph.ensure_constraints()
    await graph.upsert_tray(tray, family)
    await graph.upsert_part(sku, tray, description=f"TYPO PART {suffix}")
    await upsert_families_and_trays({family: [tray]})
    headers = {"Authorization": f"Bearer {_signup_token(monkeypatch, tmp_path, admin=True)}"}
    listed = _listed_family(family, headers)
    family_display, tray_display = f"Typo System {suffix}", f"Fixed Tray {suffix}"
    client.patch(
        f"/catalog/product-families/{listed['id']}", json={"display_name": family_display}, headers=headers
    )
    client.patch(
        f"/catalog/trays/{listed['trays'][0]['id']}", json={"display_name": tray_display}, headers=headers
    )

    found = await part_lookup.ainvoke({"term": sku, "product_family": family_display})
    assert [part["sku"] for part in found][:1] == [sku]
    assert found[0]["product_family"] == family_display

    related = await graph_query.ainvoke({"entity": sku, "relationship": "BELONGS_TO_TRAY"})
    assert [row["related_entity"] for row in related] == [tray_display]
    assert related[0]["related_properties"]["name"] == tray_display

    tray_family = await graph_query.ainvoke({"entity": tray_display, "relationship": "BELONGS_TO_FAMILY"})
    assert [row["related_entity"] for row in tray_family] == [family_display]


def test_real_family_names_pass_through_the_legacy_map_unchanged():
    # A real family name containing " - " must not be suffix-stripped.
    assert resolve_family_name("MIS - Foot Recon System") == "MIS - Foot Recon System"
    assert resolve_family_name("mis - foot recon") == "MIS - Foot Recon System"


@pytest.mark.parametrize(("legacy_tag", "family"), sorted(LEGACY_SYSTEM_FAMILIES.items()))
async def test_chat_part_lookup_resolves_each_legacy_system_tag(legacy_tag, family):
    suffix = uuid.uuid4().hex[:8]
    tray, sku = f"Legacy Lookup Tray {suffix}", f"LGL-{suffix}"
    graph = get_graph_client()
    await graph.ensure_constraints()
    await graph.upsert_tray(tray, family)
    await graph.upsert_part(sku, tray, description=f"LEGACY LOOKUP PART {suffix}")

    found = await graph.find_parts(sku, product_family=legacy_tag)

    assert [part["sku"] for part in found][:1] == [sku]
    assert found[0]["product_family"] == family


async def test_ingestion_lookups_take_the_documents_real_family_not_a_legacy_tag():
    # Ingestion scopes by the document's own Product Family now, so the
    # legacy map is no longer applied on this path.
    legacy_tag, family = "MIS - Foot Recon", "MIS - Foot Recon System"
    suffix = uuid.uuid4().hex[:8]
    tray, sku = f"Family Lookup Tray {suffix}", f"FML-{suffix}"
    graph = get_graph_client()
    await graph.ensure_constraints()
    await graph.upsert_tray(tray, family)
    await graph.upsert_part(sku, tray, description=f"FAMILY LOOKUP PART {suffix}")

    assert sku in {part["sku"] for part in await graph.list_parts_for_family(family)}
    assert tray in await graph.list_trays_for_family(family)
    assert sku not in {part["sku"] for part in await graph.list_parts_for_family(legacy_tag)}
