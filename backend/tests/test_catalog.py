"""The families-with-trays endpoints (api/routes/catalog.py) -- the tree and
its display-name edits -- and the chat tools' display-name translation
(catalog/display_names.py), against real Postgres/Neo4j, no mocking.
"""
import uuid

from fastapi.testclient import TestClient

from agents.tools.graph_query import graph_query
from agents.tools.part_lookup import part_lookup
from api.main import app
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


async def test_admin_replaces_a_family_description_and_blank_goes_back_to_generated(monkeypatch, tmp_path):
    suffix = uuid.uuid4().hex[:8]
    family, tray = f"Described Family {suffix}", f"Described Tray {suffix}"
    await upsert_families_and_trays({family: [tray]})
    headers = {"Authorization": f"Bearer {_signup_token(monkeypatch, tmp_path, admin=True)}"}
    listed = _listed_family(family, headers)
    assert listed["description_override"] is None
    assert isinstance(listed["generated_description"], str)

    response = client.patch(
        f"/catalog/product-families/{listed['id']}/description",
        json={"description": "  Nitinol staples and the Dynamic Disc.  "},
        headers=headers,
    )

    assert response.status_code == 200
    assert response.json() == {
        "id": listed["id"],
        "description_override": "Nitinol staples and the Dynamic Disc.",
    }
    relisted = _listed_family(family, headers)
    assert relisted["description_override"] == "Nitinol staples and the Dynamic Disc."
    assert relisted["generated_description"] == listed["generated_description"]

    for blank in ("", "   ", None):
        reset = client.patch(
            f"/catalog/product-families/{listed['id']}/description",
            json={"description": blank},
            headers=headers,
        )
        assert reset.json()["description_override"] is None


def test_description_edits_404_for_unknown_ids_and_are_admin_only(monkeypatch, tmp_path):
    some_id = uuid.uuid4()
    body = {"description": "Anything"}
    admin_headers = {"Authorization": f"Bearer {_signup_token(monkeypatch, tmp_path, admin=True)}"}
    assert (
        client.patch(f"/catalog/product-families/{some_id}/description", json=body, headers=admin_headers)
    ).status_code == 404
    user_headers = {"Authorization": f"Bearer {_signup_token(monkeypatch, tmp_path, admin=False)}"}
    assert (
        client.patch(f"/catalog/product-families/{some_id}/description", json=body, headers=user_headers)
    ).status_code == 403
    assert client.patch(f"/catalog/product-families/{some_id}/description", json=body).status_code == 401


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
    client.patch(
        f"/catalog/product-families/{listed['id']}/description",
        json={"description": "Edited description"},
        headers=headers,
    )

    await seed_master_catalog(graph, path)

    relisted = _listed_family(family, headers)
    assert relisted["id"] == listed["id"]
    assert relisted["display_name"] == "Edited Family"
    assert relisted["description_override"] == "Edited description"
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
    assert found[0]["trays"] == [tray_display]

    related = await graph_query.ainvoke({"entity": sku, "relationship": "BELONGS_TO_TRAY"})
    assert [row["related_entity"] for row in related] == [tray_display]
    assert related[0]["related_properties"]["name"] == tray_display

    tray_family = await graph_query.ainvoke({"entity": tray_display, "relationship": "BELONGS_TO_FAMILY"})
    assert [row["related_entity"] for row in tray_family] == [family_display]
