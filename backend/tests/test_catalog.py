"""The read-only families-with-trays endpoint (api/routes/catalog.py) and the
legacy system-tag -> family map family-scoped graph lookups go through
(catalog/legacy_systems.py) -- against real Postgres/Neo4j, no mocking.
"""
import uuid

import pytest
from fastapi.testclient import TestClient

from api.main import app
from catalog.legacy_systems import LEGACY_SYSTEM_FAMILIES, resolve_family_name
from catalog.repository import upsert_families_and_trays
from retrieval.graph_client import get_graph_client

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
