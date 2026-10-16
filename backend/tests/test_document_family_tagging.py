"""Tagging a document with one Product Family and many Trays (issue #12)
through PATCH /documents/{id}/tags -- against real Postgres, no mocking.
"""
import uuid

from fastapi.testclient import TestClient

from api.main import app
from catalog.repository import (
    ProductFamilyRecord,
    list_families_with_trays,
    upsert_families_and_trays,
)

client = TestClient(app)


def _unique_name(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _admin_headers(monkeypatch, tmp_path) -> dict:
    monkeypatch.setenv("INGEST_DATA_DIR", str(tmp_path))
    email = f"test-{uuid.uuid4().hex[:12]}@example.com"
    monkeypatch.setenv("ADMIN_EMAILS", email)
    response = client.post("/auth/signup", json={"email": email, "password": "correct horse battery"})
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


async def _seed_family(tray_count: int) -> ProductFamilyRecord:
    family_name = _unique_name("Family")
    await upsert_families_and_trays(
        {family_name: [_unique_name(f"Tray{index}") for index in range(tray_count)]}
    )
    return next(family for family in await list_families_with_trays() if family.name == family_name)


def _upload(headers: dict) -> str:
    upload = client.post(
        "/documents/upload", headers=headers, files={"file": ("doc.txt", b"x", "text/plain")}
    )
    assert upload.status_code == 200
    assert upload.json()["product_family"] is None
    assert upload.json()["trays"] == []
    return upload.json()["id"]


async def test_tagging_a_family_and_trays_shows_display_names(monkeypatch, tmp_path):
    headers = _admin_headers(monkeypatch, tmp_path)
    family = await _seed_family(tray_count=2)
    document_id = _upload(headers)

    edit = client.patch(
        f"/documents/{document_id}/tags",
        headers=headers,
        json={
            "product_family_id": str(family.id),
            "tray_ids": [str(tray.id) for tray in family.trays],
        },
    )

    assert edit.status_code == 200
    listed = next(
        document
        for document in client.get("/documents/", headers=headers).json()
        if document["id"] == document_id
    )
    assert listed["product_family"]["display_name"] == family.display_name
    assert sorted(tray["display_name"] for tray in listed["trays"]) == sorted(
        tray.display_name for tray in family.trays
    )
    assert "document_type" in listed


async def test_tray_outside_the_documents_family_is_rejected(monkeypatch, tmp_path):
    headers = _admin_headers(monkeypatch, tmp_path)
    family = await _seed_family(tray_count=1)
    other_family = await _seed_family(tray_count=1)
    document_id = _upload(headers)

    edit = client.patch(
        f"/documents/{document_id}/tags",
        headers=headers,
        json={
            "product_family_id": str(family.id),
            "tray_ids": [str(family.trays[0].id), str(other_family.trays[0].id)],
        },
    )

    assert edit.status_code == 400
    assert str(other_family.trays[0].id) in edit.json()["detail"]
    # Nothing was written.
    document = client.get(f"/documents/{document_id}", headers=headers).json()
    assert document["product_family"] is None
    assert document["trays"] == []


async def test_trays_without_a_family_are_rejected(monkeypatch, tmp_path):
    headers = _admin_headers(monkeypatch, tmp_path)
    family = await _seed_family(tray_count=1)
    document_id = _upload(headers)

    edit = client.patch(
        f"/documents/{document_id}/tags",
        headers=headers,
        json={"product_family_id": None, "tray_ids": [str(family.trays[0].id)]},
    )

    assert edit.status_code == 400


def test_unknown_family_is_rejected(monkeypatch, tmp_path):
    headers = _admin_headers(monkeypatch, tmp_path)
    document_id = _upload(headers)

    edit = client.patch(
        f"/documents/{document_id}/tags",
        headers=headers,
        json={"product_family_id": str(uuid.uuid4())},
    )

    assert edit.status_code == 400


async def test_editing_another_tag_leaves_family_and_trays_alone(monkeypatch, tmp_path):
    headers = _admin_headers(monkeypatch, tmp_path)
    family = await _seed_family(tray_count=1)
    document_id = _upload(headers)
    client.patch(
        f"/documents/{document_id}/tags",
        headers=headers,
        json={"product_family_id": str(family.id), "tray_ids": [str(family.trays[0].id)]},
    )
    document_type = client.post(
        "/document-types", headers=headers, json={"name": _unique_name("Brochure")}
    ).json()

    edit = client.patch(
        f"/documents/{document_id}/tags",
        headers=headers,
        json={"document_type_id": document_type["id"]},
    )

    assert edit.status_code == 200
    assert edit.json()["document_type"]["id"] == document_type["id"]
    assert edit.json()["product_family"]["id"] == str(family.id)
    assert [tray["id"] for tray in edit.json()["trays"]] == [str(family.trays[0].id)]
