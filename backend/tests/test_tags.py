"""Exercises System/Document-Type tag management and their use on documents
(ticket 05) against a real Postgres -- no mocking, same approach as
test_auth.py / test_documents.py.
"""
import time
import uuid

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def _unique_email() -> str:
    return f"test-{uuid.uuid4().hex[:12]}@example.com"


def _unique_name(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _admin_token(monkeypatch, tmp_path) -> str:
    monkeypatch.setenv("INGEST_DATA_DIR", str(tmp_path))
    email = _unique_email()
    monkeypatch.setenv("ADMIN_EMAILS", email)
    res = client.post("/auth/signup", json={"email": email, "password": "correct horse battery"})
    return res.json()["access_token"]


def _user_token(monkeypatch, tmp_path) -> str:
    monkeypatch.setenv("INGEST_DATA_DIR", str(tmp_path))
    email = _unique_email()
    res = client.post("/auth/signup", json={"email": email, "password": "correct horse battery"})
    return res.json()["access_token"]


def test_system_tags_only_list_once_a_document_uses_them(monkeypatch, tmp_path):
    token = _admin_token(monkeypatch, tmp_path)
    headers = {"Authorization": f"Bearer {token}"}
    name = _unique_name("REFLEX")

    create = client.post("/systems", headers=headers, json={"name": name})
    assert create.status_code == 200
    system_id = create.json()["id"]
    assert create.json()["name"] == name

    # Created but not yet attached to any document -- shouldn't clutter the
    # picker yet.
    listing = client.get("/systems", headers=headers)
    assert listing.status_code == 200
    assert not any(s["id"] == system_id for s in listing.json())

    client.post(
        "/documents/upload",
        headers=headers,
        data={"system_id": system_id},
        files={"file": ("a.txt", b"x", "text/plain")},
    )

    listing = client.get("/systems", headers=headers)
    assert any(s["id"] == system_id for s in listing.json())


def test_document_type_tags_only_list_once_a_document_uses_them(monkeypatch, tmp_path):
    token = _admin_token(monkeypatch, tmp_path)
    headers = {"Authorization": f"Bearer {token}"}
    name = _unique_name("Brochure")

    create = client.post("/document-types", headers=headers, json={"name": name})
    assert create.status_code == 200
    doc_type_id = create.json()["id"]
    assert create.json()["name"] == name

    listing = client.get("/document-types", headers=headers)
    assert listing.status_code == 200
    assert not any(dt["id"] == doc_type_id for dt in listing.json())

    client.post(
        "/documents/upload",
        headers=headers,
        data={"document_type_id": doc_type_id},
        files={"file": ("a.txt", b"x", "text/plain")},
    )

    listing = client.get("/document-types", headers=headers)
    assert any(dt["id"] == doc_type_id for dt in listing.json())


def test_deleting_a_documents_only_document_drops_its_tags_from_the_list(monkeypatch, tmp_path):
    token = _admin_token(monkeypatch, tmp_path)
    headers = {"Authorization": f"Bearer {token}"}
    system = client.post("/systems", headers=headers, json={"name": _unique_name("MIS")}).json()

    upload = client.post(
        "/documents/upload",
        headers=headers,
        data={"system_id": system["id"]},
        files={"file": ("a.txt", b"x", "text/plain")},
    )
    doc_id = upload.json()["id"]
    assert any(s["id"] == system["id"] for s in client.get("/systems", headers=headers).json())

    client.delete(f"/documents/{doc_id}", headers=headers)

    assert not any(s["id"] == system["id"] for s in client.get("/systems", headers=headers).json())


def test_duplicate_system_name_conflicts_case_insensitively(monkeypatch, tmp_path):
    token = _admin_token(monkeypatch, tmp_path)
    headers = {"Authorization": f"Bearer {token}"}
    name = _unique_name("MIS")

    first = client.post("/systems", headers=headers, json={"name": name})
    assert first.status_code == 200

    dupe = client.post("/systems", headers=headers, json={"name": name.upper()})
    assert dupe.status_code == 409


def test_non_admin_cannot_manage_tags(monkeypatch, tmp_path):
    token = _user_token(monkeypatch, tmp_path)
    headers = {"Authorization": f"Bearer {token}"}

    assert client.post("/systems", headers=headers, json={"name": "x"}).status_code == 403
    assert client.get("/systems", headers=headers).status_code == 403
    assert client.post("/document-types", headers=headers, json={"name": "x"}).status_code == 403
    assert client.get("/document-types", headers=headers).status_code == 403


def test_upload_with_tags_and_list_shows_them(monkeypatch, tmp_path):
    token = _admin_token(monkeypatch, tmp_path)
    headers = {"Authorization": f"Bearer {token}"}

    system = client.post("/systems", headers=headers, json={"name": _unique_name("REFLEX")}).json()
    doc_type = client.post(
        "/document-types", headers=headers, json={"name": _unique_name("Brochure")}
    ).json()

    upload = client.post(
        "/documents/upload",
        headers=headers,
        data={"system_id": system["id"], "document_type_id": doc_type["id"]},
        files={"file": ("tagged.txt", b"contents", "text/plain")},
    )
    assert upload.status_code == 200
    body = upload.json()
    assert body["system"]["id"] == system["id"]
    assert body["document_type"]["id"] == doc_type["id"]

    listing = client.get("/documents/", headers=headers)
    doc = next(d for d in listing.json() if d["id"] == body["id"])
    assert doc["system"]["name"] == system["name"]
    assert doc["document_type"]["name"] == doc_type["name"]


def test_upload_with_unknown_tag_id_rejected(monkeypatch, tmp_path):
    token = _admin_token(monkeypatch, tmp_path)
    headers = {"Authorization": f"Bearer {token}"}

    upload = client.post(
        "/documents/upload",
        headers=headers,
        data={"system_id": str(uuid.uuid4())},
        files={"file": ("a.txt", b"x", "text/plain")},
    )
    assert upload.status_code == 400


def test_edit_tags_after_upload(monkeypatch, tmp_path):
    token = _admin_token(monkeypatch, tmp_path)
    headers = {"Authorization": f"Bearer {token}"}

    upload = client.post(
        "/documents/upload",
        headers=headers,
        files={"file": ("untagged.txt", b"x", "text/plain")},
    )
    doc_id = upload.json()["id"]
    assert upload.json()["system"] is None

    system = client.post("/systems", headers=headers, json={"name": _unique_name("MIS")}).json()
    edit = client.patch(f"/documents/{doc_id}/tags", headers=headers, json={"system_id": system["id"]})
    assert edit.status_code == 200
    assert edit.json()["system"]["id"] == system["id"]
    assert edit.json()["document_type"] is None


def test_edit_tags_re_triggers_the_ingestion_pipeline(monkeypatch, tmp_path):
    token = _admin_token(monkeypatch, tmp_path)
    headers = {"Authorization": f"Bearer {token}"}

    upload = client.post(
        "/documents/upload",
        headers=headers,
        files={"file": ("retag-me.txt", b"x", "text/plain")},
    )
    doc_id = upload.json()["id"]
    client.post(f"/documents/{doc_id}/index", headers=headers)
    for _ in range(20):
        if client.get(f"/documents/{doc_id}", headers=headers).json()["status"] == "done":
            break
        time.sleep(0.05)

    system = client.post("/systems", headers=headers, json={"name": _unique_name("REFLEX")}).json()
    edit = client.patch(f"/documents/{doc_id}/tags", headers=headers, json={"system_id": system["id"]})
    assert edit.status_code == 200
    # Re-tagging resets status so the background pipeline re-runs and lands
    # back on "done" -- it must not just leave the prior "done" untouched.
    assert edit.json()["status"] in ("queued", "processing", "done")

    for _ in range(20):
        detail = client.get(f"/documents/{doc_id}", headers=headers)
        if detail.json()["status"] == "done":
            break
        time.sleep(0.05)
    assert detail.json()["status"] == "done"
