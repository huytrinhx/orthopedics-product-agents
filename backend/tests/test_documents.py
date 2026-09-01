"""Exercises upload/list/status against a real Postgres + real disk writes
under a temp INGEST_DATA_DIR — no mocking, same approach as test_auth.py.
"""
import os
import time
import uuid

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

needs_openai_key = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), reason="requires a real OPENAI_API_KEY"
)


def _unique_email() -> str:
    return f"test-{uuid.uuid4().hex[:12]}@example.com"


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


def test_non_admin_cannot_upload_or_list(monkeypatch, tmp_path):
    token = _user_token(monkeypatch, tmp_path)
    headers = {"Authorization": f"Bearer {token}"}

    upload = client.post(
        "/documents/upload", headers=headers, files={"file": ("a.txt", b"hello", "text/plain")}
    )
    assert upload.status_code == 403

    listing = client.get("/documents/", headers=headers)
    assert listing.status_code == 403


def test_admin_upload_appears_in_list_pending_until_indexed(monkeypatch, tmp_path):
    token = _admin_token(monkeypatch, tmp_path)
    headers = {"Authorization": f"Bearer {token}"}

    upload = client.post(
        "/documents/upload",
        headers=headers,
        files={"file": ("brochure.txt", b"MIS brochure contents", "text/plain")},
    )
    assert upload.status_code == 200
    body = upload.json()
    assert body["filename"] == "brochure.txt"
    assert body["status"] == "pending"

    # The file landed on disk under the temp INGEST_DATA_DIR.
    assert any(tmp_path.iterdir())

    listing = client.get("/documents/", headers=headers)
    assert listing.status_code == 200
    assert any(d["id"] == body["id"] for d in listing.json())

    # Indexing only runs once triggered -- via the Index/Reindex button's
    # endpoint, not as a side effect of upload.
    doc_id = body["id"]
    index = client.post(f"/documents/{doc_id}/index", headers=headers)
    assert index.status_code == 200
    assert index.json()["status"] in ("queued", "processing", "done")

    # Background task runs inline-ish via TestClient/BackgroundTasks; poll
    # briefly rather than assuming it's already finished by the time we ask.
    for _ in range(20):
        detail = client.get(f"/documents/{doc_id}", headers=headers)
        if detail.json()["status"] == "done":
            break
        time.sleep(0.05)
    assert detail.json()["status"] == "done"


def test_non_admin_cannot_index(monkeypatch, tmp_path):
    admin_headers = {"Authorization": f"Bearer {_admin_token(monkeypatch, tmp_path)}"}
    upload = client.post(
        "/documents/upload",
        headers=admin_headers,
        files={"file": ("a.txt", b"x", "text/plain")},
    )
    doc_id = upload.json()["id"]

    user_headers = {"Authorization": f"Bearer {_user_token(monkeypatch, tmp_path)}"}
    assert client.post(f"/documents/{doc_id}/index", headers=user_headers).status_code == 403


def test_index_unknown_document_404s(monkeypatch, tmp_path):
    token = _admin_token(monkeypatch, tmp_path)
    headers = {"Authorization": f"Bearer {token}"}
    assert client.post(f"/documents/{uuid.uuid4()}/index", headers=headers).status_code == 404


def test_admin_can_delete_a_document(monkeypatch, tmp_path):
    token = _admin_token(monkeypatch, tmp_path)
    headers = {"Authorization": f"Bearer {token}"}

    upload = client.post(
        "/documents/upload",
        headers=headers,
        files={"file": ("delete-me.txt", b"contents", "text/plain")},
    )
    doc_id = upload.json()["id"]
    assert any(tmp_path.iterdir())

    delete = client.delete(f"/documents/{doc_id}", headers=headers)
    assert delete.status_code == 204

    assert client.get(f"/documents/{doc_id}", headers=headers).status_code == 404
    assert not any(tmp_path.iterdir())


def test_delete_unknown_document_404s(monkeypatch, tmp_path):
    token = _admin_token(monkeypatch, tmp_path)
    headers = {"Authorization": f"Bearer {token}"}
    assert client.delete(f"/documents/{uuid.uuid4()}", headers=headers).status_code == 404


def test_non_admin_cannot_delete(monkeypatch, tmp_path):
    admin_headers = {"Authorization": f"Bearer {_admin_token(monkeypatch, tmp_path)}"}
    upload = client.post(
        "/documents/upload",
        headers=admin_headers,
        files={"file": ("a.txt", b"x", "text/plain")},
    )
    doc_id = upload.json()["id"]

    user_headers = {"Authorization": f"Bearer {_user_token(monkeypatch, tmp_path)}"}
    assert client.delete(f"/documents/{doc_id}", headers=user_headers).status_code == 403


def test_chunks_requires_auth(monkeypatch, tmp_path):
    monkeypatch.setenv("INGEST_DATA_DIR", str(tmp_path))
    assert client.get(f"/documents/{uuid.uuid4()}/chunks").status_code == 401


def test_chunks_unknown_document_404s(monkeypatch, tmp_path):
    token = _user_token(monkeypatch, tmp_path)
    headers = {"Authorization": f"Bearer {token}"}
    assert client.get(f"/documents/{uuid.uuid4()}/chunks", headers=headers).status_code == 404


def test_a_non_admin_can_read_chunks(monkeypatch, tmp_path):
    """Unlike every other /documents route, GET .../chunks isn't
    admin-gated -- it backs the chat citation viewer, which any logged-in
    user reaches (see api/routes/documents.py's module docstring)."""
    admin_headers = {"Authorization": f"Bearer {_admin_token(monkeypatch, tmp_path)}"}
    upload = client.post(
        "/documents/upload",
        headers=admin_headers,
        files={"file": ("a.txt", b"hello world", "text/plain")},
    )
    doc_id = upload.json()["id"]
    client.post(f"/documents/{doc_id}/index", headers=admin_headers)
    for _ in range(20):
        if client.get(f"/documents/{doc_id}", headers=admin_headers).json()["status"] == "done":
            break
        time.sleep(0.05)

    user_headers = {"Authorization": f"Bearer {_user_token(monkeypatch, tmp_path)}"}
    res = client.get(f"/documents/{doc_id}/chunks", headers=user_headers)
    assert res.status_code == 200
    # Without OPENAI_API_KEY, ingest_document_vectors no-ops (see
    # ingestion/pipeline.py) -- this only exercises the route/auth, not real
    # chunk content, which test_chat_routes.py's citation tests cover
    # end-to-end under needs_openai_key.
    assert isinstance(res.json(), list)


@needs_openai_key
def test_chunks_content_matches_the_indexed_document(monkeypatch, tmp_path):
    admin_headers = {"Authorization": f"Bearer {_admin_token(monkeypatch, tmp_path)}"}
    content = (
        b"REFLEX HYBRID Torque Spec\n\n"
        b"The REFLEX HYBRID driver requires 1.2 Nm of torque when seating the "
        b"4.0mm locking screw. Do not exceed 1.5 Nm or the screw head may strip."
    )
    upload = client.post(
        "/documents/upload",
        headers=admin_headers,
        files={"file": ("torque-spec.txt", content, "text/plain")},
    )
    doc_id = upload.json()["id"]
    client.post(f"/documents/{doc_id}/index", headers=admin_headers)
    for _ in range(40):
        if client.get(f"/documents/{doc_id}", headers=admin_headers).json()["status"] == "done":
            break
        time.sleep(0.25)

    user_headers = {"Authorization": f"Bearer {_user_token(monkeypatch, tmp_path)}"}
    chunks = client.get(f"/documents/{doc_id}/chunks", headers=user_headers).json()

    assert chunks
    assert [c["chunk_index"] for c in chunks] == sorted(c["chunk_index"] for c in chunks)
    assert any("1.2" in c["content"] for c in chunks)


def test_upload_requires_a_filename(monkeypatch, tmp_path):
    token = _admin_token(monkeypatch, tmp_path)
    headers = {"Authorization": f"Bearer {token}"}
    res = client.post(
        "/documents/upload", headers=headers, files={"file": ("", b"", "text/plain")}
    )
    assert res.status_code in (400, 422)
