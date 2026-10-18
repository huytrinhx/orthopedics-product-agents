"""Issue #14: a document can't be indexed until it's complete (a Product
Family, at least one Tray, a Document Type), and last_indexed_at moves only
when an indexing run finishes successfully -- against real Postgres, no
mocking except to force an indexing failure.
"""
import asyncio
import time
import uuid
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

import documents.service
from api.main import app
from documents.repository import set_status
from tests.complete_tagging import seed_complete_tags, tag_completely

client = TestClient(app)


def _admin_headers(monkeypatch, tmp_path) -> dict:
    monkeypatch.setenv("INGEST_DATA_DIR", str(tmp_path))
    email = f"test-{uuid.uuid4().hex[:12]}@example.com"
    monkeypatch.setenv("ADMIN_EMAILS", email)
    response = client.post("/auth/signup", json={"email": email, "password": "correct horse battery"})
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def _upload(headers: dict, contents: bytes = b"x") -> str:
    upload = client.post(
        "/documents/upload", headers=headers, files={"file": ("doc.txt", contents, "text/plain")}
    )
    assert upload.status_code == 200
    return upload.json()["id"]


def _get(headers: dict, document_id: str) -> dict:
    return client.get(f"/documents/{document_id}", headers=headers).json()


def _wait_until_settled(headers: dict, document_id: str) -> dict:
    for _ in range(40):
        document = _get(headers, document_id)
        if document["status"] in ("done", "failed"):
            return document
        time.sleep(0.05)
    return document


def _index_and_wait(headers: dict, document_id: str) -> dict:
    response = client.post(f"/documents/{document_id}/index", headers=headers)
    assert response.status_code == 200
    return _wait_until_settled(headers, document_id)


def test_untagged_document_lists_every_missing_tag(monkeypatch, tmp_path):
    headers = _admin_headers(monkeypatch, tmp_path)
    document_id = _upload(headers)

    document = _get(headers, document_id)

    assert document["missing_tags"] == ["Product Family", "Tray", "Document Type"]
    assert document["last_indexed_at"] is None


# Each case leaves out exactly one piece of a complete tag set. A document
# with no family can't hold trays either (trays must belong to it), so the
# family case is also missing its tray -- what matters is that the family
# is named.
@pytest.mark.parametrize(
    ("leave_out", "missing_tag"),
    [
        ("family", "Product Family"),
        ("trays", "Tray"),
        ("document_type", "Document Type"),
    ],
)
def test_index_rejects_a_document_missing_a_tag(monkeypatch, tmp_path, leave_out, missing_tag):
    headers = _admin_headers(monkeypatch, tmp_path)
    tags = seed_complete_tags(client, headers)
    document_id = _upload(headers)
    body = {
        "product_family_id": None if leave_out == "family" else str(tags.family.id),
        "tray_ids": (
            [] if leave_out in ("family", "trays") else [str(tray.id) for tray in tags.family.trays]
        ),
        "document_type_id": None if leave_out == "document_type" else tags.document_type["id"],
    }
    assert client.patch(f"/documents/{document_id}/tags", headers=headers, json=body).status_code == 200

    response = client.post(f"/documents/{document_id}/index", headers=headers)

    assert response.status_code == 409
    assert missing_tag in response.json()["detail"]
    assert missing_tag in _get(headers, document_id)["missing_tags"]
    # Refused outright -- nothing queued.
    assert _get(headers, document_id)["status"] == "pending"


def test_complete_document_indexes(monkeypatch, tmp_path):
    headers = _admin_headers(monkeypatch, tmp_path)
    document_id = _upload(headers)
    tag_completely(client, headers, document_id)

    assert _index_and_wait(headers, document_id)["status"] == "done"


def test_incomplete_indexed_document_keeps_its_status_when_reindex_is_refused(monkeypatch, tmp_path):
    """An already-indexed document left incomplete (e.g. by the legacy
    system-tag migration) stays as it is -- still "done", still
    retrievable -- it just can't be re-indexed until it's completed."""
    headers = _admin_headers(monkeypatch, tmp_path)
    document_id = _upload(headers)
    asyncio.run(set_status(uuid.UUID(document_id), "done"))

    response = client.post(f"/documents/{document_id}/index", headers=headers)

    assert response.status_code == 409
    assert _get(headers, document_id)["status"] == "done"


def test_reupload_is_refused_for_an_incomplete_document(monkeypatch, tmp_path):
    """A re-upload re-indexes, so it's behind the same gate -- refused
    before the new file is saved, leaving the old one in place."""
    headers = _admin_headers(monkeypatch, tmp_path)
    document_id = _upload(headers)
    files_before = set(tmp_path.iterdir())

    response = client.post(
        f"/documents/{document_id}/file",
        headers=headers,
        files={"file": ("replacement.txt", b"new", "text/plain")},
    )

    assert response.status_code == 409
    assert "Product Family" in response.json()["detail"]
    assert set(tmp_path.iterdir()) == files_before
    assert _get(headers, document_id)["filename"] == "doc.txt"


def test_successful_index_stamps_last_indexed_at(monkeypatch, tmp_path):
    headers = _admin_headers(monkeypatch, tmp_path)
    document_id = _upload(headers)
    tag_completely(client, headers, document_id)

    queued = client.post(f"/documents/{document_id}/index", headers=headers).json()
    # Queueing alone doesn't count as indexed.
    assert queued["last_indexed_at"] is None

    document = _wait_until_settled(headers, document_id)
    assert document["status"] == "done"
    assert document["last_indexed_at"] is not None

    # A later successful reindex moves it forward.
    reindexed = _index_and_wait(headers, document_id)
    assert datetime.fromisoformat(reindexed["last_indexed_at"]) > datetime.fromisoformat(
        document["last_indexed_at"]
    )


def test_failed_index_leaves_last_indexed_at_alone(monkeypatch, tmp_path):
    headers = _admin_headers(monkeypatch, tmp_path)
    document_id = _upload(headers)
    tag_completely(client, headers, document_id)
    indexed = _index_and_wait(headers, document_id)
    assert indexed["last_indexed_at"] is not None

    def _fail_extraction(*arguments, **keyword_arguments):
        raise RuntimeError("extraction exploded")

    monkeypatch.setattr(documents.service, "extract_text", _fail_extraction)
    failed = _index_and_wait(headers, document_id)

    assert failed["status"] == "failed"
    assert failed["last_indexed_at"] == indexed["last_indexed_at"]


def test_failed_first_index_never_stamps_last_indexed_at(monkeypatch, tmp_path):
    headers = _admin_headers(monkeypatch, tmp_path)
    document_id = _upload(headers)
    tag_completely(client, headers, document_id)

    def _fail_extraction(*arguments, **keyword_arguments):
        raise RuntimeError("extraction exploded")

    monkeypatch.setattr(documents.service, "extract_text", _fail_extraction)
    failed = _index_and_wait(headers, document_id)

    assert failed["status"] == "failed"
    assert failed["last_indexed_at"] is None


def test_tag_edit_leaves_last_indexed_at_alone(monkeypatch, tmp_path):
    headers = _admin_headers(monkeypatch, tmp_path)
    document_id = _upload(headers)
    tag_completely(client, headers, document_id)
    indexed = _index_and_wait(headers, document_id)
    other_type = client.post(
        "/document-types", headers=headers, json={"name": f"IFU-{uuid.uuid4().hex[:8]}"}
    ).json()

    edit = client.patch(
        f"/documents/{document_id}/tags",
        headers=headers,
        json={"document_type_id": other_type["id"]},
    )

    assert edit.status_code == 200
    assert edit.json()["last_indexed_at"] == indexed["last_indexed_at"]
