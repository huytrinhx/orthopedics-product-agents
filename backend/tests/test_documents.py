"""Exercises upload/list/status against a real Postgres + real disk writes
under a temp INGEST_DATA_DIR — no mocking, same approach as test_auth.py.
"""
import os
import time
import uuid

import pytest
from fastapi.testclient import TestClient

import api.routes.documents
from api.main import app
from tests.complete_tagging import tag_completely
from tests.unique_filenames import unique_filename

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
    response = client.post("/auth/signup", json={"email": email, "password": "correct horse battery"})
    return response.json()["access_token"]


def _user_token(monkeypatch, tmp_path) -> str:
    monkeypatch.setenv("INGEST_DATA_DIR", str(tmp_path))
    email = _unique_email()
    response = client.post("/auth/signup", json={"email": email, "password": "correct horse battery"})
    return response.json()["access_token"]


def test_non_admin_cannot_upload_or_list(monkeypatch, tmp_path):
    token = _user_token(monkeypatch, tmp_path)
    headers = {"Authorization": f"Bearer {token}"}

    upload = client.post(
        "/documents/upload",
        headers=headers,
        files={"file": (unique_filename("a.txt"), b"hello", "text/plain")},
    )
    assert upload.status_code == 403

    listing = client.get("/documents/", headers=headers)
    assert listing.status_code == 403


def test_health_check_reports_all_components_ok(monkeypatch, tmp_path):
    """Real round-trips against the local test infra (docker-compose's
    postgres-test/neo4j-test, see tests/conftest.py) -- not mocked, since the
    point of this endpoint is proving those connections actually work, not
    just that the code compiles.
    """
    token = _admin_token(monkeypatch, tmp_path)
    headers = {"Authorization": f"Bearer {token}"}

    response = client.get("/documents/health", headers=headers)
    assert response.status_code == 200
    body = response.json()
    assert body["volume"] == {"ok": True, "detail": "0 files"}
    assert body["graph_db"]["ok"] is True
    assert body["vector_db"]["ok"] is True


def test_health_check_volume_reflects_file_count(monkeypatch, tmp_path):
    token = _admin_token(monkeypatch, tmp_path)
    headers = {"Authorization": f"Bearer {token}"}

    for name in ("a.txt", "b.txt"):
        client.post(
            "/documents/upload",
            headers=headers,
            files={"file": (unique_filename(name), b"x", "text/plain")},
        )

    response = client.get("/documents/health", headers=headers)
    assert response.json()["volume"] == {"ok": True, "detail": "2 files"}


def test_non_admin_cannot_check_health(monkeypatch, tmp_path):
    token = _user_token(monkeypatch, tmp_path)
    headers = {"Authorization": f"Bearer {token}"}
    assert client.get("/documents/health", headers=headers).status_code == 403


def test_admin_upload_appears_in_list_pending_until_indexed(monkeypatch, tmp_path):
    token = _admin_token(monkeypatch, tmp_path)
    headers = {"Authorization": f"Bearer {token}"}

    filename = unique_filename("brochure.txt")
    upload = client.post(
        "/documents/upload",
        headers=headers,
        files={"file": (filename, b"MIS brochure contents", "text/plain")},
    )
    assert upload.status_code == 200
    body = upload.json()
    assert body["filename"] == filename
    assert body["status"] == "pending"

    # The file landed on disk under the temp INGEST_DATA_DIR.
    assert any(tmp_path.iterdir())

    listing = client.get("/documents/", headers=headers)
    assert listing.status_code == 200
    assert any(listed_document["id"] == body["id"] for listed_document in listing.json())

    # Indexing only runs once triggered -- via the Index/Reindex button's
    # endpoint, not as a side effect of upload -- and only once it's
    # completely tagged (issue #14, see test_indexing_gate.py).
    document_id = body["id"]
    tag_completely(client, headers, document_id)
    index =client.post(f"/documents/{document_id}/index", headers=headers)
    assert index.status_code == 200
    assert index.json()["status"] in ("queued", "processing", "done")

    # Background task runs inline-ish via TestClient/BackgroundTasks; poll
    # briefly rather than assuming it's already finished by the time we ask.
    for _ in range(20):
        detail = client.get(f"/documents/{document_id}", headers=headers)
        if detail.json()["status"] == "done":
            break
        time.sleep(0.05)
    assert detail.json()["status"] == "done"


def test_non_admin_cannot_index(monkeypatch, tmp_path):
    admin_headers = {"Authorization": f"Bearer {_admin_token(monkeypatch, tmp_path)}"}
    upload = client.post(
        "/documents/upload",
        headers=admin_headers,
        files={"file": (unique_filename("a.txt"), b"x", "text/plain")},
    )
    document_id = upload.json()["id"]

    user_headers = {"Authorization": f"Bearer {_user_token(monkeypatch, tmp_path)}"}
    assert client.post(f"/documents/{document_id}/index", headers=user_headers).status_code == 403


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
        files={"file": (unique_filename("delete-me.txt"), b"contents", "text/plain")},
    )
    document_id = upload.json()["id"]
    assert any(tmp_path.iterdir())

    delete = client.delete(f"/documents/{document_id}", headers=headers)
    assert delete.status_code == 204

    assert client.get(f"/documents/{document_id}", headers=headers).status_code == 404
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
        files={"file": (unique_filename("a.txt"), b"x", "text/plain")},
    )
    document_id = upload.json()["id"]

    user_headers = {"Authorization": f"Bearer {_user_token(monkeypatch, tmp_path)}"}
    assert client.delete(f"/documents/{document_id}", headers=user_headers).status_code == 403


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
        files={"file": (unique_filename("a.txt"), b"hello world", "text/plain")},
    )
    document_id = upload.json()["id"]
    tag_completely(client, admin_headers, document_id)
    client.post(f"/documents/{document_id}/index", headers=admin_headers)
    for _ in range(20):
        if client.get(f"/documents/{document_id}", headers=admin_headers).json()["status"] == "done":
            break
        time.sleep(0.05)

    user_headers = {"Authorization": f"Bearer {_user_token(monkeypatch, tmp_path)}"}
    response = client.get(f"/documents/{document_id}/chunks", headers=user_headers)
    assert response.status_code == 200
    # Without OPENAI_API_KEY, ingest_document_vectors no-ops (see
    # ingestion/pipeline.py) -- this only exercises the route/auth, not real
    # chunk content, which test_chat_routes.py's citation tests cover
    # end-to-end under needs_openai_key.
    assert isinstance(response.json(), list)


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
        files={"file": (unique_filename("torque-spec.txt"), content, "text/plain")},
    )
    document_id = upload.json()["id"]
    tag_completely(client, admin_headers, document_id)
    client.post(f"/documents/{document_id}/index", headers=admin_headers)
    for _ in range(40):
        if client.get(f"/documents/{document_id}", headers=admin_headers).json()["status"] == "done":
            break
        time.sleep(0.25)

    user_headers = {"Authorization": f"Bearer {_user_token(monkeypatch, tmp_path)}"}
    chunks = client.get(f"/documents/{document_id}/chunks", headers=user_headers).json()

    assert chunks
    assert [chunk["chunk_index"] for chunk in chunks] == sorted(chunk["chunk_index"] for chunk in chunks)
    assert any("1.2" in chunk["content"] for chunk in chunks)


def test_admin_can_reupload_document_file(monkeypatch, tmp_path):
    token = _admin_token(monkeypatch, tmp_path)
    headers = {"Authorization": f"Bearer {token}"}

    upload = client.post(
        "/documents/upload",
        headers=headers,
        files={"file": (unique_filename("original.txt"), b"original contents", "text/plain")},
    )
    document_id = upload.json()["id"]
    tag_completely(client, headers, document_id)
    original_files = set(tmp_path.iterdir())

    replacement_filename = unique_filename("replacement.txt")
    reupload = client.post(
        f"/documents/{document_id}/file",
        headers=headers,
        files={"file": (replacement_filename, b"replacement contents", "text/plain")},
    )
    assert reupload.status_code == 200
    body = reupload.json()
    assert body["id"] == document_id
    # Same id, new display filename, and queued for reindexing against the
    # new file -- same pipeline seam as Index/Reindex and re-tagging.
    assert body["filename"] == replacement_filename
    assert body["status"] in ("queued", "processing", "done")

    # The old file is gone and a new one replaced it -- not both kept.
    current_files = set(tmp_path.iterdir())
    assert not (original_files & current_files)
    assert len(current_files) == 1

    for _ in range(20):
        detail = client.get(f"/documents/{document_id}", headers=headers)
        if detail.json()["status"] == "done":
            break
        time.sleep(0.05)
    assert detail.json()["status"] == "done"


def test_reupload_unknown_document_404s(monkeypatch, tmp_path):
    token = _admin_token(monkeypatch, tmp_path)
    headers = {"Authorization": f"Bearer {token}"}
    response = client.post(
        f"/documents/{uuid.uuid4()}/file",
        headers=headers,
        files={"file": (unique_filename("a.txt"), b"x", "text/plain")},
    )
    assert response.status_code == 404


def test_non_admin_cannot_reupload(monkeypatch, tmp_path):
    admin_headers = {"Authorization": f"Bearer {_admin_token(monkeypatch, tmp_path)}"}
    upload = client.post(
        "/documents/upload",
        headers=admin_headers,
        files={"file": (unique_filename("a.txt"), b"x", "text/plain")},
    )
    document_id = upload.json()["id"]

    user_headers = {"Authorization": f"Bearer {_user_token(monkeypatch, tmp_path)}"}
    response = client.post(
        f"/documents/{document_id}/file",
        headers=user_headers,
        files={"file": (unique_filename("b.txt"), b"y", "text/plain")},
    )
    assert response.status_code == 403


def test_admin_can_download_document_file(monkeypatch, tmp_path):
    token = _admin_token(monkeypatch, tmp_path)
    headers = {"Authorization": f"Bearer {token}"}

    upload = client.post(
        "/documents/upload",
        headers=headers,
        files={"file": (unique_filename("brochure.txt"), b"downloadable contents", "text/plain")},
    )
    document_id = upload.json()["id"]

    download = client.get(f"/documents/{document_id}/file", headers=headers)
    assert download.status_code == 200
    assert download.content == b"downloadable contents"


def test_upload_requires_a_filename(monkeypatch, tmp_path):
    token = _admin_token(monkeypatch, tmp_path)
    headers = {"Authorization": f"Bearer {token}"}
    response = client.post(
        "/documents/upload", headers=headers, files={"file": ("", b"", "text/plain")}
    )
    assert response.status_code in (400, 422)


# Issue #31: a filename belongs to one document, ignoring case and
# surrounding whitespace.


def _upload_named(headers: dict, filename: str, contents: bytes = b"x"):
    return client.post(
        "/documents/upload", headers=headers, files={"file": (filename, contents, "text/plain")}
    )


def _filenames_listed(headers: dict) -> list[str]:
    listing = client.get("/documents/", headers=headers).json()
    return [listed_document["filename"] for listed_document in listing]


def test_uploading_a_duplicate_filename_is_refused_naming_the_existing_document(
    monkeypatch, tmp_path
):
    headers = {"Authorization": f"Bearer {_admin_token(monkeypatch, tmp_path)}"}
    filename = unique_filename("Surgical Technique.pdf")
    existing = _upload_named(headers, filename).json()
    tagged = tag_completely(client, headers, existing["id"])
    files_before = set(tmp_path.iterdir())

    duplicate = _upload_named(headers, filename, b"second copy")

    assert duplicate.status_code == 409
    detail = duplicate.json()["detail"]
    assert filename in detail["message"]
    assert detail["existing_document"] == {
        "id": existing["id"],
        "filename": filename,
        "document_type": tagged["document_type"],
        "product_family": tagged["product_family"],
        "uploaded_at": existing["created_at"],
    }
    # Nothing written: no new file on disk, no second row.
    assert set(tmp_path.iterdir()) == files_before
    assert _filenames_listed(headers).count(filename) == 1


def test_duplicate_filenames_match_ignoring_case_and_surrounding_whitespace(
    monkeypatch, tmp_path
):
    headers = {"Authorization": f"Bearer {_admin_token(monkeypatch, tmp_path)}"}
    filename = unique_filename("Brochure.pdf")
    existing_id = _upload_named(headers, filename).json()["id"]

    for variant in (filename.upper(), filename.lower(), f"  {filename} "):
        duplicate = _upload_named(headers, variant)
        assert duplicate.status_code == 409, variant
        assert duplicate.json()["detail"]["existing_document"]["id"] == existing_id


def test_replacing_a_documents_file_with_its_own_name_is_allowed(monkeypatch, tmp_path):
    headers = {"Authorization": f"Bearer {_admin_token(monkeypatch, tmp_path)}"}
    filename = unique_filename("ifu.txt")
    document_id = _upload_named(headers, filename).json()["id"]
    tag_completely(client, headers, document_id)

    for replacement_name in (filename, filename.upper()):
        reupload = client.post(
            f"/documents/{document_id}/file",
            headers=headers,
            files={"file": (replacement_name, b"corrected contents", "text/plain")},
        )
        assert reupload.status_code == 200, replacement_name
        assert reupload.json()["filename"] == replacement_name


def test_replacing_a_documents_file_with_another_documents_name_is_refused(
    monkeypatch, tmp_path
):
    headers = {"Authorization": f"Bearer {_admin_token(monkeypatch, tmp_path)}"}
    other_filename = unique_filename("other.txt")
    other_id = _upload_named(headers, other_filename).json()["id"]
    own_filename = unique_filename("own.txt")
    document_id = _upload_named(headers, own_filename).json()["id"]
    tag_completely(client, headers, document_id)
    files_before = set(tmp_path.iterdir())

    reupload = client.post(
        f"/documents/{document_id}/file",
        headers=headers,
        files={"file": (other_filename.upper(), b"new", "text/plain")},
    )

    assert reupload.status_code == 409
    assert reupload.json()["detail"]["existing_document"]["id"] == other_id
    assert set(tmp_path.iterdir()) == files_before
    document = client.get(f"/documents/{document_id}", headers=headers).json()
    assert document["filename"] == own_filename


async def _no_document_found(filename, excluding_document_id=None):
    return None


def test_a_concurrent_duplicate_upload_gets_the_same_409(monkeypatch, tmp_path):
    """Two uploads racing past the route's check: the unique index refuses
    the second insert, and it gets the same 409 with its saved file
    removed again. Skipping the check stands in for losing the race."""
    headers = {"Authorization": f"Bearer {_admin_token(monkeypatch, tmp_path)}"}
    filename = unique_filename("race.txt")
    existing_id = _upload_named(headers, filename).json()["id"]
    files_before = set(tmp_path.iterdir())
    monkeypatch.setattr(api.routes.documents, "find_document_by_filename", _no_document_found)

    duplicate = _upload_named(headers, f" {filename.upper()}")

    assert duplicate.status_code == 409
    assert duplicate.json()["detail"]["existing_document"]["id"] == existing_id
    assert set(tmp_path.iterdir()) == files_before
    assert _filenames_listed(headers).count(filename) == 1


def test_a_concurrent_duplicate_file_replacement_gets_the_same_409(monkeypatch, tmp_path):
    headers = {"Authorization": f"Bearer {_admin_token(monkeypatch, tmp_path)}"}
    other_filename = unique_filename("race-other.txt")
    other_id = _upload_named(headers, other_filename).json()["id"]
    own_filename = unique_filename("race-own.txt")
    document_id = _upload_named(headers, own_filename).json()["id"]
    tag_completely(client, headers, document_id)
    files_before = set(tmp_path.iterdir())
    monkeypatch.setattr(api.routes.documents, "find_document_by_filename", _no_document_found)

    reupload = client.post(
        f"/documents/{document_id}/file",
        headers=headers,
        files={"file": (other_filename, b"new", "text/plain")},
    )

    assert reupload.status_code == 409
    assert reupload.json()["detail"]["existing_document"]["id"] == other_id
    assert set(tmp_path.iterdir()) == files_before
    document = client.get(f"/documents/{document_id}", headers=headers).json()
    assert document["filename"] == own_filename
