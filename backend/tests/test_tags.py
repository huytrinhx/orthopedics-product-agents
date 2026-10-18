"""Exercises System/Document-Type tag management and their use on documents
(ticket 05) against a real Postgres -- no mocking, same approach as
test_auth.py / test_documents.py.
"""
import asyncio
import time
import uuid

from fastapi.testclient import TestClient

from api.main import app
from config.db import get_connection
from doc_type_hierarchy.repository import list_hierarchy, set_document_types
from tests.complete_tagging import tag_completely

client = TestClient(app)

# text-embedding-3-small's dimension -- see ..._create_chunks_table.py.
_EMBEDDING_DIM = 1536


def _unique_email() -> str:
    return f"test-{uuid.uuid4().hex[:12]}@example.com"


def _unique_name(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


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


def _upload_tagged(headers: dict, **tags: str) -> str:
    """Upload takes no tags (issue #13) -- a document gets them afterward,
    through the tag modal's PATCH."""
    upload = client.post(
        "/documents/upload", headers=headers, files={"file": ("a.txt", b"x", "text/plain")}
    )
    assert upload.status_code == 200
    document_id = upload.json()["id"]
    edit = client.patch(f"/documents/{document_id}/tags", headers=headers, json=tags)
    assert edit.status_code == 200
    return document_id


def test_system_tags_list_immediately_on_creation(monkeypatch, tmp_path):
    """The picker (and any tag-management UI) shows every tag an admin has
    created, not just ones already attached to a document -- otherwise a
    freshly created tag would be invisible to pick until some other flow
    (upload) referenced it first.
    """
    token = _admin_token(monkeypatch, tmp_path)
    headers = {"Authorization": f"Bearer {token}"}
    name = _unique_name("REFLEX")

    create = client.post("/systems", headers=headers, json={"name": name})
    assert create.status_code == 200
    system_id = create.json()["id"]
    assert create.json()["name"] == name

    listing = client.get("/systems", headers=headers)
    assert listing.status_code == 200
    assert any(listed_system["id"] == system_id for listed_system in listing.json())


def test_document_type_tags_list_immediately_on_creation(monkeypatch, tmp_path):
    token = _admin_token(monkeypatch, tmp_path)
    headers = {"Authorization": f"Bearer {token}"}
    name = _unique_name("Brochure")

    create = client.post("/document-types", headers=headers, json={"name": name})
    assert create.status_code == 200
    document_type_id = create.json()["id"]
    assert create.json()["name"] == name

    listing = client.get("/document-types", headers=headers)
    assert listing.status_code == 200
    assert any(listed_document_type["id"] == document_type_id for listed_document_type in listing.json())


def test_deleting_a_documents_only_document_leaves_its_tags_listed(monkeypatch, tmp_path):
    """Tags are reusable, admin-managed labels, not derived from what's
    currently tagged -- a tag's row (and its visibility in the list) only
    goes away via an explicit delete, never as a side effect of untagging
    the last document that used it.
    """
    token = _admin_token(monkeypatch, tmp_path)
    headers = {"Authorization": f"Bearer {token}"}
    system = client.post("/systems", headers=headers, json={"name": _unique_name("MIS")}).json()

    document_id = _upload_tagged(headers, system_id=system["id"])
    assert any(listed_system["id"] == system["id"] for listed_system in client.get("/systems", headers=headers).json())

    client.delete(f"/documents/{document_id}", headers=headers)

    assert any(listed_system["id"] == system["id"] for listed_system in client.get("/systems", headers=headers).json())


def test_delete_unused_system_tag(monkeypatch, tmp_path):
    token = _admin_token(monkeypatch, tmp_path)
    headers = {"Authorization": f"Bearer {token}"}
    system = client.post("/systems", headers=headers, json={"name": _unique_name("MIS")}).json()

    response = client.delete(f"/systems/{system['id']}", headers=headers)
    assert response.status_code == 204
    assert not any(listed_system["id"] == system["id"] for listed_system in client.get("/systems", headers=headers).json())


def test_delete_unused_document_type_tag(monkeypatch, tmp_path):
    token = _admin_token(monkeypatch, tmp_path)
    headers = {"Authorization": f"Bearer {token}"}
    document_type = client.post(
        "/document-types", headers=headers, json={"name": _unique_name("Brochure")}
    ).json()

    response = client.delete(f"/document-types/{document_type['id']}", headers=headers)
    assert response.status_code == 204
    assert not any(
        listed_document_type["id"] == document_type["id"] for listed_document_type in client.get("/document-types", headers=headers).json()
    )


def test_delete_system_tag_still_in_use_conflicts(monkeypatch, tmp_path):
    token = _admin_token(monkeypatch, tmp_path)
    headers = {"Authorization": f"Bearer {token}"}
    system = client.post("/systems", headers=headers, json={"name": _unique_name("REFLEX")}).json()
    _upload_tagged(headers, system_id=system["id"])

    response = client.delete(f"/systems/{system['id']}", headers=headers)
    assert response.status_code == 409
    # Still there -- the blocked delete didn't half-apply.
    assert any(listed_system["id"] == system["id"] for listed_system in client.get("/systems", headers=headers).json())


def test_delete_document_type_tag_still_in_use_conflicts(monkeypatch, tmp_path):
    token = _admin_token(monkeypatch, tmp_path)
    headers = {"Authorization": f"Bearer {token}"}
    document_type = client.post(
        "/document-types", headers=headers, json={"name": _unique_name("IFU")}
    ).json()
    _upload_tagged(headers, document_type_id=document_type["id"])

    response = client.delete(f"/document-types/{document_type['id']}", headers=headers)
    assert response.status_code == 409


def test_delete_nonexistent_tag_404s(monkeypatch, tmp_path):
    token = _admin_token(monkeypatch, tmp_path)
    headers = {"Authorization": f"Bearer {token}"}
    assert client.delete(f"/systems/{uuid.uuid4()}", headers=headers).status_code == 404
    assert client.delete(f"/document-types/{uuid.uuid4()}", headers=headers).status_code == 404


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
    assert client.delete(f"/systems/{uuid.uuid4()}", headers=headers).status_code == 403
    assert client.post("/document-types", headers=headers, json={"name": "x"}).status_code == 403
    assert client.get("/document-types", headers=headers).status_code == 403
    assert client.delete(f"/document-types/{uuid.uuid4()}", headers=headers).status_code == 403


def test_upload_takes_no_tags_and_lands_pending(monkeypatch, tmp_path):
    """Tags are set in the tag modal after upload (issue #13): tag fields
    sent alongside the file are ignored, not applied."""
    token = _admin_token(monkeypatch, tmp_path)
    headers = {"Authorization": f"Bearer {token}"}
    document_type = client.post(
        "/document-types", headers=headers, json={"name": _unique_name("Brochure")}
    ).json()

    upload = client.post(
        "/documents/upload",
        headers=headers,
        data={"document_type_id": document_type["id"]},
        files={"file": ("tagged.txt", b"contents", "text/plain")},
    )

    assert upload.status_code == 200
    body = upload.json()
    assert body["status"] == "pending"
    assert body["document_type"] is None
    assert body["product_family"] is None
    assert body["trays"] == []


def test_tagged_document_type_shows_in_the_list(monkeypatch, tmp_path):
    token = _admin_token(monkeypatch, tmp_path)
    headers = {"Authorization": f"Bearer {token}"}
    document_type = client.post(
        "/document-types", headers=headers, json={"name": _unique_name("Brochure")}
    ).json()

    document_id = _upload_tagged(headers, document_type_id=document_type["id"])

    listing = client.get("/documents/", headers=headers)
    document = next(listed_document for listed_document in listing.json() if listed_document["id"] == document_id)
    assert document["document_type"]["name"] == document_type["name"]


def test_tagging_with_unknown_tag_id_rejected(monkeypatch, tmp_path):
    token = _admin_token(monkeypatch, tmp_path)
    headers = {"Authorization": f"Bearer {token}"}
    upload = client.post(
        "/documents/upload", headers=headers, files={"file": ("a.txt", b"x", "text/plain")}
    )

    edit = client.patch(
        f"/documents/{upload.json()['id']}/tags",
        headers=headers,
        json={"document_type_id": str(uuid.uuid4())},
    )
    assert edit.status_code == 400


def test_edit_tags_after_upload(monkeypatch, tmp_path):
    token = _admin_token(monkeypatch, tmp_path)
    headers = {"Authorization": f"Bearer {token}"}

    upload = client.post(
        "/documents/upload",
        headers=headers,
        files={"file": ("untagged.txt", b"x", "text/plain")},
    )
    document_id = upload.json()["id"]
    assert upload.json()["system"] is None

    system = client.post("/systems", headers=headers, json={"name": _unique_name("MIS")}).json()
    edit = client.patch(f"/documents/{document_id}/tags", headers=headers, json={"system_id": system["id"]})
    assert edit.status_code == 200
    assert edit.json()["system"]["id"] == system["id"]
    assert edit.json()["document_type"] is None


def test_edit_tags_on_an_indexed_document_does_not_reindex(monkeypatch, tmp_path):
    token = _admin_token(monkeypatch, tmp_path)
    headers = {"Authorization": f"Bearer {token}"}

    upload = client.post(
        "/documents/upload",
        headers=headers,
        files={"file": ("retag-me.txt", b"x", "text/plain")},
    )
    document_id = upload.json()["id"]
    tag_completely(client, headers, document_id)
    client.post(f"/documents/{document_id}/index", headers=headers)
    for _ in range(20):
        if client.get(f"/documents/{document_id}", headers=headers).json()["status"] == "done":
            break
        time.sleep(0.05)
    assert client.get(f"/documents/{document_id}", headers=headers).json()["status"] == "done"

    system = client.post("/systems", headers=headers, json={"name": _unique_name("REFLEX")}).json()
    edit = client.patch(f"/documents/{document_id}/tags", headers=headers, json={"system_id": system["id"]})
    assert edit.status_code == 200
    # Re-tagging updates in place (issue #13) -- no re-queue.
    assert edit.json()["status"] == "done"


def _upload_with_document_type(headers: dict, document_type_id: str) -> str:
    return _upload_tagged(headers, document_type_id=document_type_id)


def _insert_chunk(document_id: str, document_type_id: str) -> None:
    """Writes one chunk row directly, carrying the denormalized
    document_type_id the way ingestion would -- merge has to re-point these
    too, not just the parent document.
    """
    async def _run() -> None:
        connection = await get_connection()
        try:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    "INSERT INTO chunks (document_id, chunk_index, content, embedding, document_type_id) "
                    "VALUES (%s, 0, 'merge me', %s::vector, %s)",
                    (document_id, str([0.0] * _EMBEDDING_DIM), document_type_id),
                )
            await connection.commit()
        finally:
            await connection.close()

    asyncio.run(_run())


def _chunk_document_type_ids(document_id: str) -> list[str]:
    async def _run() -> list[str]:
        connection = await get_connection()
        try:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    "SELECT document_type_id FROM chunks WHERE document_id = %s", (document_id,)
                )
                return [str(row[0]) for row in await cursor.fetchall()]
        finally:
            await connection.close()

    return asyncio.run(_run())


def test_rename_document_type(monkeypatch, tmp_path):
    token = _admin_token(monkeypatch, tmp_path)
    headers = {"Authorization": f"Bearer {token}"}
    document_type = client.post(
        "/document-types", headers=headers, json={"name": _unique_name("Brochure")}
    ).json()
    document_id = _upload_with_document_type(headers, document_type["id"])
    new_name = _unique_name("Product Brochure")

    response = client.patch(
        f"/document-types/{document_type['id']}", headers=headers, json={"name": f"  {new_name}  "}
    )
    assert response.status_code == 200
    assert response.json() == {"id": document_type["id"], "name": new_name}

    # Documents reference the type by id, so they pick up the new display
    # name without being touched.
    document = client.get(f"/documents/{document_id}", headers=headers).json()
    assert document["document_type"]["name"] == new_name


def test_rename_document_type_collision_conflicts_case_insensitively(monkeypatch, tmp_path):
    token = _admin_token(monkeypatch, tmp_path)
    headers = {"Authorization": f"Bearer {token}"}
    existing_name = _unique_name("IFU")
    client.post("/document-types", headers=headers, json={"name": existing_name})
    other = client.post(
        "/document-types", headers=headers, json={"name": _unique_name("Brochure")}
    ).json()

    response = client.patch(
        f"/document-types/{other['id']}", headers=headers, json={"name": existing_name.upper()}
    )
    assert response.status_code == 409
    listed = next(
        listed_document_type
        for listed_document_type in client.get("/document-types", headers=headers).json()
        if listed_document_type["id"] == other["id"]
    )
    assert listed["name"] == other["name"]


def test_rename_document_type_rejects_blank_and_unknown(monkeypatch, tmp_path):
    token = _admin_token(monkeypatch, tmp_path)
    headers = {"Authorization": f"Bearer {token}"}
    document_type = client.post(
        "/document-types", headers=headers, json={"name": _unique_name("Brochure")}
    ).json()

    blank = client.patch(f"/document-types/{document_type['id']}", headers=headers, json={"name": "   "})
    assert blank.status_code == 400
    unknown = client.patch(f"/document-types/{uuid.uuid4()}", headers=headers, json={"name": _unique_name("x")})
    assert unknown.status_code == 404


def test_merge_document_type_repoints_documents_and_chunks_then_deletes_stray(monkeypatch, tmp_path):
    token = _admin_token(monkeypatch, tmp_path)
    headers = {"Authorization": f"Bearer {token}"}
    surviving = client.post(
        "/document-types", headers=headers, json={"name": _unique_name("Surgical Technique")}
    ).json()
    stray = client.post(
        "/document-types", headers=headers, json={"name": _unique_name("Surgical Techniqe")}
    ).json()
    stray_document_id = _upload_with_document_type(headers, stray["id"])
    _insert_chunk(stray_document_id, stray["id"])
    surviving_document_id = _upload_with_document_type(headers, surviving["id"])

    response = client.post(
        f"/document-types/{stray['id']}/merge", headers=headers, json={"into_id": surviving["id"]}
    )
    assert response.status_code == 200
    assert response.json() == surviving

    moved = client.get(f"/documents/{stray_document_id}", headers=headers).json()
    assert moved["document_type"] == surviving
    assert _chunk_document_type_ids(stray_document_id) == [surviving["id"]]
    untouched = client.get(f"/documents/{surviving_document_id}", headers=headers).json()
    assert untouched["document_type"] == surviving

    listed_ids = {
        listed_document_type["id"] for listed_document_type in client.get("/document-types", headers=headers).json()
    }
    assert stray["id"] not in listed_ids
    assert surviving["id"] in listed_ids


def test_merge_into_missing_document_type_changes_nothing(monkeypatch, tmp_path):
    """A merge that can't complete must not have half-applied -- the
    stray's documents and chunks stay on it, and the stray still exists.
    """
    token = _admin_token(monkeypatch, tmp_path)
    headers = {"Authorization": f"Bearer {token}"}
    stray = client.post(
        "/document-types", headers=headers, json={"name": _unique_name("Brochure")}
    ).json()
    document_id = _upload_with_document_type(headers, stray["id"])
    _insert_chunk(document_id, stray["id"])

    response = client.post(
        f"/document-types/{stray['id']}/merge", headers=headers, json={"into_id": str(uuid.uuid4())}
    )
    assert response.status_code == 404

    assert client.get(f"/documents/{document_id}", headers=headers).json()["document_type"] == stray
    assert _chunk_document_type_ids(document_id) == [stray["id"]]
    assert any(
        listed_document_type["id"] == stray["id"]
        for listed_document_type in client.get("/document-types", headers=headers).json()
    )


def test_merge_document_type_merges_its_question_type_hierarchy_rows(monkeypatch, tmp_path):
    """Issue #17: a question type that retrieved from the stray retrieves
    from the survivor after the merge. Re-pointed where the question type
    only listed the stray; deduped (survivor keeps the better rank) where it
    listed both. Uses throwaway question-type slugs so no real question
    type's mapping is touched."""
    token = _admin_token(monkeypatch, tmp_path)
    headers = {"Authorization": f"Bearer {token}"}

    def new_document_type(prefix: str) -> dict:
        return client.post("/document-types", headers=headers, json={"name": _unique_name(prefix)}).json()

    surviving = new_document_type("Surgical Technique")
    stray = new_document_type("Surgical Techniqe")
    other = new_document_type("Setup Guide")
    suffix = uuid.uuid4().hex[:8]
    only_stray = f"test-merge-only-stray-{suffix}"
    stray_ranked_higher = f"test-merge-stray-higher-{suffix}"
    survivor_ranked_higher = f"test-merge-survivor-higher-{suffix}"
    rankings = {
        only_stray: [other["id"], stray["id"]],
        stray_ranked_higher: [stray["id"], other["id"], surviving["id"]],
        survivor_ranked_higher: [surviving["id"], stray["id"], other["id"]],
    }
    for question_type, document_type_ids in rankings.items():
        asyncio.run(set_document_types(question_type, [uuid.UUID(document_type_id) for document_type_id in document_type_ids]))

    try:
        response = client.post(
            f"/document-types/{stray['id']}/merge", headers=headers, json={"into_id": surviving["id"]}
        )
        assert response.status_code == 200

        hierarchy = asyncio.run(list_hierarchy())

        def ranked_ids(question_type: str) -> list[str]:
            return [str(document_type.id) for document_type in hierarchy[question_type]]

        assert ranked_ids(only_stray) == [other["id"], surviving["id"]]
        assert ranked_ids(stray_ranked_higher) == [surviving["id"], other["id"]]
        assert ranked_ids(survivor_ranked_higher) == [surviving["id"], other["id"]]
    finally:
        for question_type in rankings:
            asyncio.run(set_document_types(question_type, []))


def test_merge_document_type_into_itself_rejected(monkeypatch, tmp_path):
    token = _admin_token(monkeypatch, tmp_path)
    headers = {"Authorization": f"Bearer {token}"}
    document_type = client.post(
        "/document-types", headers=headers, json={"name": _unique_name("IFU")}
    ).json()

    response = client.post(
        f"/document-types/{document_type['id']}/merge", headers=headers, json={"into_id": document_type["id"]}
    )
    assert response.status_code == 400


def test_non_admin_cannot_rename_or_merge_document_types(monkeypatch, tmp_path):
    token = _user_token(monkeypatch, tmp_path)
    headers = {"Authorization": f"Bearer {token}"}

    rename = client.patch(f"/document-types/{uuid.uuid4()}", headers=headers, json={"name": "x"})
    assert rename.status_code == 403
    merge = client.post(
        f"/document-types/{uuid.uuid4()}/merge", headers=headers, json={"into_id": str(uuid.uuid4())}
    )
    assert merge.status_code == 403
