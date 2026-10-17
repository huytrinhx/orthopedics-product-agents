"""Tagging a document with one Product Family and many Trays (issue #12)
through PATCH /documents/{id}/tags -- against real Postgres, no mocking.
"""
import uuid

from fastapi.testclient import TestClient

import api.routes.documents
from api.main import app
from catalog.repository import (
    ProductFamilyRecord,
    list_families_with_trays,
    upsert_families_and_trays,
)
from documents.repository import set_status
from retrieval.graph_client import get_graph_client
from retrieval.vector_store import get_vector_store

client = TestClient(app)

# text-embedding-3-small's dimension -- see ..._create_chunks_table.py.
_EMBEDDING_DIM = 1536


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


async def _chunk_rows(document_id: str) -> list[tuple]:
    async with get_vector_store() as store, store._connection.cursor() as cursor:
        await cursor.execute(
            "SELECT chunk_index, content, embedding::text, product_family_id, document_type_id "
            "FROM chunks WHERE document_id = %s ORDER BY chunk_index",
            (document_id,),
        )
        return await cursor.fetchall()


async def test_retagging_an_indexed_document_updates_chunks_and_graph_in_place(
    monkeypatch, tmp_path
):
    """Issue #13: the tag modal's save rewrites the chunks' filter columns
    and the graph Document node's properties without re-indexing -- no
    process_document run, no status change, chunk text and embeddings
    untouched."""
    headers = _admin_headers(monkeypatch, tmp_path)
    old_family = await _seed_family(tray_count=1)
    new_family = await _seed_family(tray_count=2)
    old_type, new_type = (
        client.post("/document-types", headers=headers, json={"name": _unique_name(prefix)}).json()
        for prefix in ("Brochure", "Technique Guide")
    )
    document_id = _upload(headers)
    client.patch(
        f"/documents/{document_id}/tags",
        headers=headers,
        json={
            "product_family_id": str(old_family.id),
            "tray_ids": [str(old_family.trays[0].id)],
            "document_type_id": old_type["id"],
        },
    )

    # Stand in for a finished index run: chunks carrying the old tags, the
    # graph node ingestion would have written, and status "done".
    async with get_vector_store() as store:
        await store.upsert_chunks(
            document_id,
            [
                {
                    "chunk_index": index,
                    "content": f"chunk {index}",
                    "embedding": [float(index)] * _EMBEDDING_DIM,
                    "product_family_id": old_family.id,
                    "document_type_id": uuid.UUID(old_type["id"]),
                }
                for index in range(2)
            ],
        )
    graph = get_graph_client()
    await graph.upsert_document(
        document_id,
        "doc.txt",
        doc_type=old_type["name"],
        product_family=old_family.name,
        trays=[old_family.trays[0].name],
    )
    await set_status(uuid.UUID(document_id), "done")
    chunks_before = await _chunk_rows(document_id)

    def _fail_reindex(*arguments, **keyword_arguments):
        raise AssertionError("re-tagging must not re-index the document")

    monkeypatch.setattr(api.routes.documents, "process_document", _fail_reindex)

    edit = client.patch(
        f"/documents/{document_id}/tags",
        headers=headers,
        json={
            "product_family_id": str(new_family.id),
            "tray_ids": [str(tray.id) for tray in new_family.trays],
            "document_type_id": new_type["id"],
        },
    )

    assert edit.status_code == 200
    assert edit.json()["status"] == "done"
    assert edit.json()["product_family"]["id"] == str(new_family.id)
    assert edit.json()["document_type"]["id"] == new_type["id"]

    chunks_after = await _chunk_rows(document_id)
    # Same chunks, same text, same vectors -- only the tag columns moved.
    assert [row[:3] for row in chunks_after] == [row[:3] for row in chunks_before]
    assert {row[3] for row in chunks_after} == {new_family.id}
    assert {row[4] for row in chunks_after} == {uuid.UUID(new_type["id"])}

    properties = await graph.get_document_properties(document_id)
    assert properties["product_family"] == new_family.name
    assert sorted(properties["trays"]) == sorted(tray.name for tray in new_family.trays)
    assert properties["doc_type"] == new_type["name"]
    assert properties["filename"] == "doc.txt"


async def test_retagging_a_never_indexed_document_creates_no_graph_node(monkeypatch, tmp_path):
    headers = _admin_headers(monkeypatch, tmp_path)
    family = await _seed_family(tray_count=1)
    document_id = _upload(headers)

    edit = client.patch(
        f"/documents/{document_id}/tags",
        headers=headers,
        json={"product_family_id": str(family.id), "tray_ids": [str(family.trays[0].id)]},
    )

    assert edit.status_code == 200
    assert edit.json()["status"] == "pending"
    assert await get_graph_client().get_document_properties(document_id) is None
