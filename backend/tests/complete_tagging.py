"""Shared by any test that indexes a document: indexing is refused until a
document is complete -- a Product Family, at least one Tray, and a
Document Type (issue #14) -- so those tests tag it first through the same
PATCH /documents/{id}/tags the tag modal uses.
"""
import asyncio
import uuid
from dataclasses import dataclass

from fastapi.testclient import TestClient

from catalog.repository import (
    ProductFamilyRecord,
    list_families_with_trays,
    upsert_families_and_trays,
)


@dataclass
class CompleteTags:
    family: ProductFamilyRecord
    document_type: dict


def _unique_name(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def seed_complete_tags(client: TestClient, headers: dict, tray_count: int = 1) -> CompleteTags:
    """A fresh family (with `tray_count` trays) and document type to tag
    with -- unique names, so tests never collide on shared catalog rows."""
    family_name = _unique_name("Family")
    asyncio.run(
        upsert_families_and_trays(
            {family_name: [_unique_name(f"Tray{index}") for index in range(tray_count)]}
        )
    )
    family = next(
        family for family in asyncio.run(list_families_with_trays()) if family.name == family_name
    )
    document_type = client.post(
        "/document-types", headers=headers, json={"name": _unique_name("Brochure")}
    ).json()
    return CompleteTags(family=family, document_type=document_type)


def tag_completely(
    client: TestClient, headers: dict, document_id: str, tags: CompleteTags | None = None
) -> dict:
    """Tags the document as complete (seeding tags if none are given) and
    returns the updated document."""
    tags = tags or seed_complete_tags(client, headers)
    response = client.patch(
        f"/documents/{document_id}/tags",
        headers=headers,
        json={
            "product_family_id": str(tags.family.id),
            "tray_ids": [str(tray.id) for tray in tags.family.trays],
            "document_type_id": tags.document_type["id"],
        },
    )
    assert response.status_code == 200
    assert response.json()["missing_tags"] == []
    return response.json()
