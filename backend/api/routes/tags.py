"""Document-Type tag management (ticket 05). Admin-only, same as the rest of
document management (see auth.dependencies.require_admin). No fixed enum --
an admin grows the list as new document types show up.

Document Types are a closed, admin-curated list (issue #16): the Settings
page (frontend/app/admin/) is the only UI that creates, renames, merges, or
deletes them; the Document Manager only picks from the list.
"""
import uuid

import psycopg
from fastapi import APIRouter, Depends, HTTPException, status

from auth.dependencies import require_admin
from auth.repository import UserRecord
from tags.models import CreateTagRequest, MergeTagRequest, RenameTagRequest, TagOut
from tags.repository import (
    TagNotFound,
    create_document_type,
    delete_document_type,
    list_document_types,
    merge_document_type,
    rename_document_type,
)

router = APIRouter()


def _tag_out(tag) -> TagOut:
    return TagOut(id=tag.id, name=tag.name)


@router.post("/document-types", response_model=TagOut)
async def create_document_type_tag(
    body: CreateTagRequest, admin: UserRecord = Depends(require_admin)
) -> TagOut:
    name = body.name.strip()
    if not name:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Name is required")
    try:
        return _tag_out(await create_document_type(name))
    except psycopg.errors.UniqueViolation:
        raise HTTPException(
            status.HTTP_409_CONFLICT, f"A document type named '{name}' already exists"
        )


@router.get("/document-types", response_model=list[TagOut])
async def list_document_type_tags(admin: UserRecord = Depends(require_admin)) -> list[TagOut]:
    return [_tag_out(tag) for tag in await list_document_types()]


@router.delete("/document-types/{tag_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document_type_tag(
    tag_id: uuid.UUID, admin: UserRecord = Depends(require_admin)
) -> None:
    try:
        deleted = await delete_document_type(tag_id)
    except psycopg.errors.ForeignKeyViolation:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "This document type is still assigned to one or more documents",
        )
    if not deleted:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document type not found")


@router.patch("/document-types/{tag_id}", response_model=TagOut)
async def rename_document_type_tag(
    tag_id: uuid.UUID, body: RenameTagRequest, admin: UserRecord = Depends(require_admin)
) -> TagOut:
    name = body.name.strip()
    if not name:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Name is required")
    try:
        renamed = await rename_document_type(tag_id, name)
    except psycopg.errors.UniqueViolation:
        raise HTTPException(
            status.HTTP_409_CONFLICT, f"A document type named '{name}' already exists"
        )
    if renamed is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document type not found")
    return _tag_out(renamed)


@router.post("/document-types/{tag_id}/merge", response_model=TagOut)
async def merge_document_type_tag(
    tag_id: uuid.UUID, body: MergeTagRequest, admin: UserRecord = Depends(require_admin)
) -> TagOut:
    """Folds document type `tag_id` (the stray) into `body.into_id` (the
    survivor) and returns the survivor -- see tags.repository's
    merge_document_type for the single-transaction re-point + delete.
    """
    if tag_id == body.into_id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Can't merge a document type into itself")
    try:
        return _tag_out(await merge_document_type(tag_id, body.into_id))
    except TagNotFound as missing:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Document type {missing.tag_id} not found")
