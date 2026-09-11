"""System and Document-Type tag management, backing the Document Manager
UI's tag pickers (ticket 05). Admin-only, same as the rest of document
management (see auth.dependencies.require_admin). Deliberately no fixed
enum -- an admin grows these two lists as new product systems and document
types show up.
"""
import uuid

import psycopg
from fastapi import APIRouter, Depends, HTTPException, status

from auth.dependencies import require_admin
from auth.repository import UserRecord
from tags.models import CreateTagRequest, TagOut
from tags.repository import (
    create_document_type,
    create_system,
    delete_document_type,
    delete_system,
    list_document_types,
    list_systems,
)

router = APIRouter()


def _tag_out(tag) -> TagOut:
    return TagOut(id=tag.id, name=tag.name)


@router.post("/systems", response_model=TagOut)
async def create_system_tag(
    body: CreateTagRequest, admin: UserRecord = Depends(require_admin)
) -> TagOut:
    name = body.name.strip()
    if not name:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Name is required")
    try:
        return _tag_out(await create_system(name))
    except psycopg.errors.UniqueViolation:
        raise HTTPException(status.HTTP_409_CONFLICT, f"A system named '{name}' already exists")


@router.get("/systems", response_model=list[TagOut])
async def list_system_tags(admin: UserRecord = Depends(require_admin)) -> list[TagOut]:
    return [_tag_out(tag) for tag in await list_systems()]


@router.delete("/systems/{tag_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_system_tag(tag_id: uuid.UUID, admin: UserRecord = Depends(require_admin)) -> None:
    try:
        deleted = await delete_system(tag_id)
    except psycopg.errors.ForeignKeyViolation:
        raise HTTPException(
            status.HTTP_409_CONFLICT, "This system is still assigned to one or more documents"
        )
    if not deleted:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "System not found")


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
