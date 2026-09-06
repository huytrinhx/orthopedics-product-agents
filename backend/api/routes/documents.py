"""Document upload/list/status endpoints, backing the Document Manager UI.
Admin-only (see auth.dependencies.require_admin — the is_admin gating
decided in agents.md), except GET /{document_id}/chunks: any authenticated
user can already surface a document's chunk text indirectly through chat
retrieval, so reading it directly to back the chat citation viewer isn't a
new exposure -- see backend/api/routes/chat.py. Upload writes the raw file
under INGEST_DATA_DIR and leaves the document "pending" -- indexing only
runs once an admin triggers it via POST /{document_id}/index (or re-tags
the document), which kicks off backend/documents/service.py's background
processing.
"""
import mimetypes
import os
import uuid
from pathlib import Path

import psycopg
from fastapi import APIRouter, BackgroundTasks, Depends, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from auth.dependencies import get_current_user, require_admin
from auth.repository import UserRecord
from documents.models import ChunkOut, DocumentOut, SetDocumentTagsRequest
from documents.repository import (
    create_document,
    delete_document,
    get_document,
    list_chunks,
    list_documents,
    replace_file,
    set_status,
    set_tags,
)
from documents.service import process_document
from tags.models import TagOut

router = APIRouter()


def _ingest_data_dir() -> Path:
    path = Path(os.environ.get("INGEST_DATA_DIR", "./data"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _document_out(doc) -> DocumentOut:
    return DocumentOut(
        id=doc.id,
        filename=doc.filename,
        status=doc.status,
        error=doc.error,
        uploaded_by=doc.uploaded_by,
        created_at=doc.created_at,
        updated_at=doc.updated_at,
        system=TagOut(id=doc.system_id, name=doc.system_name) if doc.system_id else None,
        document_type=(
            TagOut(id=doc.document_type_id, name=doc.document_type_name)
            if doc.document_type_id
            else None
        ),
    )


async def _save_upload(file: UploadFile) -> Path:
    """Prefix with a fresh id so two uploads of the same filename never
    collide on disk; the original filename is kept (and shown in the UI) in
    the documents row, not derived from this path. Shared by upload
    (new document) and reupload_document_file (existing document's file).
    """
    storage_path = _ingest_data_dir() / f"{uuid.uuid4()}-{file.filename}"
    contents = await file.read()
    storage_path.write_bytes(contents)
    return storage_path


@router.post("/upload", response_model=DocumentOut)
async def upload_document(
    file: UploadFile,
    system_id: uuid.UUID | None = Form(None),
    document_type_id: uuid.UUID | None = Form(None),
    admin: UserRecord = Depends(require_admin),
) -> DocumentOut:
    if not file.filename:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Uploaded file has no filename")

    storage_path = await _save_upload(file)

    try:
        doc = await create_document(
            filename=file.filename,
            storage_path=str(storage_path),
            uploaded_by=admin.id,
            system_id=system_id,
            document_type_id=document_type_id,
        )
    except psycopg.errors.ForeignKeyViolation:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Unknown system or document type")
    # Stays "pending" (the row's default) until an admin explicitly indexes
    # it -- see index_document below.
    return _document_out(doc)


@router.get("/", response_model=list[DocumentOut])
async def list_all_documents(admin: UserRecord = Depends(require_admin)) -> list[DocumentOut]:
    return [_document_out(doc) for doc in await list_documents()]


@router.get("/{document_id}", response_model=DocumentOut)
async def get_one_document(
    document_id: uuid.UUID, admin: UserRecord = Depends(require_admin)
) -> DocumentOut:
    doc = await get_document(document_id)
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    return _document_out(doc)


@router.get("/{document_id}/chunks", response_model=list[ChunkOut])
async def get_document_chunks(
    document_id: uuid.UUID, user: UserRecord = Depends(get_current_user)
) -> list[ChunkOut]:
    """Backs the chat citation viewer's right-hand pane -- any authenticated
    user, not just admins (see module docstring)."""
    if await get_document(document_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    return [
        ChunkOut(
            chunk_index=c.chunk_index,
            content=c.content,
            section_title=c.section_title,
            page_number=c.page_number,
        )
        for c in await list_chunks(document_id)
    ]


@router.get("/{document_id}/file")
async def get_document_file(
    document_id: uuid.UUID, user: UserRecord = Depends(get_current_user)
) -> FileResponse:
    """Serves the original uploaded file's raw bytes -- backs the chat
    citation viewer's PDF render (ticket 26), same any-authenticated-user
    access level as /chunks above (see module docstring): a rep can already
    read this document's extracted text via chat/citations, so reading the
    original file it came from isn't a new exposure.
    """
    doc = await get_document(document_id)
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    path = Path(doc.storage_path)
    if not path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document file not found")
    media_type = mimetypes.guess_type(doc.filename)[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type, filename=doc.filename)


@router.post("/{document_id}/file", response_model=DocumentOut)
async def reupload_document_file(
    document_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    file: UploadFile,
    admin: UserRecord = Depends(require_admin),
) -> DocumentOut:
    """Re-upload button: swaps an existing document's underlying file (e.g.
    a corrected PDF, or recovering one lost to a pre-volume Railway deploy --
    see README's Railway deploy section) without losing its id, tags, or
    upload history. Same tags/re-tag pipeline seam as set_document_tags/
    index_document below: queue + process_document re-extracts and
    re-indexes against the new file, upserting this document_id's chunks/
    graph entities in place rather than duplicating them.
    """
    doc = await get_document(document_id)
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    if not file.filename:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Uploaded file has no filename")

    storage_path = await _save_upload(file)
    old_storage_path = doc.storage_path
    doc = await replace_file(document_id, file.filename, str(storage_path))
    assert doc is not None
    # Best-effort, same as delete_one_document below: the DB row (now
    # pointing at the new file) is the source of truth, so a missing/
    # already-gone old file shouldn't turn a reupload into a 500.
    Path(old_storage_path).unlink(missing_ok=True)

    await set_status(document_id, "queued")
    background_tasks.add_task(process_document, doc.id, doc.storage_path)
    doc = await get_document(document_id)
    assert doc is not None
    return _document_out(doc)


@router.post("/{document_id}/index", response_model=DocumentOut)
async def index_document(
    document_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    admin: UserRecord = Depends(require_admin),
) -> DocumentOut:
    """Manual Index/Reindex trigger from the Document Manager UI -- runs the
    same (currently stubbed, see documents/service.py) pipeline seam as a
    re-tag, on demand rather than only as a side effect of tagging.
    """
    doc = await get_document(document_id)
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    await set_status(document_id, "queued")
    background_tasks.add_task(process_document, doc.id, doc.storage_path)
    doc = await get_document(document_id)
    assert doc is not None
    return _document_out(doc)


@router.patch("/{document_id}/tags", response_model=DocumentOut)
async def set_document_tags(
    document_id: uuid.UUID,
    body: SetDocumentTagsRequest,
    background_tasks: BackgroundTasks,
    admin: UserRecord = Depends(require_admin),
) -> DocumentOut:
    if await get_document(document_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    try:
        doc = await set_tags(document_id, body.system_id, body.document_type_id)
    except psycopg.errors.ForeignKeyViolation:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Unknown system or document type")
    assert doc is not None

    # System/document-type feed the retrieval filters and (once tickets
    # 06/07 land) the chunk/entity metadata written alongside a document's
    # vectors and graph nodes -- a re-tag has to re-run the same pipeline
    # seam as upload so that metadata stays in sync.
    await set_status(document_id, "queued")
    background_tasks.add_task(process_document, doc.id, doc.storage_path)
    doc = await get_document(document_id)
    assert doc is not None
    return _document_out(doc)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_one_document(
    document_id: uuid.UUID, admin: UserRecord = Depends(require_admin)
) -> None:
    doc = await delete_document(document_id)
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    # Best-effort: the DB row is the source of truth for the UI, so a
    # missing/already-gone file on disk shouldn't turn a delete into a 500.
    Path(doc.storage_path).unlink(missing_ok=True)
    # Tickets 06/07 land real vector/graph indexing keyed by document_id --
    # once they do, this also needs to purge that document's chunks/entities.
