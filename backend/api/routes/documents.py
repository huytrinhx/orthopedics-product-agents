"""Document upload/list/status endpoints, backing the Document Manager UI.
Admin-only (see auth.dependencies.require_admin — the is_admin gating
decided in agents.md), except GET /{document_id}/chunks: any authenticated
user can already surface a document's chunk text indirectly through chat
retrieval, so reading it directly to back the chat citation viewer isn't a
new exposure -- see backend/api/routes/chat.py. Upload writes the raw file
under INGEST_DATA_DIR and leaves the document untagged and "pending" --
indexing only runs once an admin triggers it via POST /{document_id}/index
(or re-uploads the file), which kicks off backend/documents/service.py's
background processing -- and only once the document is complete (Product
Family, a Tray, a Document Type; issue #14's gate). A re-tag never re-indexes: it updates what indexing
already wrote in place (documents.service.retag_document).
"""
import mimetypes
import os
import uuid
from pathlib import Path

import neo4j.exceptions
import psycopg
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from auth.dependencies import get_current_user, require_admin
from auth.repository import UserRecord
from catalog.models import TrayOut
from document_skus.models import DocumentSkuOut
from document_skus.service import list_document_skus
from documents.models import (
    ChunkOut,
    ComponentHealth,
    DocumentOut,
    DocumentProductFamilyOut,
    SetDocumentTagsRequest,
    SystemHealthOut,
)
from documents.repository import (
    TraysOutsideFamily,
    create_document,
    delete_document,
    get_document,
    list_chunks,
    list_documents,
    replace_file,
    set_status,
)
from documents.service import missing_index_tags, process_document, retag_document
from retrieval.graph_client import get_graph_client
from retrieval.vector_store import get_vector_store
from tags.models import TagOut

router = APIRouter()


def _ingest_data_dir() -> Path:
    path = Path(os.environ.get("INGEST_DATA_DIR", "./data"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _document_out(document) -> DocumentOut:
    return DocumentOut(
        id=document.id,
        filename=document.filename,
        status=document.status,
        error=document.error,
        uploaded_by=document.uploaded_by,
        created_at=document.created_at,
        updated_at=document.updated_at,
        document_type=(
            TagOut(id=document.document_type_id, name=document.document_type_name)
            if document.document_type_id
            else None
        ),
        product_family=(
            DocumentProductFamilyOut(
                id=document.product_family_id,
                name=document.product_family_name,
                display_name=document.product_family_display_name,
            )
            if document.product_family_id
            else None
        ),
        trays=[
            TrayOut(id=tray.id, name=tray.name, display_name=tray.display_name)
            for tray in document.trays
        ],
        last_indexed_at=document.last_indexed_at,
        missing_tags=missing_index_tags(document),
    )


def _require_complete(document) -> None:
    """The indexing gate (issue #14), shared by every route that queues
    process_document. Refuses before anything is written or queued, naming
    the missing tags so the admin knows what to fill in the tag modal."""
    missing = missing_index_tags(document)
    if missing:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "Document can't be indexed until it's tagged; missing: "
            + ", ".join(missing)
            + ". Set them in the document's Tags.",
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
    admin: UserRecord = Depends(require_admin),
) -> DocumentOut:
    """Takes the file only -- no tags (issue #13). The document lands
    untagged and "pending" (the row's default); an admin tags it in the
    Document Manager's tag modal (PATCH /{document_id}/tags) and then
    indexes it explicitly (POST /{document_id}/index).
    """
    if not file.filename:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Uploaded file has no filename")

    storage_path = await _save_upload(file)
    document = await create_document(
        filename=file.filename,
        storage_path=str(storage_path),
        uploaded_by=admin.id,
    )
    return _document_out(document)


@router.get("/", response_model=list[DocumentOut])
async def list_all_documents(admin: UserRecord = Depends(require_admin)) -> list[DocumentOut]:
    return [_document_out(document) for document in await list_documents()]


async def _check_volume() -> ComponentHealth:
    try:
        count = sum(1 for path in _ingest_data_dir().iterdir() if path.is_file())
    except OSError as error:
        return ComponentHealth(ok=False, detail=str(error))
    return ComponentHealth(ok=True, detail=f"{count} file{'s' if count != 1 else ''}")


async def _check_graph_db() -> ComponentHealth:
    try:
        await get_graph_client().ping()
    except neo4j.exceptions.GqlError as error:
        # Common ancestor of both the driver's own errors (connection
        # refused, DNS failure) and Neo4jError (auth, query) -- see
        # neo4j.exceptions' hierarchy.
        return ComponentHealth(ok=False, detail=str(error))
    return ComponentHealth(ok=True, detail="connected")


async def _check_vector_db() -> ComponentHealth:
    try:
        async with get_vector_store():
            pass
    except psycopg.Error as error:
        return ComponentHealth(ok=False, detail=str(error))
    return ComponentHealth(ok=True, detail="connected")


# Placed before "/{document_id}" -- registration order decides the match
# here (Starlette has no notion of a literal path beating a param one), so
# a GET /documents/health that landed after "/{document_id}" would 422
# trying to parse "health" as a UUID instead of ever reaching this route.
@router.get("/health", response_model=SystemHealthOut)
async def check_system_health(admin: UserRecord = Depends(require_admin)) -> SystemHealthOut:
    # Real round-trips, not "did the client construct" -- a poke each time
    # the Document Manager page loads (see frontend/app/documents/page.tsx),
    # not a cached/periodic check, so it reflects what's true right now.
    return SystemHealthOut(
        volume=await _check_volume(),
        graph_db=await _check_graph_db(),
        vector_db=await _check_vector_db(),
    )


@router.get("/{document_id}", response_model=DocumentOut)
async def get_one_document(
    document_id: uuid.UUID, admin: UserRecord = Depends(require_admin)
) -> DocumentOut:
    document = await get_document(document_id)
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    return _document_out(document)


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
            chunk_index=chunk.chunk_index,
            content=chunk.content,
            section_title=chunk.section_title,
            page_number=chunk.page_number,
            heading_path=chunk.heading_path,
        )
        for chunk in await list_chunks(document_id)
    ]


@router.get("/{document_id}/skus", response_model=list[DocumentSkuOut])
async def get_document_skus(
    document_id: uuid.UUID, admin: UserRecord = Depends(require_admin)
) -> list[DocumentSkuOut]:
    """The document's SKU index (issue #27): every SKU its chunks mention,
    with chunk, page and catalog description, plus flagged tokens that
    aren't in the catalog. Admin-only, for reviewing a document."""
    if await get_document(document_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    return await list_document_skus(document_id)


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
    document = await get_document(document_id)
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    path = Path(document.storage_path)
    if not path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document file not found")
    media_type = mimetypes.guess_type(document.filename)[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type, filename=document.filename)


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
    upload history. Same pipeline seam as index_document below: queue +
    process_document re-extracts and
    re-indexes against the new file, upserting this document_id's chunks/
    graph entities in place rather than duplicating them. Because it
    re-indexes, it's behind the same completeness gate as index_document --
    checked before the new file is saved, so a refused re-upload leaves the
    document (and its old file) exactly as it was.
    """
    document = await get_document(document_id)
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    _require_complete(document)
    if not file.filename:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Uploaded file has no filename")

    storage_path = await _save_upload(file)
    old_storage_path = document.storage_path
    document = await replace_file(document_id, file.filename, str(storage_path))
    assert document is not None
    # Best-effort, same as delete_one_document below: the DB row (now
    # pointing at the new file) is the source of truth, so a missing/
    # already-gone old file shouldn't turn a reupload into a 500.
    Path(old_storage_path).unlink(missing_ok=True)

    await set_status(document_id, "queued")
    background_tasks.add_task(process_document, document.id, document.storage_path)
    document = await get_document(document_id)
    assert document is not None
    return _document_out(document)


@router.post("/{document_id}/index", response_model=DocumentOut)
async def index_document(
    document_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    admin: UserRecord = Depends(require_admin),
) -> DocumentOut:
    """Manual Index/Reindex trigger from the Document Manager UI -- the only
    way (besides a re-upload) a document's text gets (re-)extracted and
    embedded; tagging never triggers it (see set_document_tags below).
    Refuses an incomplete document with a 409 naming its missing tags
    (_require_complete); an already-indexed one keeps its status and stays
    retrievable.
    """
    document = await get_document(document_id)
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    _require_complete(document)
    await set_status(document_id, "queued")
    background_tasks.add_task(process_document, document.id, document.storage_path)
    document = await get_document(document_id)
    assert document is not None
    return _document_out(document)


@router.patch("/{document_id}/tags", response_model=DocumentOut)
async def set_document_tags(
    document_id: uuid.UUID,
    body: SetDocumentTagsRequest,
    admin: UserRecord = Depends(require_admin),
) -> DocumentOut:
    """Backs the Document Manager's tag modal. Updates an already-indexed
    document's chunk filter columns and graph Document node in place
    (documents.service.retag_document) -- no re-index, so the document's
    status is left exactly as it was.
    """
    current = await get_document(document_id)
    if current is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    # Omitted fields keep their current value (see SetDocumentTagsRequest).
    sent = body.model_fields_set
    family_sent = "product_family_id" in sent or "tray_ids" in sent
    try:
        document = await retag_document(
            document_id,
            document_type_id=(
                body.document_type_id if "document_type_id" in sent else current.document_type_id
            ),
            product_family_id=(
                body.product_family_id if family_sent else current.product_family_id
            ),
            tray_ids=body.tray_ids if family_sent else [tray.id for tray in current.trays],
        )
    except psycopg.errors.ForeignKeyViolation:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "Unknown document type, product family, or tray"
        )
    except TraysOutsideFamily as error:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Every tray must belong to the document's product family; not in it: "
            + ", ".join(str(tray_id) for tray_id in error.tray_ids),
        )
    except neo4j.exceptions.GqlError as error:
        # Postgres (tags + chunk columns) already committed; only the graph
        # node is behind. Say so rather than pretend nothing was saved.
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            f"Tags saved, but the graph could not be updated ({error}); "
            "reindex the document to resync it.",
        )
    assert document is not None
    return _document_out(document)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_one_document(
    document_id: uuid.UUID, admin: UserRecord = Depends(require_admin)
) -> None:
    document = await delete_document(document_id)
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    # Best-effort: the DB row is the source of truth for the UI, so a
    # missing/already-gone file on disk shouldn't turn a delete into a 500.
    Path(document.storage_path).unlink(missing_ok=True)
    # Tickets 06/07 land real vector/graph indexing keyed by document_id --
    # once they do, this also needs to purge that document's chunks/entities.
