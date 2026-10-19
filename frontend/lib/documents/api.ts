import { API_BASE, authHeaders, handleUnauthorized, request, unwrap } from "../api/client";
import type {
  DocumentChunk,
  DocumentRecord,
  DuplicateFilenameDetail,
  ExistingDocument,
  SystemHealth,
} from "./types";

// Thrown by uploadDocument/reuploadDocumentFile when another document
// already has the file's name (issue #31) -- carries that document so the
// Document Manager can name it and point at its row.
export class DuplicateFilenameError extends Error {
  readonly existingDocument: ExistingDocument;

  constructor(detail: DuplicateFilenameDetail) {
    super(detail.message);
    this.name = "DuplicateFilenameError";
    this.existingDocument = detail.existing_document;
  }
}

// A 409 is also what an incomplete document's Re-upload gets (a plain
// string detail, issue #14), so only one naming an existing document is a
// DuplicateFilenameError; everything else goes through unwrap as usual.
async function unwrapFileUpload(response: Response): Promise<DocumentRecord> {
  if (response.status === 409) {
    const body = await response.clone().json().catch(() => ({}));
    if (body.detail?.existing_document) throw new DuplicateFilenameError(body.detail);
  }
  return unwrap<DocumentRecord>(response);
}

// File only -- a new upload lands untagged and "pending", and gets its tags
// in the Document Manager's tag modal before it's indexed (issue #13).
// Throws DuplicateFilenameError if a document with this name exists.
export async function uploadDocument(file: File): Promise<DocumentRecord> {
  const form = new FormData();
  form.append("file", file);
  const response = await fetch(`${API_BASE}/documents/upload`, {
    method: "POST",
    headers: authHeaders(), // no Content-Type: fetch sets the multipart boundary itself
    body: form,
  });
  return unwrapFileUpload(response);
}

export async function listDocuments(): Promise<DocumentRecord[]> {
  return request("/documents/");
}

// Real round-trips (volume file count, a Neo4j query, a Postgres/pgvector
// connection), not cached -- see backend/api/routes/documents.py's
// check_system_health. Meant to be called fresh on every page load.
export async function checkSystemHealth(): Promise<SystemHealth> {
  return request("/documents/health");
}

// Only the tags passed are changed; the rest keep their current value (see
// backend/documents/models.py's SetDocumentTagsRequest). A family and its
// tray set always travel together, and the server rejects a tray outside
// the family. Saving never re-indexes: an indexed document's chunks and
// graph node are updated in place and its status is left alone.
export async function setDocumentTags(
  documentId: string,
  tags: {
    documentTypeId?: string | null;
    family?: { productFamilyId: string | null; trayIds: string[] };
  }
): Promise<DocumentRecord> {
  const body: Record<string, unknown> = {};
  if (tags.documentTypeId !== undefined) body.document_type_id = tags.documentTypeId;
  if (tags.family) {
    body.product_family_id = tags.family.productFamilyId;
    body.tray_ids = tags.family.trayIds;
  }
  return request(`/documents/${documentId}/tags`, {
    method: "PATCH",
    body: JSON.stringify(body),
  });
}

export async function indexDocument(documentId: string): Promise<DocumentRecord> {
  return request(`/documents/${documentId}/index`, { method: "POST" });
}

// Re-upload: swaps this document's underlying file (same id/tags/history)
// and queues it for reindexing, same multipart shape as uploadDocument.
// The file may keep this document's own name, but another document's name
// throws DuplicateFilenameError.
export async function reuploadDocumentFile(
  documentId: string,
  file: File
): Promise<DocumentRecord> {
  const form = new FormData();
  form.append("file", file);
  const response = await fetch(`${API_BASE}/documents/${documentId}/file`, {
    method: "POST",
    headers: authHeaders(), // no Content-Type: fetch sets the multipart boundary itself
    body: form,
  });
  return unwrapFileUpload(response);
}

export async function getDocumentChunks(documentId: string): Promise<DocumentChunk[]> {
  return request(`/documents/${documentId}/chunks`);
}

// GET /documents/{id}/file needs an Authorization header, which a bare
// <iframe src>/react-pdf `file={url}` can't attach -- fetched as a Blob and
// handed to react-pdf's `file` prop instead (ticket 26).
export async function getDocumentFile(documentId: string): Promise<Blob> {
  const response = await fetch(`${API_BASE}/documents/${documentId}/file`, {
    headers: authHeaders(),
  });
  if (!response.ok) {
    handleUnauthorized(response);
    throw new Error(`request failed: ${response.status}`);
  }
  return response.blob();
}

// Download button: reuses getDocumentFile's authenticated blob fetch (the
// same one the citation viewer uses), then drives a normal browser
// save-as via a throwaway <a download> -- a bare <a href> can't carry the
// Authorization header GET /file requires.
export async function downloadDocumentFile(documentId: string, filename: string): Promise<void> {
  const blob = await getDocumentFile(documentId);
  const url = URL.createObjectURL(blob);
  try {
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    link.click();
  } finally {
    URL.revokeObjectURL(url);
  }
}

export async function deleteDocument(documentId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/documents/${documentId}`, {
    method: "DELETE",
    headers: authHeaders(),
  });
  if (!response.ok) {
    handleUnauthorized(response);
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail ?? `request failed: ${response.status}`);
  }
}
