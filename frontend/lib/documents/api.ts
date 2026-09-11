import { API_BASE, authHeaders, handleUnauthorized, request, unwrap } from "../api/client";
import type { DocumentChunk, DocumentRecord, SystemHealth } from "./types";

export async function uploadDocument(
  file: File,
  tags?: { systemId?: string; documentTypeId?: string }
): Promise<DocumentRecord> {
  const form = new FormData();
  form.append("file", file);
  if (tags?.systemId) form.append("system_id", tags.systemId);
  if (tags?.documentTypeId) form.append("document_type_id", tags.documentTypeId);
  const res = await fetch(`${API_BASE}/documents/upload`, {
    method: "POST",
    headers: authHeaders(), // no Content-Type: fetch sets the multipart boundary itself
    body: form,
  });
  return unwrap<DocumentRecord>(res);
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

export async function setDocumentTags(
  documentId: string,
  tags: { systemId: string | null; documentTypeId: string | null }
): Promise<DocumentRecord> {
  return request(`/documents/${documentId}/tags`, {
    method: "PATCH",
    body: JSON.stringify({ system_id: tags.systemId, document_type_id: tags.documentTypeId }),
  });
}

export async function indexDocument(documentId: string): Promise<DocumentRecord> {
  return request(`/documents/${documentId}/index`, { method: "POST" });
}

// Re-upload: swaps this document's underlying file (same id/tags/history)
// and queues it for reindexing, same multipart shape as uploadDocument.
export async function reuploadDocumentFile(
  documentId: string,
  file: File
): Promise<DocumentRecord> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/documents/${documentId}/file`, {
    method: "POST",
    headers: authHeaders(), // no Content-Type: fetch sets the multipart boundary itself
    body: form,
  });
  return unwrap<DocumentRecord>(res);
}

export async function getDocumentChunks(documentId: string): Promise<DocumentChunk[]> {
  return request(`/documents/${documentId}/chunks`);
}

// GET /documents/{id}/file needs an Authorization header, which a bare
// <iframe src>/react-pdf `file={url}` can't attach -- fetched as a Blob and
// handed to react-pdf's `file` prop instead (ticket 26).
export async function getDocumentFile(documentId: string): Promise<Blob> {
  const res = await fetch(`${API_BASE}/documents/${documentId}/file`, {
    headers: authHeaders(),
  });
  if (!res.ok) {
    handleUnauthorized(res);
    throw new Error(`request failed: ${res.status}`);
  }
  return res.blob();
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
  const res = await fetch(`${API_BASE}/documents/${documentId}`, {
    method: "DELETE",
    headers: authHeaders(),
  });
  if (!res.ok) {
    handleUnauthorized(res);
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `request failed: ${res.status}`);
  }
}
