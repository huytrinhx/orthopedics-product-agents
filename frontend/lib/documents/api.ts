import { API_BASE, authHeaders, request, unwrap } from "../api/client";
import type { DocumentChunk, DocumentRecord } from "./types";

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

export async function getDocumentChunks(documentId: string): Promise<DocumentChunk[]> {
  return request(`/documents/${documentId}/chunks`);
}

export async function deleteDocument(documentId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/documents/${documentId}`, {
    method: "DELETE",
    headers: authHeaders(),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `request failed: ${res.status}`);
  }
}
