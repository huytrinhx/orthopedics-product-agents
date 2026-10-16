import { API_BASE, authHeaders, handleUnauthorized, request } from "../../api/client";
import type { Tag } from "./types";

export async function listSystems(): Promise<Tag[]> {
  return request("/systems");
}

export async function createSystem(name: string): Promise<Tag> {
  return request("/systems", { method: "POST", body: JSON.stringify({ name }) });
}

// 204 No Content on success -- goes through raw fetch rather than
// request()/unwrap(), which always calls res.json() and would choke on the
// empty body (same reason lib/documents/api.ts's deleteDocument does this).
export async function deleteSystem(id: string): Promise<void> {
  const response = await fetch(`${API_BASE}/systems/${id}`, { method: "DELETE", headers: authHeaders() });
  if (!response.ok) {
    handleUnauthorized(response);
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail ?? `request failed: ${response.status}`);
  }
}

export async function listDocumentTypes(): Promise<Tag[]> {
  return request("/document-types");
}

// Document Types are a closed, admin-curated list (issue #16): create,
// rename, merge, and delete are only ever called from the Settings page
// (app/admin/document-types-section.tsx). The Document Manager only reads
// the list via listDocumentTypes above.
export async function createDocumentType(name: string): Promise<Tag> {
  return request("/document-types", { method: "POST", body: JSON.stringify({ name }) });
}

export async function renameDocumentType(id: string, name: string): Promise<Tag> {
  return request(`/document-types/${id}`, { method: "PATCH", body: JSON.stringify({ name }) });
}

// Moves every document and chunk on `strayId` to `intoId` and deletes the
// stray, server-side in one transaction. Resolves to the surviving type.
export async function mergeDocumentType(strayId: string, intoId: string): Promise<Tag> {
  return request(`/document-types/${strayId}/merge`, {
    method: "POST",
    body: JSON.stringify({ into_id: intoId }),
  });
}

export async function deleteDocumentType(id: string): Promise<void> {
  const response = await fetch(`${API_BASE}/document-types/${id}`, {
    method: "DELETE",
    headers: authHeaders(),
  });
  if (!response.ok) {
    handleUnauthorized(response);
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail ?? `request failed: ${response.status}`);
  }
}
