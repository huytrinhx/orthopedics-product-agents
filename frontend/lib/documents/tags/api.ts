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
  const res = await fetch(`${API_BASE}/systems/${id}`, { method: "DELETE", headers: authHeaders() });
  if (!res.ok) {
    handleUnauthorized(res);
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `request failed: ${res.status}`);
  }
}

export async function listDocumentTypes(): Promise<Tag[]> {
  return request("/document-types");
}

export async function createDocumentType(name: string): Promise<Tag> {
  return request("/document-types", { method: "POST", body: JSON.stringify({ name }) });
}

export async function deleteDocumentType(id: string): Promise<void> {
  const res = await fetch(`${API_BASE}/document-types/${id}`, {
    method: "DELETE",
    headers: authHeaders(),
  });
  if (!res.ok) {
    handleUnauthorized(res);
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `request failed: ${res.status}`);
  }
}
