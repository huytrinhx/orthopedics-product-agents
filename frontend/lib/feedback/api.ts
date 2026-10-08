import { API_BASE, authHeaders, handleUnauthorized, request } from "../api/client";
import type { ChatFeedback } from "../chat/types";
import type { FeedbackListItem } from "./types";

// Ticket 15 (Eval tab): every feedback row, flagged or not, newest first,
// with the actual question/answer text -- admin-only (backend/api/routes/
// feedback.py's require_admin), a non-admin gets a 403 the caller surfaces
// as an error.
export async function listFeedback(): Promise<FeedbackListItem[]> {
  return request("/feedback/");
}

// Only valid on a flagged row -- the backend 404s an unflagged one.
export async function setFeedbackResolved(
  messageId: string,
  resolved: boolean
): Promise<ChatFeedback> {
  return request(`/feedback/${messageId}/resolved`, {
    method: "PATCH",
    body: JSON.stringify({ resolved }),
  });
}

// 204 No Content on success -- mirrors lib/documents/api.ts's deleteDocument
// (raw fetch, not request()/unwrap(), since unwrap always calls res.json()
// and a 204 response has no body to parse).
export async function deleteFeedback(messageId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/feedback/${messageId}`, {
    method: "DELETE",
    headers: authHeaders(),
  });
  if (!response.ok) {
    handleUnauthorized(response);
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail ?? `request failed: ${response.status}`);
  }
}
