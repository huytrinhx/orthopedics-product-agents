import { API_BASE, authHeaders, request } from "../api/client";
import { parseSseStream } from "../chat/api";
import type { ChatFeedback, ChatStreamEvent } from "../chat/types";
import type { FlaggedFeedback, Rerun } from "./types";

// Ticket 15 (Eval tab): every flagged item, newest first, with the actual
// question/answer text -- admin-only (backend/api/routes/feedback.py's
// require_admin), a non-admin gets a 403 the caller surfaces as an error.
export async function listFlaggedFeedback(): Promise<FlaggedFeedback[]> {
  return request("/feedback/flagged");
}

export async function setFeedbackResolved(
  messageId: string,
  resolved: boolean
): Promise<ChatFeedback> {
  return request(`/feedback/${messageId}/resolved`, {
    method: "PATCH",
    body: JSON.stringify({ resolved }),
  });
}

export async function listReruns(messageId: string): Promise<Rerun[]> {
  return request(`/feedback/${messageId}/reruns`);
}

// 204 No Content on success -- mirrors lib/documents/api.ts's deleteDocument
// (raw fetch, not request()/unwrap(), since unwrap always calls res.json()
// and a 204 response has no body to parse).
export async function deleteFlaggedFeedback(messageId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/feedback/${messageId}`, {
    method: "DELETE",
    headers: authHeaders(),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `request failed: ${res.status}`);
  }
}

// Re-asks a flagged question, in its real conversational context, against
// an admin-chosen workflow -- backend/api/routes/chat.py's rerun_chat does
// the actual history replay (graph.aupdate_state) and streams the new turn
// through the exact same _stream_graph every other chat turn uses, so this
// yields the identical ChatStreamEvent shape streamChat/resumeChat do.
export async function* rerunChat(
  originalThreadId: string,
  originalMessageId: string,
  workflowName: string
): AsyncGenerator<ChatStreamEvent> {
  const res = await fetch(`${API_BASE}/chat/rerun`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify({
      original_thread_id: originalThreadId,
      original_message_id: originalMessageId,
      workflow_name: workflowName,
    }),
  });
  yield* parseSseStream(res);
}

// A rerun can pause on the exact same clarification interrupt a real chat
// turn can (ticket 09's mechanic, reused unchanged) -- unlike the regular
// chat UI's resumeChat, this must target the *rerun's own* workflow
// (POST /chat/{workflow_name}/resume), not whichever workflow is the admin
// default, since a rerun's workflow is an explicit per-thread choice that
// can differ from it.
export async function* resumeRerunChat(
  threadId: string,
  workflowName: string,
  humanInput: string
): AsyncGenerator<ChatStreamEvent> {
  const res = await fetch(`${API_BASE}/chat/${workflowName}/resume`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify({ thread_id: threadId, human_input: humanInput }),
  });
  yield* parseSseStream(res);
}
