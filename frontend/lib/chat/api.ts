import { API_BASE, authHeaders, request } from "../api/client";
import type { ChatFeedback, ChatStreamEvent, ChatThread, ChatTranscript, SubmitFeedbackRequest } from "./types";

// SSE (event: <name>\ndata: <json>\n\n) but over a POST, so the native
// EventSource API (GET-only) can't consume it -- reads the fetch response
// body's stream and parses frames by hand. Shared by streamChat and
// resumeChat below, since both hit a POST .../stream-shaped endpoint that
// emits the same frame format (backend/api/routes/chat.py's _stream_graph
// drives both).
// Exported for lib/feedback/api.ts's rerun functions (ticket 15) -- same
// frame format, since backend/api/routes/chat.py's _stream_graph drives
// POST /chat/rerun and its resume too.
export async function* parseSseStream(res: Response): AsyncGenerator<ChatStreamEvent> {
  if (!res.ok || !res.body) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `request failed: ${res.status}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";
    for (const frame of frames) {
      let eventName = "";
      let data = "";
      for (const line of frame.split("\n")) {
        if (line.startsWith("event: ")) eventName = line.slice("event: ".length);
        else if (line.startsWith("data: ")) data = line.slice("data: ".length);
      }
      if (eventName && data) {
        yield { event: eventName, data: JSON.parse(data) } as ChatStreamEvent;
      }
    }
  }
}

// No workflow name here -- the backend runs whichever workflow the admin
// has configured as default (ticket 14); the chat UI never picks one
// itself.
export async function* streamChat(
  message: string,
  threadId?: string
): AsyncGenerator<ChatStreamEvent> {
  const res = await fetch(`${API_BASE}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify({ message, thread_id: threadId }),
  });
  yield* parseSseStream(res);
}

// Ticket 09: answers a pending detect_intent clarification (a clicked
// option's exact text, or free text) and resumes the same suspended graph
// run -- not a new turn, so no message param, just the answer.
export async function* resumeChat(
  threadId: string,
  humanInput: string
): AsyncGenerator<ChatStreamEvent> {
  const res = await fetch(`${API_BASE}/chat/resume`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify({ thread_id: threadId, human_input: humanInput }),
  });
  yield* parseSseStream(res);
}

export async function listChatThreads(): Promise<ChatThread[]> {
  return request("/chat/threads");
}

export async function getChatThread(threadId: string): Promise<ChatTranscript> {
  return request(`/chat/threads/${threadId}`);
}

// Ticket 11: submits (or, resubmitting on the same message_id, overwrites --
// backend/feedback/repository.py's upsert) one message's 4-axis
// score/flag/comment. Also used comment-only by ticket 12's free-text
// "Give feedback" entry point.
export async function submitFeedback(payload: SubmitFeedbackRequest): Promise<ChatFeedback> {
  return request("/feedback/", { method: "POST", body: JSON.stringify(payload) });
}
