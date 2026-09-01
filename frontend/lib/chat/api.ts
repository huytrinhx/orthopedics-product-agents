import { API_BASE, authHeaders, request } from "../api/client";
import type { ChatStreamEvent, ChatThread, ChatTranscript } from "./types";

// POST /chat/stream is SSE (event: <name>\ndata: <json>\n\n) but over a
// POST, so the native EventSource API (GET-only) can't consume it -- this
// reads the fetch response body's stream and parses frames by hand. No
// workflow name here -- the backend runs whichever workflow the admin has
// configured as default (ticket 14); the chat UI never picks one itself.
export async function* streamChat(
  message: string,
  threadId?: string
): AsyncGenerator<ChatStreamEvent> {
  const res = await fetch(`${API_BASE}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify({ message, thread_id: threadId }),
  });
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

export async function listChatThreads(): Promise<ChatThread[]> {
  return request("/chat/threads");
}

export async function getChatThread(threadId: string): Promise<ChatTranscript> {
  return request(`/chat/threads/${threadId}`);
}
