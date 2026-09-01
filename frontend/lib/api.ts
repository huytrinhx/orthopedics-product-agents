import { getToken } from "./auth";
import type {
  AuthUser,
  ChatStreamEvent,
  ChatThread,
  ChatTranscript,
  DocumentRecord,
  Tag,
  TokenResponse,
} from "./types";

// Empty string = same-origin, correct in production (the backend serves
// this static export, see root Dockerfile / ADR 0004). Local dev runs the
// frontend on Next's own dev server, separate from the backend on :8000, so
// frontend/.env.local sets NEXT_PUBLIC_API_BASE=http://localhost:8000 there.
export const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "";

function authHeaders(): Record<string, string> {
  const token = getToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function unwrap<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `request failed: ${res.status}`);
  }
  return res.json();
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...authHeaders() },
    ...init,
  });
  return unwrap<T>(res);
}

export async function signup(email: string, password: string): Promise<TokenResponse> {
  return request("/auth/signup", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });
}

export async function login(email: string, password: string): Promise<TokenResponse> {
  return request("/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });
}

export async function getCurrentUser(): Promise<AuthUser> {
  return request("/auth/me");
}

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

export async function listSystems(): Promise<Tag[]> {
  return request("/systems");
}

export async function createSystem(name: string): Promise<Tag> {
  return request("/systems", { method: "POST", body: JSON.stringify({ name }) });
}

export async function listDocumentTypes(): Promise<Tag[]> {
  return request("/document-types");
}

export async function createDocumentType(name: string): Promise<Tag> {
  return request("/document-types", { method: "POST", body: JSON.stringify({ name }) });
}

// POST /chat/{workflow}/stream is SSE (event: <name>\ndata: <json>\n\n) but
// over a POST, so the native EventSource API (GET-only) can't consume it --
// this reads the fetch response body's stream and parses frames by hand.
export async function* streamChat(
  workflowName: string,
  message: string,
  threadId?: string
): AsyncGenerator<ChatStreamEvent> {
  const res = await fetch(`${API_BASE}/chat/${workflowName}/stream`, {
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
