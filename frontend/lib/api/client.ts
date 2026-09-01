import { getToken } from "../auth/token";

// Empty string = same-origin, correct in production (the backend serves
// this static export, see root Dockerfile / ADR 0004). Local dev runs the
// frontend on Next's own dev server, separate from the backend on :8000, so
// frontend/.env.local sets NEXT_PUBLIC_API_BASE=http://localhost:8000 there.
export const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "";

export function authHeaders(): Record<string, string> {
  const token = getToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

export async function unwrap<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `request failed: ${res.status}`);
  }
  return res.json();
}

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...authHeaders() },
    ...init,
  });
  return unwrap<T>(res);
}
