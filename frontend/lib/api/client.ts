import { clearToken, getToken } from "../auth/token";
import { notifySessionExpired } from "../auth/session-expiry";

// Empty string = same-origin, correct in production (the backend serves
// this static export, see root Dockerfile / ADR 0004). Local dev runs the
// frontend on Next's own dev server, separate from the backend on :8000, so
// frontend/.env.local sets NEXT_PUBLIC_API_BASE=http://localhost:8000 there.
export const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "";

export function authHeaders(): Record<string, string> {
  const token = getToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

// A 401 means the stored token expired or was revoked -- clear it and tell
// AuthProvider so every page's `user`-gated render falls back to its
// logged-out state instead of quietly failing each subsequent request.
// Called from every fetch call site in lib/*/api.ts that can see a 401, not
// just unwrap() below (parseSseStream and a few raw-fetch call sites that
// can't use unwrap because they don't always have a JSON body).
export function handleUnauthorized(res: Response): void {
  if (res.status !== 401) return;
  clearToken();
  notifySessionExpired();
}

export async function unwrap<T>(res: Response): Promise<T> {
  if (!res.ok) {
    handleUnauthorized(res);
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
