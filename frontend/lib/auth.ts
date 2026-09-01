// Client-side session storage. Static export (see next.config.js) means
// there's no server to hold a session, so the JWT lives in the browser --
// same approach as fhir-bridge's auth.ts.
const TOKEN_KEY = "auth_token";

export function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem(TOKEN_KEY);
}

export function setToken(token: string): void {
  window.localStorage.setItem(TOKEN_KEY, token);
}

export function clearToken(): void {
  window.localStorage.removeItem(TOKEN_KEY);
}
