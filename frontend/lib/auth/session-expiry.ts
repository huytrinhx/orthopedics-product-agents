// A 401 from any API call can happen anywhere in the tree (chat, documents,
// feedback), but only AuthProvider owns the `user`/`sessionExpired` React
// state that drives every page's logged-in/logged-out rendering. This is
// the plain-module <-> React bridge: api/client.ts's handleUnauthorized
// dispatches, AuthProvider's effect is the sole listener.
const EVENT = "orthomate:session-expired";

export function notifySessionExpired(): void {
  if (typeof window === "undefined") return;
  window.dispatchEvent(new Event(EVENT));
}

export function onSessionExpired(handler: () => void): () => void {
  if (typeof window === "undefined") return () => {};
  window.addEventListener(EVENT, handler);
  return () => window.removeEventListener(EVENT, handler);
}
