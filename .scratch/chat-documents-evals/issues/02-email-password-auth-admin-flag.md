# 02: Email/password auth with admin flag

**What to build:** A user can sign up and log in with email/password, gets a persisted session, and the app can tell an admin from a regular user. This is the foundation every later feature (chat history, admin-gated pages) depends on.

**Blocked by:** 01 (Add Postgres migration tooling and a users table)

**Status:** done

- [x] Signup creates a `users` row with a securely hashed password
- [x] At signup, `is_admin` is set to true if the account's email matches the `ADMIN_EMAILS` env var allowlist, false otherwise
- [x] Login issues a JWT; the frontend stores it and attaches it to authenticated requests
- [x] A "current user" endpoint returns the logged-in user (including `is_admin`) so the frontend can render an authenticated shell vs. a logged-out state
- [x] Logout clears the client-side session
- [x] Passwords are never logged or returned in any response

Verified: `backend/tests/test_auth.py` (7 tests, real Postgres) + a live end-to-end pass (uvicorn + `next dev` against a throwaway DB — signup/login/me over curl, login/signup pages rendering with no console/compile errors). No browser tool was available in this session, so interactive click-through (does submitting the form actually redirect) wasn't visually verified, only the underlying fetch/rendering.

Also fixed along the way (pre-existing gaps that blocked this ticket's own tests/CI, not introduced by it): missing `python-multipart` dependency (`documents.py`'s `UploadFile` route needed it), `agents/state.py`'s `EvalScores` using `typing.TypedDict` (pydantic v2 rejects this on Python &lt;3.12 once a model references it), and the frontend having no ESLint config/dependency at all despite CI running `npm run lint`.
