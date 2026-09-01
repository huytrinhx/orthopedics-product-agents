# 03: Google OAuth sign-in

**What to build:** A second way into the same session mechanism ticket 02 built — "Sign in with Google" — so users aren't required to create a password.

**Blocked by:** 02 (Email/password auth with admin flag)

**Status:** done

- [x] A "Sign in with Google" button starts the OAuth redirect flow
- [x] On successful Google auth, a `users` row is created (or matched by email) and the same `ADMIN_EMAILS` allowlist check applies
- [x] The callback hands the frontend a session the same way password login does (same JWT/token shape), so the rest of the app doesn't need to know which login path was used
- [x] Signing in with Google using an email that already has a password account links to the same user rather than creating a duplicate
