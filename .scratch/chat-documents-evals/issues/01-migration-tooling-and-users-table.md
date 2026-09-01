# 01: Add Postgres migration tooling and a users table

**What to build:** A repeatable way to evolve the Postgres schema over time, plus the first real table (`users`) that auth and every later feature will build on. Today there is no migration tooling anywhere in the backend.

**Blocked by:** None (can start immediately)

**Status:** done

- [x] A migration tool (Alembic) is wired up against `DATABASE_URL`, runnable via a documented command from `backend/`
- [x] `docker compose up -d` + running migrations produces a `users` table (id, email, hashed_password nullable for OAuth-only accounts, is_admin, created_at)
- [x] README/local-dev docs mention the migration command as part of first-time setup
- [x] CI runs migrations against a throwaway Postgres before backend tests

Also added (not a listed criterion, but the natural completion of this ticket): a `releaseCommand` in `railway.toml` so production deploys apply migrations too.
