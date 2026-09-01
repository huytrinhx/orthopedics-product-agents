# 04: Document upload + list with status (admin-only)

**What to build:** An admin can upload a source document, see it appear in a list, and see its processing status change over time. This replaces the current `NotImplementedError` stubs in the documents route.

**Blocked by:** 01 (Add Postgres migration tooling and a users table), 02 (Email/password auth with admin flag)

**Status:** done

- [x] Only `is_admin` users can reach the Document Manager page or call its API routes (others get a 403 / are redirected)
- [x] Uploading a file writes it under `INGEST_DATA_DIR` and creates a `documents` row (filename, status, uploaded_by, created_at)
- [x] Upload kicks off a background task that updates status through queued → processing → done (or failed, with a visible error)
- [x] The document list shows filename, status, and upload time, and reflects status changes without a manual page reload (poll or refetch)
- [x] This ticket does not need real chunking/embedding yet — the background task can be a no-op that flips status to "done"; that's tickets 06/07

Verified: `backend/tests/test_documents.py` (real Postgres + real temp-dir disk writes, admin-vs-non-admin 403 check) + a live end-to-end pass (uvicorn against a throwaway DB: upload via curl → file confirmed on disk → status reached "done" → list reflects it; `/documents` page confirmed rendering via `next dev` against the live backend). No browser tool available, so the interactive upload-form click-through wasn't visually verified.
