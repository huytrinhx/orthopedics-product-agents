# 0005 — Local-disk document storage, not Blob Storage

## Status

Accepted

## Context

`docker-compose.yml` ran an Azurite (Azure Blob Storage emulator) service
and `.env.example` carried `AZURE_STORAGE_CONNECTION_STRING`, as a
placeholder for future raw-document storage. A repo-wide search found no
actual usage anywhere in the codebase — `backend/api/routes/documents.py`
and `backend/ingestion/pipeline.py` were (and still are) stubs that never
referenced blob storage. It was speculative scaffolding, not a load-bearing
dependency.

## Decision

Drop Azurite/Blob Storage entirely. Uploaded/ingested documents live under
`INGEST_DATA_DIR` (default `./data`) on local disk, mounted as a Railway
volume in production.

## Consequences

- One fewer local Docker service and one fewer env var.
- Document storage now depends on a correctly-mounted Railway volume in
  production — without it, uploaded documents are lost on every redeploy
  (container filesystems are otherwise ephemeral). See README's "Deploying
  to Railway" step 5.
- If object storage is needed later (e.g. multi-instance horizontal
  scaling, where local disk per-instance stops being viable), that's a new
  decision to make explicitly, not a silent reintroduction of Blob
  Storage/Azurite.
