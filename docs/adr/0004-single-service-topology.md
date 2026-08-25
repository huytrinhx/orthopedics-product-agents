# 0004 — Single Railway service, frontend as a static export

## Status

Accepted

## Context

The scaffold's `deploy.yml` built and pushed separate backend and frontend
container images to AKS — two deployables. Moving to Railway (ADR 0001)
didn't require collapsing that into one service; two smaller Railway
services was also an option, and would have kept the Next.js frontend on
its own Node server with room for server-only features (route handlers,
server actions) later. The frontend today has no such features — every
page is a static client component with no API routes.

## Decision

Ship one Railway service. The frontend builds as a static export
(`output: "export"` in `frontend/next.config.js`) and the backend serves
it as static files (`backend/api/main.py` mounts `frontend/out`) alongside
the API, from one FastAPI process. No second host, no CORS configuration
in production.

## Consequences

- Simpler production topology: one Dockerfile, one Railway service, one
  healthcheck.
- Rules out Next.js server-only features going forward (route handlers,
  server actions, ISR, middleware that needs a Node runtime) unless this
  decision is revisited — see the note in `agents.md`.
- The Docker image installs the backend editable (`pip install -e
  ./backend`) specifically so `main.py`'s `__file__`-relative lookup of
  `frontend/out` keeps resolving correctly; a normal wheel install would
  copy the package into `site-packages` and break that relative path.
