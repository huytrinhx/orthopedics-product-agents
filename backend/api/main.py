"""FastAPI entrypoint. Wires together the workflow registry, checkpointer/
store, and route modules. Deployed as a single Railway service, which also
serves the built frontend as static files (see root Dockerfile).
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from agents import workflows  # noqa: F401  (registers all workflows)
from api.routes import chat, documents, feedback

app = FastAPI(title="Orthopedics Product Agents")
app.include_router(chat.router, prefix="/chat", tags=["chat"])
app.include_router(documents.router, prefix="/documents", tags=["documents"])
app.include_router(feedback.router, prefix="/feedback", tags=["feedback"])


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


# Serves the built frontend (frontend/out, produced by `next build` with
# output: "export" — see frontend/next.config.js) so one process hosts both
# the API and the UI, no CORS. Registered last so it never shadows the
# routes above: Starlette matches routes in registration order, and this
# mount's prefix is "/". Only present once the Docker image copies it in
# (see root Dockerfile) — absent in local dev, where the frontend runs on
# its own dev server instead.
FRONTEND_DIST = Path(__file__).resolve().parent.parent.parent / "frontend" / "out"
if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
