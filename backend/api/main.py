"""FastAPI entrypoint. Wires together the workflow registry, checkpointer/
store, and route modules. Deployed as a single Railway service, which also
serves the built frontend as static files (see root Dockerfile).
"""
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from agents import workflows  # noqa: F401  (registers all workflows)
from api.routes import auth, chat, documents, feedback, tags

app = FastAPI(title="OrthoMate")

# In production the frontend is served same-origin (mounted below), so this
# never applies there. Locally the frontend runs on Next's own dev server
# (:3000) while this API runs separately (:8000, see README's "Local
# Development") -- without this, every browser fetch from the app to the
# API is blocked by CORS before it reaches any route.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.environ.get("FRONTEND_PUBLIC_URL", "http://localhost:3000")],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(chat.router, prefix="/chat", tags=["chat"])
app.include_router(documents.router, prefix="/documents", tags=["documents"])
app.include_router(feedback.router, prefix="/feedback", tags=["feedback"])
app.include_router(tags.router, tags=["tags"])


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
