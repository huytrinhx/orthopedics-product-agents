"""FastAPI entrypoint. Wires together the workflow registry, checkpointer/
store, and route modules. Deployed as a single Railway service, which also
serves the built frontend as static files (see root Dockerfile).
"""
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from agents import workflows  # noqa: F401  (registers all workflows)
from api.routes import admin, auth, chat, documents, feedback, tags, users
from memory.checkpointer import get_checkpointer
from observability.langfuse_setup import configure_langfuse


class CleanUrlStaticFiles(StaticFiles):
    """`next build`'s static export (output: "export") writes each page as a
    flat file -- frontend/out/users.html, not out/users/index.html -- so a
    plain StaticFiles(html=True) mount can serve "/" (index.html) and
    "/users.html" but has no logic to resolve the extensionless "/users" a
    browser actually requests. In-app <Link> navigation never hits this (it's
    a client-side transition, no server round trip), which is why the gap
    went unnoticed until a hard/direct navigation to a route other than "/"
    404'd. This tries appending ".html" to an extensionless miss before
    giving up, mirroring the "clean URL" rewrite most static hosts do.
    """

    def lookup_path(self, path: str) -> tuple[str, os.stat_result | None]:
        full_path, stat_result = super().lookup_path(path)
        if stat_result is None and path and "." not in os.path.basename(path):
            return super().lookup_path(f"{path}.html")
        return full_path, stat_result


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initializes (and logs a warning if unconfigured) up front, rather than
    # silently on the first chat request -- see langfuse_setup.py.
    configure_langfuse()
    # One checkpointer for the process's whole lifetime, matching
    # backend/retrieval/graph_client.py's Neo4j-driver singleton pattern --
    # AsyncPostgresSaver's connection pool is scoped to this `async with`
    # block (see backend/memory/checkpointer.py), so it can't be built fresh
    # per request without paying for a new pool every chat turn.
    async with get_checkpointer(os.environ["DATABASE_URL"]) as checkpointer:
        app.state.checkpointer = checkpointer
        yield


app = FastAPI(title="OrthoMate", lifespan=lifespan)

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

app.include_router(admin.router, prefix="/admin", tags=["admin"])
app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(chat.router, prefix="/chat", tags=["chat"])
app.include_router(documents.router, prefix="/documents", tags=["documents"])
app.include_router(feedback.router, prefix="/feedback", tags=["feedback"])
app.include_router(tags.router, tags=["tags"])
app.include_router(users.router, prefix="/users", tags=["users"])


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
    app.mount("/", CleanUrlStaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
