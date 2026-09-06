# Single-service deploy: builds the frontend as a static export, then
# serves it and the API out of one FastAPI process (backend/api/main.py
# mounts frontend/out as static files). See README.md's "Deploying to
# Railway" section.

# node:22, not 20 -- react-pdf's pdfjs-dist (ticket 26) calls
# Promise.withResolvers(), which doesn't exist before Node 21. Node 20
# builds fine but crashes prerendering /chat with "TypeError:
# Promise.withResolvers is not a function".
FROM node:22-slim AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim AS backend
WORKDIR /app

COPY backend/ ./backend/
# Editable install: keeps source at /app/backend rather than copying into
# site-packages, so backend/api/main.py's __file__-relative lookup of
# ../../frontend/out still resolves correctly.
RUN pip install --no-cache-dir -e ./backend

COPY --from=frontend-build /app/frontend/out ./frontend/out

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

# Shell form so $PORT (Railway sets this at runtime) actually expands.
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}
