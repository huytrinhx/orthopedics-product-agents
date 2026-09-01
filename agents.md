# agents.md — context for whoever (human or AI) works on this repo next

This file exists to carry forward decisions and working conventions so they
don't have to be re-derived or re-litigated. If you're an AI assistant
picking this repo up cold, read this before making structural changes.

## Standing technical decisions (don't silently reverse these)

- **Deployment is a single Railway service**, not AKS/Kubernetes. The root
  `Dockerfile` builds the Next.js static export and serves it plus the API
  from one FastAPI process (`backend/api/main.py` mounts `frontend/out`).
  This repo previously had Azure-native infra (Bicep: AKS, AI Search,
  Postgres Flexible Server, ACR) which was deliberately retired in favor of
  Railway — see `docs/adr/0001-railway-deployment.md`. Don't reintroduce
  Kubernetes/Bicep without discussing the tradeoff first.
- **Retrieval is Postgres/pgvector, not Azure AI Search.** The vector leg
  (`backend/retrieval/vector_store.py`) is pgvector cosine similarity; the
  keyword leg is Postgres full-text search (`tsvector`) — there is no
  AI-Search-style synonym-map index to keep in sync. Query-time synonym
  expansion instead queries the Neo4j/AuraDB graph directly
  (`backend/agents/tools/synonym_resolve.py`). See
  `docs/adr/0002-postgres-pgvector-retrieval.md`.
- **LLM/embeddings provider is plain OpenAI**
  (`backend/config/llm_clients.py`, `OPENAI_API_KEY`), not Azure OpenAI —
  the enterprise service-principal auth pattern (`AZURE_TENANT_ID`/
  `CLIENT_ID`/`CLIENT_SECRET`) was removed along with Azure AI Search. See
  `docs/adr/0003-openai-direct-llm-provider.md`.
  - **Kyma (kymaapi.com) is a live option for the chat model on deployment,
    not yet wired up.** There's deliberately no provider-switch abstraction
    in the code for this — it would be speculative complexity for a model
    not yet committed to. If this gets picked up, it belongs in the same PR
    that stands up the Kyma account, alongside `get_chat_model()` in
    `backend/config/llm_clients.py`.
- **Uploaded/ingested documents live on local disk (`INGEST_DATA_DIR`), not
  object storage.** No Azure Blob Storage/Azurite — a Railway volume covers
  persistence in production. See
  `docs/adr/0005-local-disk-document-storage.md`.
- **Neo4j is AuraDB in production, Docker locally** — this is independent
  of the Railway/AKS decision above; AuraDB is Neo4j's own managed control
  plane either way, not something provisioned through app infra.
- **Judge rubric is shared across three call sites.** `backend/agents/judge.py`
  scores faithfulness/relevance/style/citation and is used by the inline
  self-eval retry loop, the offline eval harness (`backend/evals/harness.py`),
  and the human feedback UI — on the same schema, so human and automated
  scores stay directly comparable. Don't fork the rubric per call site.
- **Workflow registry pattern.** Each agent architecture is its own
  LangGraph graph registered in `backend/agents/registry.py` under a shared
  state schema (`agents/state.py`) — the API and eval harness select by
  name. New architectures plug in via `agents/workflows/__init__.py`; they
  shouldn't require changes to callers.
- **Domain package pattern.** A domain's logic lives in its own package —
  `models.py` + `repository.py` (+ `service.py` where there's real
  processing, see `backend/documents/`) — with exactly one seam: the route
  file that imports from it (`backend/api/routes/chat.py` importing
  `chat_threads.models`/`chat_threads.repository`, `documents.py` importing
  `documents.*` and `tags.models`, etc.). No route file inlines model or
  persistence logic, and packages don't cross-import except where one
  domain genuinely depends on another (documents on tags, for tagging).
  This is why ticket 10 (chat history sidebar) needed zero `main.py`
  edits — `chat_threads/` is a self-contained package a ticket can own
  end-to-end. New domains follow this shape by default; don't grow logic
  directly inside `backend/api/routes/*.py`.
  - The frontend mirrors this 1:1 under `frontend/lib/`: one folder per
    domain (`auth/`, `documents/`, `chat/`), each holding `api.ts` +
    `types.ts` (+ `token.ts` for auth's session storage). `tags` nests
    under `documents/tags/` there — frontend only ever reads tags in
    service of the documents page, unlike the backend's dedicated
    `tags.py` admin route — so the two sides deliberately diverge in
    shape while sharing the same packaging convention. The one exception
    is `frontend/lib/api/client.ts`: shared HTTP plumbing (`API_BASE`,
    `authHeaders`, `request`, `unwrap`) that every domain imports — not a
    domain itself, low-churn, so it isn't a collision risk the way a flat
    `lib/api.ts` with every domain's functions appended to it was.
- **Golden datasets are per-product-system, not generic.** `mis.jsonl` /
  `reflex.jsonl` correspond to real implant systems (MIS, REFLEX); the
  intent-detection dataset routes a query to the right system before
  retrieval. `backend/evals/golden_datasets/build_dataset.py` regenerates
  these from `feedback-notes.csv` (human-reviewed Q&A) — rerun it after new
  rows land there, don't hand-edit the JSONL.
- **Auth is JWT-based, stateless, client-held.** `backend/auth/` issues a
  signed JWT on signup/login (`JWT_SECRET`); the frontend stores it in
  `localStorage` (`frontend/lib/auth.ts`) and attaches it as a Bearer token —
  there's no server-side session store to invalidate on logout. `is_admin` is
  decided once, at signup, from the `ADMIN_EMAILS` allowlist — it's not
  editable via any UI yet. Admin-gated pages (Documents, Evals) check this
  flag; regular chat access doesn't require it.
- **Schema changes go through Alembic (`backend/migrations/`), not hand-run
  SQL.** `alembic upgrade head` reads `DATABASE_URL` (same env var as
  everything else) and applies pending migrations; it's a required step in
  local setup (README), CI (runs against a throwaway Postgres service before
  `pytest`), and production (`railway.toml`'s `releaseCommand`, once per
  deploy before the new container takes traffic). Add new tables via
  `alembic revision -m "..."`, not a one-off script.

## Deployment direction (as of this writing)

Single Railway service, no GitHub Actions deploy step — `railway.toml`
points Railway at the root `Dockerfile`, and it redeploys on every push to
`main`. Postgres is Supabase (pgvector-enabled); Neo4j is AuraDB. Full
step-by-step is in `README.md`'s "Deploying to Railway" section. Highlights:

- The frontend is a Next.js **static export** (`output: "export"` in
  `frontend/next.config.js`) served by the backend — this rules out
  server-only Next.js features (route handlers, server actions, ISR). If a
  future feature genuinely needs one of those, that's a call to revisit the
  single-service topology (`docs/adr/0004-single-service-topology.md`), not
  to quietly bolt on a second Node process.
- `backend/api/main.py` mounts `frontend/out` as static files, registered
  **last** so it never shadows `/chat`, `/documents`, `/feedback` — Starlette
  matches routes in registration order.
- The Docker image installs the backend with `pip install -e ./backend`
  (editable), not a normal wheel install — `backend/api/main.py` locates
  `frontend/out` via a `__file__`-relative path, which only resolves
  correctly if the source tree stays where it was copied (`/app/backend`),
  not copied again into `site-packages`.

## Where to look for more

- `product.md` — what's actually known about the product surface, and
  what's still open.
- `build-log.md` — chronological record of how this repo got here.
- `CONTEXT.md` / `docs/adr/` — domain vocabulary and architectural decision
  records; see `docs/agents/domain.md` for how to consume them.
- `docs/agents/issue-tracker.md` — how issues are tracked for this repo.
- `README.md` — how to actually run the thing.
