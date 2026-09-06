# Orthopedics Product Agents

Agentic retrieval over an orthopedics product/clinical knowledge base

## Architecture decisions

1. **Orchestration: LangGraph.** Suspend/resume clarification, loops,
   streaming, and conversation/user memory all map onto LangGraph primitives
   (`interrupt()`, cycles, `astream_events`, checkpointer, Store). Self-hosted
   as a FastAPI service on Railway, not the managed LangGraph Platform.
2. **Graph DB: Neo4j** (AuraDB in production). Keeps ops effort off running
   the graph DB ourselves; locally it's the plain Docker image. The query
   layer is `backend/retrieval/graph_client.py`.
3. **Retrieval: hybrid vector (Postgres/pgvector) + graph (Neo4j).** Neo4j
   holds the canonical entity/synonym graph; a synonym-resolution step
   (`backend/agents/tools/synonym_resolve.py`) queries it directly to expand
   a query before retrieval runs. The vector leg is pgvector cosine
   similarity; the keyword leg is Postgres full-text search (`tsvector`) —
   both live in `backend/retrieval/vector_store.py`, combined every
   workflow.
4. **Self-eval & feedback share one rubric.** `backend/agents/judge.py`
   scores faithfulness/relevance/style/citation and is used by (a) an inline
   self-eval node driving retry loops, (b) the offline eval harness against
   golden datasets, and (c) — via the same schema — the human feedback UI, so
   human and automated scores are directly comparable.
5. **Workflow registry.** Each agent architecture (deterministic pipeline,
   ReAct agent, supervisor/multi-agent) is a separate LangGraph graph
   registered in `backend/agents/registry.py` under a shared state schema
   and entrypoint contract. The API and eval harness select a workflow by
   name — new architectures plug in without touching callers.
6. **Frontend: single Next.js app**, `/chat` and `/documents` and `/evals` as
   routes rather than separate apps — one deployable, shared auth/session.
   Built as a static export (`output: "export"`, see
   `frontend/next.config.js`) and served by the backend in production — see
   decision 9.
7. **Observability: OpenTelemetry**, exported to any OTLP collector
   (`OTEL_EXPORTER_OTLP_ENDPOINT` — no cloud-specific export path), plus
   **self-hosted Langfuse** for LLM/agent-specific tracing — prompts, token
   usage, per-node execution, judge scores — kept off external SaaS since it
   carries prompt/document content.
8. **Persistence: one Postgres instance**, multiple roles — LangGraph
   checkpointer (conversation memory), LangGraph Store (user memory),
   document/ingestion metadata, feedback, eval results, and the pgvector
   chunk index.
9. **Deployment: Railway, single service.** The root `Dockerfile` builds the
   frontend's static export and serves it plus the API out of one FastAPI
   process (`backend/api/main.py` mounts `frontend/out`) — no separate
   frontend host, no CORS in production. `.railway/railway.ts` (Infrastructure
   as Code — Railway's `railway.toml`/`railway.json` "Config as Code" is
   deprecated, hard cutoff 2026-12-01) is all Railway needs to build/deploy
   on every push to `main`; there's no GitHub Actions deploy step. Postgres
   runs on Supabase (pgvector-enabled); Neo4j runs on AuraDB.
10. **Model provider: OpenAI API**, direct (`OPENAI_API_KEY`) — see
    `backend/config/llm_clients.py`. LangGraph is model-agnostic, so this is
    a low-cost-to-change default; swapping the chat model to Kyma
    (kymaapi.com) for deployment is a live option under consideration, not
    yet wired up (nothing in this codebase depends on it — see `agents.md`).

## Repo layout

```
backend/            FastAPI + LangGraph service (Python)
  agents/
    workflows/       one module per agent architecture, self-registers via agents/registry.py
    tools/           retrieval tools shared across workflows
    state.py          shared LangGraph state schema
    judge.py          shared 4-axis LLM-judge
    registry.py        workflow name -> compiled graph
  config/             OpenAI chat/embedding client factories
  retrieval/          Postgres/pgvector + Neo4j/AuraDB clients
  ingestion/          chunking, embedding, entity extraction -> pgvector index + graph
  memory/             Postgres-backed checkpointer (conversation) + store (user)
  api/                FastAPI app: chat (streaming/resume), documents, feedback routes;
                       also serves the built frontend in production (see main.py)
  evals/              golden-dataset harness, run against any registered workflow
  observability/      OpenTelemetry + Langfuse setup
  tests/
frontend/            Next.js app: /chat, /documents, /evals (static export)
Dockerfile            builds the frontend, then serves it + the API from one process
.railway/railway.ts    tells Railway to build with that Dockerfile (see "Deploying to Railway")
.github/workflows/     ci.yml (lint/test/eval on PR) — no deploy workflow, Railway
                        deploys straight from git via .railway/railway.ts
```

## Local Development

1. **Prerequisites**: Docker Desktop, Python 3.11+, Node 20+.
2. **Start local infra**: `docker compose up -d` — brings up Postgres with
   pgvector (`localhost:5432`) and Neo4j (browser `localhost:7474`, bolt
   `localhost:7687`).
3. **Configure env**: `cp .env.example .env`, then fill in `OPENAI_API_KEY`.
   The Neo4j/Postgres values already match `docker-compose.yml`'s defaults.
4. **Optional — Langfuse (per-question LLM/agent tracing)**: no self-hosting
   needed — sign up at [cloud.langfuse.com](https://cloud.langfuse.com),
   create a project, and copy its keys into `.env`
   (`LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_BASE_URL`).
   The same project is used for local dev, production, and offline evals —
   `LANGFUSE_TRACING_ENVIRONMENT` tags which is which, so traces stay
   segmented within the one project rather than needing separate infra.
   Leave the keys blank to disable tracing entirely (it no-ops safely).
5. **Run the backend**: `cd backend && uv venv .venv --python 3.11 && .venv/bin/pip install -e ".[dev]" && .venv/bin/alembic upgrade head && .venv/bin/uvicorn api.main:app --reload` — serves on `http://localhost:8000`. The `alembic upgrade head` step applies schema migrations (`backend/migrations/`) against `DATABASE_URL`; rerun it after pulling new migrations.
6. **Seed the knowledge graph** (once — in a second terminal, from `backend/`, once step 5's `pip install` has run): `.venv/bin/python -m ingestion.seed_master_catalog && .venv/bin/python -m ingestion.seed_synonyms` — populates Neo4j from the real fixtures in `backend/evals/` (`unite-master-csv.txt`, `synonyms-map.csv`). Per-document prose extraction (ticket 07) only attaches facts to parts this seed already created, so uploaded documents won't produce any graph facts until this has run at least once.
7. **Run the frontend**: `cp frontend/.env.local.example frontend/.env.local`, then `cd frontend && npm install && npm run dev` — serves on `http://localhost:3000` (or a different port if Langfuse is already on 3000). `.env.local` points the frontend at the backend on `:8000`, since Next's dev server and FastAPI run as two separate processes locally (they're one process in production — see ADR 0004).
8. **Run backend tests**: `docker compose --profile test up -d` (once) brings up a second, disposable Postgres (`:5433`) and Neo4j (`:7688`) alongside your dev instances. `cd backend && pytest` runs against those automatically — `backend/tests/conftest.py` points `DATABASE_URL`/`NEO4J_URI` at the test containers and applies migrations/constraints itself, so the test suite never writes into the database you're doing manual dev work against. `docker compose --profile test down` drops the test containers (and their data) when you're done; bring them back with the same `up` command.

Unlike a prior draft of this project, there's no enterprise-cloud dependency
for local dev — `OPENAI_API_KEY` works the same way locally and in
production, and every other service (Postgres, Neo4j) runs in Docker.

## Deploying to Railway

The app deploys as a **single Railway service** (root `Dockerfile` +
`.railway/railway.ts`): it builds the frontend's static export and serves
it plus the API from one FastAPI process, so there's no separate frontend
host and no CORS in production.

`.railway/railway.ts` is Railway's Infrastructure as Code format — its
older Config as Code format (`railway.toml`/`railway.json`) is deprecated
and stops working entirely on **2026-12-01**. `railway.toml` has been
deleted from this repo (its fields are all mirrored in `.railway/railway.ts`
and applied live), but **deploys are not currently working on this
project** — the last known-good deployment (from before this migration)
is still active and serving, so the app itself is not down, but a fresh
deploy fails silently within seconds of a successful build, for a cause
this session couldn't isolate via `railway logs`. See
`.railway/railway.ts`'s own comments for the full timeline, what's been
ruled out, and the recommended next step (check the Railway dashboard's
Deployments tab directly for an error the CLI doesn't surface). Do not
assume this migration is finished. Separately, `deploy.restartPolicyType`
doesn't reliably persist via `railway config apply` as of CLI v5.43.1 and
has no CaC fallback anymore — verify "Restart Policy" reads "On Failure"
in the Railway dashboard once deploys work again. The root-level
`package.json`/`package-lock.json` exist solely to provide the `railway`
npm package (`railway-ts-sdk`) that `.railway/railway.ts` imports from —
unrelated to `frontend/`'s own `package.json`.

1. **Database.** Postgres needs the `pgvector` extension. Either use
   Supabase (has it built in) or a `pgvector`-flavored Postgres template on
   Railway. Run `CREATE EXTENSION IF NOT EXISTS vector;` once against it.
   If pointing at Supabase, either connection string works — every psycopg
   connection this app opens (`config/db.py`, `retrieval/vector_store.py`,
   and LangGraph's own checkpointer/store) sets `prepare_threshold=0`, so a
   PgBouncer transaction-mode pooler (Supabase's pooled connection string,
   port 6543) swapping the underlying server connection between queries
   won't produce `prepared statement ... does not exist` errors.
2. **Graph DB.** Neo4j runs as AuraDB — provision it via the Neo4j Aura
   console / Azure or AWS Marketplace listing, not through this repo. Point
   `NEO4J_URI`/`NEO4J_USER`/`NEO4J_PASSWORD` at it.
3. **Create the Railway service** from this GitHub repo. Railway picks up
   `.railway/railway.ts`/`Dockerfile` automatically and redeploys on every
   push to `main` — no GitHub Actions deploy step involved.
4. **Environment variables.** Required: `OPENAI_API_KEY`, `DATABASE_URL`,
   `NEO4J_URI`/`NEO4J_USER`/`NEO4J_PASSWORD`, `JWT_SECRET` (a real random
   value — it falls back to an insecure dev default if unset), `ADMIN_EMAILS`
   (comma-separated; grants `is_admin`). Required only if Google sign-in is
   used: `GOOGLE_CLIENT_ID`/`GOOGLE_CLIENT_SECRET` and `GOOGLE_REDIRECT_URI`
   (set to `https://<your-railway-domain>/auth/google/callback`) — that
   exact URL must also be added as an authorized redirect URI on the OAuth
   client in the Google Cloud Console (a dashboard step, not code — see
   `auth/oauth.py`). Optional: `OTEL_EXPORTER_OTLP_ENDPOINT`/`LANGFUSE_*`. Railway
   injects `PORT` itself; don't set it. `NEXT_PUBLIC_API_BASE` and
   `FRONTEND_PUBLIC_URL` are local-dev-only (production is same-origin, see
   above) and should be left unset.
5. **Document storage volume.** `INGEST_DATA_DIR` (default `./data`) needs a
   Railway volume mounted at that path — ingested documents are plain files
   on disk, not object storage; a volume is required or they're lost on
   every redeploy since container filesystems are otherwise ephemeral.
6. **Seed the knowledge graph once, against production.** Uploaded
   documents produce zero graph facts until this has run — nothing else in
   this section triggers it, so it's easy to miss on a first deploy.

   Run it from your own machine with the Railway CLI, *not* `railway ssh`:
   `railway run` executes locally but with the service's production env
   vars injected, and since these two scripts only read local fixture CSVs
   (already in your checkout, under `backend/evals/`) and write to Neo4j
   over the network — no local output file the way `INGEST_DATA_DIR`-based
   document ingestion has — running locally against production Neo4j/Postgres
   is correct here, not a shortcut.

   ```bash
   cd backend
   railway run python -m ingestion.seed_master_catalog
   railway run python -m ingestion.seed_synonyms
   ```

   Order matters — `seed_synonyms` doesn't depend on the catalog, but
   `seed_master_catalog` must run before any document is indexed, since
   prose extraction only *attaches* facts to parts this seed already
   created. Both scripts are pure `MERGE`s in Neo4j, so they're idempotent —
   safe to rerun (e.g. after fixing a bad row in the source CSV) without
   duplicating anything.

## Adding a new agent workflow

1. Add `backend/agents/workflows/<name>.py` building on `agents.state.BaseAgentState`,
   ending with `register("<name>", build_graph)`.
2. Import it from `backend/agents/workflows/__init__.py`.
3. It's now selectable via `/chat/<name>/stream` and runnable through
   `backend/evals/harness.py` against the golden datasets — no other code
   changes required.

## Status

Auth, document ingestion (vector + graph + tray-layout legs), chat (both
the `deterministic` and `react_agent` workflows), the evals dashboard, and
the citation-scrolling PDF viewer are all built and deployed. The
`supervisor` multi-agent workflow is registered but deliberately left a
stub (`NotImplementedError`) — see ticket 27
(`.scratch/chat-documents-evals/issues/27-supervisor-multi-agent-deferred.md`).

See `agents.md` for standing technical decisions and conventions carried
forward for whoever (human or AI) works on this repo next, and
`build-log.md` for a chronological record of how it got here (currently
recorded through ticket 11 — later tickets are captured in their own
`.scratch/chat-documents-evals/issues/*.md` files instead).
