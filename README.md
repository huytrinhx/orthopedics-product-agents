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
   frontend host, no CORS in production. `railway.toml` is all Railway needs
   to build/deploy on every push to `main`; there's no GitHub Actions deploy
   step. Postgres runs on Supabase (pgvector-enabled); Neo4j runs on AuraDB.
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
railway.toml           tells Railway to build with that Dockerfile
.github/workflows/     ci.yml (lint/test/eval on PR) — no deploy workflow, Railway
                        deploys straight from git via railway.toml
```

## Local Development

1. **Prerequisites**: Docker Desktop, Python 3.11+, Node 20+.
2. **Start local infra**: `docker compose up -d` — brings up Postgres with
   pgvector (`localhost:5432`) and Neo4j (browser `localhost:7474`, bolt
   `localhost:7687`).
3. **Configure env**: `cp .env.example .env`, then fill in `OPENAI_API_KEY`.
   The Neo4j/Postgres values already match `docker-compose.yml`'s defaults.
4. **Optional — Langfuse (local LLM tracing)**: Langfuse's self-host stack
   (web, worker, Postgres, ClickHouse, Redis, MinIO) is non-trivial enough
   that we run their own maintained compose file rather than duplicating it
   here:
   ```
   git clone https://github.com/langfuse/langfuse.git ../langfuse-local
   cd ../langfuse-local && docker compose up -d
   ```
   Open `http://localhost:3000`, create a project, and copy the generated
   public/secret keys into `.env` (`LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY`).
5. **Run the backend**: `cd backend && uv venv .venv --python 3.11 && .venv/bin/pip install -e ".[dev]" && .venv/bin/alembic upgrade head && .venv/bin/uvicorn api.main:app --reload` — serves on `http://localhost:8000`. The `alembic upgrade head` step applies schema migrations (`backend/migrations/`) against `DATABASE_URL`; rerun it after pulling new migrations.
6. **Seed the knowledge graph** (once — in a second terminal, from `backend/`, once step 5's `pip install` has run): `.venv/bin/python -m ingestion.seed_master_catalog && .venv/bin/python -m ingestion.seed_synonyms` — populates Neo4j from the real fixtures in `backend/evals/` (`unite-master-csv.txt`, `synonyms-map.csv`). Per-document prose extraction (ticket 07) only attaches facts to parts this seed already created, so uploaded documents won't produce any graph facts until this has run at least once.
7. **Run the frontend**: `cp frontend/.env.local.example frontend/.env.local`, then `cd frontend && npm install && npm run dev` — serves on `http://localhost:3000` (or a different port if Langfuse is already on 3000). `.env.local` points the frontend at the backend on `:8000`, since Next's dev server and FastAPI run as two separate processes locally (they're one process in production — see ADR 0004).

Unlike a prior draft of this project, there's no enterprise-cloud dependency
for local dev — `OPENAI_API_KEY` works the same way locally and in
production, and every other service (Postgres, Neo4j) runs in Docker.

## Deploying to Railway

The app deploys as a **single Railway service** (root `Dockerfile` +
`railway.toml`): it builds the frontend's static export and serves it plus
the API from one FastAPI process, so there's no separate frontend host and
no CORS in production.

1. **Database.** Postgres needs the `pgvector` extension. Either use
   Supabase (has it built in) or a `pgvector`-flavored Postgres template on
   Railway. Run `CREATE EXTENSION IF NOT EXISTS vector;` once against it.
2. **Graph DB.** Neo4j runs as AuraDB — provision it via the Neo4j Aura
   console / Azure or AWS Marketplace listing, not through this repo. Point
   `NEO4J_URI`/`NEO4J_USER`/`NEO4J_PASSWORD` at it.
3. **Create the Railway service** from this GitHub repo. Railway picks up
   `railway.toml`/`Dockerfile` automatically and redeploys on every push to
   `main` — no GitHub Actions deploy step involved.
4. **Environment variables.** Set `OPENAI_API_KEY`, `DATABASE_URL`,
   `NEO4J_URI`/`NEO4J_USER`/`NEO4J_PASSWORD`, and optionally
   `OTEL_EXPORTER_OTLP_ENDPOINT`/`LANGFUSE_*`.
5. **Document storage volume.** `INGEST_DATA_DIR` (default `./data`) needs a
   Railway volume mounted at that path — ingested documents are plain files
   on disk, not object storage; a volume is required or they're lost on
   every redeploy since container filesystems are otherwise ephemeral.

## Adding a new agent workflow

1. Add `backend/agents/workflows/<name>.py` building on `agents.state.BaseAgentState`,
   ending with `register("<name>", build_graph)`.
2. Import it from `backend/agents/workflows/__init__.py`.
3. It's now selectable via `/chat/<name>/stream` and runnable through
   `backend/evals/harness.py` against the golden datasets — no other code
   changes required.

## Status

This is a scaffold: directory structure, shared contracts (state schema,
judge interface, workflow registry), and infra/CI wiring are in place.
Workflow graph logic, retrieval clients, and the frontend UI are stubs
(`NotImplementedError` / `TODO`) pending the experimentation phase.

See `agents.md` for standing technical decisions and conventions carried
forward for whoever (human or AI) works on this repo next, and
`build-log.md` for a chronological record of how it got here.
