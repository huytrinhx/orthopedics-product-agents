# Build Log

Chronological record of how this repo got here, phase by phase.

## Phase 0 — Initial scaffold

Directory structure, shared contracts, and infra/CI wiring landed as one
scaffold: LangGraph workflow registry (`backend/agents/`), shared judge
(`backend/agents/judge.py`), FastAPI app (`backend/api/`), retrieval/
ingestion contracts, Postgres-backed memory, Next.js frontend shell
(`/chat`, `/documents`), OpenTelemetry/Langfuse observability hooks, and an
Azure-native deployment path (Bicep: AKS, AI Search, Postgres Flexible
Server, ACR). Every retrieval/LLM/ingestion call was left as a stub
(`NotImplementedError`) pending the experimentation phase. Golden eval
datasets for two real product systems (MIS, REFLEX) were added under
`backend/evals/golden_datasets/`, generated from human-reviewed
`feedback-notes.csv` via `build_dataset.py`.

The README's architecture-decision list, drafted partly against a
different project (fhir-bridge) as a reference, ended up with a few
self-contradictions never caught before this phase: it claimed both
"AKS" and "Supabase + Railway" as the deployment target, both
"Azure AI Search" (in code) and "OpenAI API" (in the decision list) as the
model/retrieval provider, and referenced Azure Blob Storage
(`docker-compose.yml`'s `azurite` service) that nothing in the codebase
actually used.

## Phase 1 — Infra migration: Azure/AKS to Railway

Resolved the contradictions above by deliberately moving off Azure
infrastructure, in favor of Railway (see `docs/adr/0001` through `0005` for
the reasoning behind each piece):

- **Deployment**: retired `infra/bicep/*` and `.github/workflows/deploy.yml`.
  Added a root `Dockerfile` + `railway.toml` for a single Railway service,
  deployed via git integration (no CI deploy step).
- **Retrieval**: dropped Azure AI Search. Postgres gained pgvector
  (`docker-compose.yml`'s `postgres` service, `backend/retrieval/vector_store.py`)
  for the vector leg; Postgres full-text search covers the keyword leg.
  Neo4j/AuraDB is unchanged, and synonym resolution already queried it
  directly rather than through a synced index.
- **LLM/embeddings**: dropped Azure OpenAI and the enterprise
  service-principal auth (`backend/config/azure_clients.py` removed,
  replaced by `backend/config/llm_clients.py` against the plain OpenAI
  API). Kyma (kymaapi.com) was raised as a possible future chat-model
  provider for deployment, but deliberately left unwired — nothing depends
  on it yet.
- **Document storage**: dropped Azurite/Blob Storage (`docker-compose.yml`);
  nothing in the codebase used it. Documents now live under
  `INGEST_DATA_DIR` on local disk / a Railway volume.
- **Service topology**: frontend now builds as a Next.js static export and
  is served by the FastAPI backend as one Railway service, rather than two
  separately deployed images.
- **Docs**: added `agents.md`, `product.md`, `build-log.md` (this file),
  `CONTEXT.md`, `docs/adr/`, and `docs/agents/` — a persistent-context
  layout for whoever (human or AI) picks this repo up next, so these
  decisions and the reasoning behind them aren't re-derived from scratch.

Nothing in this phase touched actual retrieval/agent logic — every call
site changed was still a stub before and after.
