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

## Phase 2 — Chat, Documents, and Evals interfaces

Scoped via a grilling session into 17 tracer-bullet tickets under
`.scratch/chat-documents-evals/issues/` (auth, document upload/tagging, the
retrieval pipeline's two legs, the chat UI itself, and the evals dashboard +
feedback-promotion workflow), worked one at a time in dependency order.

- **Ticket 01 — Alembic migrations + `users` table.** The backend had zero
  migration tooling before this; every later ticket needs new tables, so
  this landed first. `alembic upgrade head` against `DATABASE_URL` is now
  part of local setup, CI (a throwaway Postgres service runs it before
  `pytest`), and production (`railway.toml`'s `releaseCommand`, once per
  deploy). First table: `users` (email, nullable `hashed_password` for
  OAuth-only accounts, `is_admin`, `created_at`).
- **Ticket 02 — Email/password auth + admin flag.** `backend/auth/`
  (bcrypt hashing, PyJWT sessions, `ADMIN_EMAILS` allowlist checked once at
  signup) plus `/auth/signup`, `/auth/login`, `/auth/me`. Frontend gained an
  `AuthProvider` (`frontend/lib/auth-context.tsx`, JWT in `localStorage`),
  `/login` and `/signup` pages, and an authenticated home page that only
  shows Documents/Evals links to admins. Along the way, fixed two
  pre-existing gaps that would have broken this ticket's own tests/CI:
  `documents.py`'s `UploadFile` route needed `python-multipart` (never
  installed), and `agents/state.py`'s `EvalScores` used `typing.TypedDict`,
  which pydantic v2 rejects on Python &lt;3.12 once a model (here,
  `feedback.py`'s `FeedbackRequest`) references it. Also: the frontend had no
  ESLint config or dependency at all despite CI running `npm run lint` —
  added `eslint-config-next` and a `.eslintrc.json`.
- **Ticket 03 — Google OAuth sign-in.** `backend/auth/oauth.py` (authorization-code
  flow against `GOOGLE_CLIENT_ID`/`GOOGLE_CLIENT_SECRET`) plus
  `/auth/google/login` and `/auth/google/callback` on the same router as
  ticket 02's password routes. The callback matches or creates a `users` row
  by email (nullable `hashed_password` — an OAuth-only account, or linked to
  an existing password account by matching email), applies the same
  `ADMIN_EMAILS` check, and hands the frontend a session by redirecting to
  `/?auth_token=...` — `AuthProvider` picks that query param up on load,
  since a full-page OAuth redirect has no other way to reach a static-export
  SPA. `/login` and `/signup` each gained a "Sign in/up with Google" link.
- **Ticket 04 — Document upload + list/status (admin-only).** `documents`
  table (status enum: queued/processing/done/failed) plus `backend/documents/`
  (repository + a placeholder background task in `service.py` that flips
  straight to "done" — real chunking/graph-writing land in tickets 06/07 by
  replacing that function's body, not by touching the route/background-task
  plumbing again). `backend/api/routes/documents.py`'s upload/list/get
  routes are gated by `auth.dependencies.require_admin`. Frontend: a real
  `/documents` page (admin-only, polls while anything's queued/processing)
  replacing the stub.
- **Branding + visual theme: OrthoMate.** The product was named and given a
  visual identity — a lit X-ray light table (dark) against a bone-white
  "diffuser" working surface, Instrument Serif for display type, IBM Plex
  Sans/Mono for body and data — first as a marketing landing page (an
  Artifact, not part of this repo), then carried into the actual frontend:
  `app/globals.css` (design tokens + component classes), a shared `app/nav.tsx`
  rendered from the root layout, and matching restyles of `/`, `/login`,
  `/signup`, and `/documents` (status now reads as a semantic badge —
  queued/processing/done/failed each get their own color, separate from the
  brand accent). Fonts load via a plain Google Fonts `<link>` in
  `app/layout.tsx` rather than `next/font/google`, which needs network access
  at `next build` time — this sandbox's Node trusts a different CA store than
  curl/the browser do, so `next/font` failed to compile locally; a runtime
  `<link>` sidesteps that risk entirely (relevant if the same gap exists in
  CI/Railway's build step).
  While verifying this in an actual browser (not just curl) for the first
  time, found and fixed a pre-existing gap: `api/main.py` had no
  `CORSMiddleware`, so every browser-side call from the frontend dev server
  (`:3000`) to the backend (`:8000`) was silently blocked by CORS — login,
  signup, everything. Fixed by allowing `FRONTEND_PUBLIC_URL` (already used
  for the OAuth redirect, defaults to `http://localhost:3000`) as the CORS
  origin; a no-op in production, where the frontend is served same-origin.
- **Ticket 05 — System and Document-Type tags.** Two admin-managed lookup
  tables, `systems` and `document_types` (`backend/tags/`, migration
  `eaeb89f1fc32`) — same shape, case-insensitive-unique `name`, no fixed
  enum. `documents` gained nullable `system_id`/`document_type_id` FKs;
  `backend/documents/repository.py` now LEFT JOINs both tables so list/get
  responses carry the tag names directly, no N+1 lookups from the frontend.
  New routes: `POST/GET /systems`, `POST/GET /document-types` (admin-only,
  409 on a duplicate name), and `PATCH /documents/{id}/tags` so a tag can be
  assigned either at upload (`system_id`/`document_type_id` form fields on
  `/documents/upload`) or afterward by editing. Frontend: a shared
  `TagSelect` (`frontend/app/documents/tag-select.tsx`) used both by the
  upload form's two pickers and by an inline picker in each table row —
  "pick an existing tag or add a new one on the spot," each `+ New` inline
  add calling straight through to the create endpoint. One structural bug
  caught by browser-driven (Playwright) testing before it shipped: the tag
  pickers were originally nested inside the upload `<form>`, and
  `TagSelect`'s own "add new" UI is itself a `<form>` — invalid, nested HTML
  forms that silently broke the "Add" button. Fixed by moving the pickers
  outside the upload form (their selection was already tracked in React
  state, never read from the form's own fields).
- **Manual Index/Reindex trigger.** Upload no longer auto-triggers the
  (still stubbed) ingestion pipeline — a new `pending` value on the
  `document_status` enum (migration `a796a69506ed`, added via
  `autocommit_block()` since Postgres won't let a new enum value be used in
  the same transaction it's added in) is now the default a freshly uploaded
  document rests in. A new admin-only `POST /documents/{id}/index` route
  (`backend/api/routes/documents.py`) moves it into the existing
  queued/processing/done/failed sequence — the same pipeline seam a re-tag
  already re-triggers, just invokable on demand. Each row in the Document
  Manager table gets an Index/Reindex button (`pending`/`failed` → "Index",
  `done` → "Reindex", `queued`/`processing` → disabled "Indexing…").
  Browser-driven (Playwright) testing confirmed the pending → queued → done
  transition end-to-end through the running app, not just the API tests.
- **Ticket 07 — Ingestion graph leg (entity extraction + Neo4j).** Before
  implementing, worked out the entity/relationship schema against the real
  fixtures in `backend/evals/` (`unite-master-csv.txt`, `doctype-hierarchy.csv`,
  `synonyms-map.csv`) and the golden-dataset question types, not just the
  literal stub signatures — see the schema note now in
  `.scratch/chat-documents-evals/issues/07-ingestion-graph-leg.md` and the new
  glossary entries in `CONTEXT.md` (Product Family, Tray, Master item file,
  Part, Procedure, Canonical term). Two separate ingestion paths, both
  writing through `backend/retrieval/graph_client.py`'s schema
  (`Part`/`Tray`/`ProductFamily`/`Procedure`/`Document`/`CanonicalTerm`,
  `COMPATIBLE_WITH`/`REQUIRES_TOOL`/`DIFFERENTIATES_FROM`/`REQUIRES`/
  `SOURCED_FROM`/`ALIAS_OF`/`ABBREVIATION_OF`):
  - `backend/ingestion/seed_master_catalog.py` and `seed_synonyms.py` are
    one-off, deterministic, CLI-run scripts (not part of per-document
    ingestion) that parse the real master item file and synonym map. The
    plate↔screw compatibility matrix resolves via exact SKU-prefix matching
    (a column header's parenthetical, e.g. `(MPSL27xx)`, names a real Item-No.
    prefix); tool requirements (guidewire/drill-bit/driver) resolve via
    numeric-token/driver-token matching restricted to the same tray, only
    written when exactly one candidate matches — ambiguous cases stay as a
    flat spec property rather than a guessed edge. `Part` identity is its SKU
    and only its SKU; the same SKU legitimately repeats across multiple tray
    rows (a shared screw/driver cataloged under more than one system), so
    `upsert_part` MERGEs by sku and adds a `BELONGS_TO_TRAY` edge per tray.
  - `backend/ingestion/entity_extraction.py` (LLM-based, via structured
    output) is the per-document path, wired into
    `backend/documents/service.py`'s `process_document` through
    `backend/ingestion/pipeline.py`. It's scoped to the document's tagged
    System's known-SKUs/known-trays list (fetched from the graph) and only
    ever attaches facts to `Part`/`Tray` nodes that already exist from the
    seed — `graph_client.py`'s `attach_differentiation`/`attach_procedure`
    additionally no-op server-side if a referenced SKU/tray isn't real, so a
    hallucinated reference is silently dropped, never merged in fuzzily.
    Guarded to degrade gracefully rather than fail documents in environments
    without `OPENAI_API_KEY` configured (CI has no such secret — see
    `ci.yml`'s note) or without a system tag or seeded catalog for it.
  - `backend/agents/tools/graph_query.py` and `synonym_resolve.py` are now
    real, both thin wrappers over `graph_client`.
  - Found and fixed a real bug while writing tests: `get_graph_client()`'s
    original `@lru_cache` singleton bound the async Neo4j driver to whichever
    event loop called it first — fine in production (one uvicorn loop for the
    process's life) but broke the moment two different loops touched it in
    the same process (FastAPI's `TestClient` vs. pytest-asyncio's session
    loop), surfacing as documents spuriously landing on `failed`. Fixed by
    caching one client per running loop instead of one per process.
  - CI (`ci.yml`) gained a `neo4j:5-community` service (free, no secret) so
    `graph_client`/the seed scripts get real integration coverage; LLM-
    dependent extraction tests skip themselves via `OPENAI_API_KEY`
    `skipif` rather than requiring that secret in CI.
- **Ticket 06 — Ingestion vector leg (chunking + embedding + pgvector).**
  Design settled via a grilling session against the *actual* shipped state
  of the pipeline (ticket 07 had landed by then) rather than the original
  stub signatures alone — see the fleshed-out spec in
  `.scratch/chat-documents-evals/issues/06-ingestion-vector-leg.md`.
  - New `backend/ingestion/text_extraction.py`, shared by both ingestion
    legs: `.pdf` via `pdfplumber` (kept for its per-character font-size
    data), `.txt`/`.md` via plain decode, anything else raises a typed
    `UnsupportedDocumentFormat`. Detected PDF headings (font size >1.15x
    the page's median body size) get normalized into a Markdown-style `# `
    prefix, so `backend/ingestion/chunking.py` has one uniform heading
    signal regardless of source format — a `.md` file already uses that
    syntax natively, and a `.txt` file with no such signal falls back to
    blank-line paragraph boundaries.
  - `chunk_document` packs each heading-delimited section's paragraphs into
    ~800-token windows (`tiktoken`, 100-token overlap); an oversized single
    paragraph gets token-windowed on its own. Chunk identity is a UUID PK
    plus a `document_id#chunk_index` citation string, matching the golden
    dataset's citation format.
  - New migration (`4b7c2d00de10`): `CREATE EXTENSION IF NOT EXISTS
    vector`, a `chunks` table (`pgvector.sqlalchemy.Vector(1536)`, a
    generated `tsvector` + GIN index for full-text, an HNSW index for
    cosine distance, `document_id` FK **ON DELETE CASCADE** so a document
    delete can't orphan its chunks, and denormalized nullable
    `system_id`/`document_type_id` for retrieval filtering without a join).
  - `VectorStoreClient` (`backend/retrieval/vector_store.py`):
    `upsert_chunks` deletes-then-inserts per `document_id` in one
    transaction (idempotent Reindex, no stale trailing chunks);
    `hybrid_search` fuses the vector-cosine and `ts_rank` legs with
    Reciprocal Rank Fusion (`k=10`, `2*top_k` candidates per leg) behind an
    optional typed `RetrievalFilters(system_id, document_type_id)`.
    `vector_search` (`backend/agents/tools/vector_search.py`) is now a real
    thin wrapper — embed the query, call `hybrid_search` — with no
    tool-level filter args yet, since nothing calls this tool until the
    ReAct graph (tickets 08/09) exists to validate that shape against.
  - `backend/ingestion/pipeline.py` gained `ingest_document_vectors` as a
    **sibling** to ticket 07's `ingest_document`, not a merge into it — the
    two legs take different auxiliary params and fail independently.
    `backend/documents/service.py`'s `process_document` now extracts text
    once and runs both legs against it; an extraction failure (including an
    unsupported format) now fails the document instead of the previous
    placeholder's silent skip. `ingest_document_vectors` picked up the same
    "no `OPENAI_API_KEY` configured -> skip rather than fail" guard the
    graph leg already had — without it, every existing OpenAI-key-less test
    environment (including this repo's own CI) would have started failing
    every document instead of leaving it at `done`.
  - Found and fixed two environment gotchas along the way, in the same
    "avoid the sandbox's fragile CA trust" spirit as `next/font`
    (documented earlier in this log): `tiktoken.get_encoding("cl100k_base")`
    fetches its merge table from Azure Blob Storage over HTTPS on first use,
    which both fails in this sandbox (different trusted CA store than curl)
    and is a fragile runtime dependency on a cold Railway container
    regardless — fixed by vendoring the hash-verified file at
    `backend/ingestion/.tiktoken_cache/` and pointing `TIKTOKEN_CACHE_DIR`
    at it, verified network-free by sabotaging `socket.socket.connect` in
    process. Separately, `psycopg`/pgvector needs an explicit `::vector`
    cast on a query *parameter* used with the `<=>` operator — Postgres
    accepts a bare Python-list-as-`double precision[]` parameter into an
    `INSERT` on a `vector` column (an implicit assignment cast), but `<=>`
    has no operator overload for that pairing.
  - Tests mirror ticket 07's split: `test_chunking.py`/`test_text_extraction.py`
    (the latter against a real PDF built at test time with `fpdf2`, a
    dev-only dependency) need no external service at all;
    `test_vector_store.py`/`test_ingestion_pipeline_vectors.py` run for
    real against Postgres/pgvector using small deterministic fake vectors
    (no key needed); only `test_embedding.py`'s real-OpenAI-call test is
    behind an `OPENAI_API_KEY` `skipif`. Manual/CLI check ran the full
    `process_document` orchestration against a real generated PDF end to
    end: real chunks, real `text-embedding-3-small` embeddings, and
    `hybrid_search` correctly ranking a torque-related query above an
    unrelated sterilization chunk.
- **Ticket 08 — Chat baseline (`deterministic`) workflow, end-to-end.**
  `backend/agents/workflows/deterministic.py`'s fixed pipeline
  (`resolve_synonyms -> hybrid_retrieve -> rerank -> generate -> self_eval`,
  looping through `reformulate` back to `resolve_synonyms` when faithfulness
  or relevance score low, bounded by `MAX_RETRIEVAL_LOOPS`) is real, backed
  by a real `judge_answer` (`backend/agents/judge.py`, structured-output
  LLM scoring on the same four axes as the feedback UI). `hybrid_retrieve`
  calls `vector_search` (ticket 06) only — `graph_query` isn't part of this
  fixed pipeline; choosing which entity to look up from free text is exactly
  the kind of judgment call a "no agentic tool choice" baseline doesn't
  make, that's `react_agent`'s job later. Full design, including why
  `query` and a new `search_query` field are kept separate, in
  `.scratch/chat-documents-evals/issues/08-chat-baseline-workflow.md`.
  - `backend/memory/checkpointer.py`'s `get_checkpointer` wraps
    `AsyncPostgresSaver.from_conn_string()` (itself an async context
    manager) the same way `retrieval/vector_store.py`'s `get_vector_store`
    already does; `backend/api/main.py` opens it once in a FastAPI
    `lifespan` and stores it on `app.state.checkpointer` -- one checkpointer
    for the process's life, matching `graph_client.py`'s Neo4j-driver
    singleton pattern rather than a fresh pool per chat request.
    `agents/registry.py`'s factory signature changed to take the
    checkpointer as a parameter (`react_agent.py`/`supervisor.py`'s stubs
    updated to match, still otherwise unimplemented).
  - `backend/api/routes/chat.py`'s `POST /chat/{workflow}/stream` streams
    `astream_events` over hand-rolled SSE (`thread`/`status`/`token`/`done`/
    `error` frames) -- it's a POST, so the native GET-only `EventSource`
    can't consume it; `frontend/lib/api.ts`'s `streamChat` parses the frames
    itself on the frontend side too. `thread_id` is server-generated as
    `{user_id}:{uuid4()}`; a client-resumed thread_id is checked by prefix
    and rejected (403) if it doesn't belong to the caller.
  - Real, not simulated: `frontend/app/chat/page.tsx` replaces the ticket-04
    placeholder with a live streaming UI (progressive status → token-by-token
    answer → citations), verified in an actual browser (Playwright), not
    just via `curl`.
  - Four real bugs found only by running the pipeline end-to-end (each now
    has a regression test, detailed in the ticket doc): `generate` was
    silently ignoring every prior conversation turn despite the checkpointer
    correctly persisting them; a retry appended *two* AI turns to permanent
    history instead of replacing the discarded draft (fixed by a new
    `finalize` node that's the only place `messages` gets written to);
    `retrieval_loop_count` had no reducer and could leak a spent retry
    budget across turns via the checkpointer; and the `done` event's
    `citations` listed every passage in the context window rather than only
    the ones the answer actually cited.
  - Also fixed: ticket 06's `TIKTOKEN_CACHE_DIR` vendoring fix only
    covered `backend/ingestion/chunking.py`'s own direct tiktoken call --
    `langchain_openai`'s embeddings client calls `tiktoken.encoding_for_model()`
    internally too, on a different code path that skipped the fix entirely
    whenever `chunking.py` hadn't already been imported first. Centralized
    into `backend/config/tiktoken_cache.py`, imported for its side effect
    wherever tiktoken gets used, directly or indirectly.

- **Ticket 11 — Chat inline per-message 4-axis feedback.** Wires up
  `backend/api/routes/feedback.py`'s `submit_feedback` (previously
  `NotImplementedError`) to real persistence: a new `feedback` table
  (migration `5d95b6897886`) plus `backend/feedback/` (models.py/
  repository.py, mirroring the domain package pattern). Scoped via a
  grilling session before implementing (`.scratch/chat-documents-evals/
  issues/11-inline-four-axis-feedback.md`); the key open question was where
  a stable `message_id` even comes from, since nothing exposed one before
  this ticket:
  - **`message_id` is LangGraph's own auto-assigned per-message uuid**
    (`langgraph.graph.message.add_messages` silently assigns one to every
    message once it lands in the checkpointer) -- not invented here, just
    surfaced for the first time through `ChatMessageOut`/the `"done"` SSE
    event. `feedback.message_id` is the table's primary key itself (no
    surrogate id), mirroring `chat_threads` using `thread_id` the same way;
    resubmitting on the same message overwrites via
    `ON CONFLICT (message_id) DO UPDATE` rather than a separate edit flow.
  - Every score is optional (`EvalScores` stays `total=False` all the way
    through) -- ticket 12 (free-text-only "Give feedback", not yet built)
    will reuse this same endpoint comment-only.
  - `submit_feedback` trusts `chat_threads.repository.owns_thread` (the
    thread-id-prefix check, extracted out of `chat.py` where it was
    previously private/duplicated) as the only real boundary; it does not
    verify `message_id` actually belongs to `thread_id` against the
    checkpointer -- the frontend only ever sends an id it already got from
    the backend, so a malformed one would just be a harmless orphan row.
  - `GET /chat/threads/{id}` now embeds each message's own `feedback` (if
    any) so reopening an old thread shows previously-submitted ratings
    rather than a blank control, not just the live turn that just streamed.
  - Frontend: `frontend/app/chat/message-feedback.tsx`, a compact
    always-visible control under every assistant bubble -- four 1-5 star
    rows (mapped to 0/.25/.5/.75/1), a flag toggle, an optional comment, one
    explicit "Save feedback" button (no per-click auto-save). Deliberately
    never pre-fills from the judge's own `eval_scores` (which also isn't
    persisted per-message anywhere) -- anchoring the human rater on the AI's
    own self-score would undermine the point of collecting an independent
    signal.
  - Verified for real against local Postgres (not just curl): 5 new
    `backend/tests/test_feedback_routes.py` cases (auth, cross-user
    ownership rejection, upsert-not-duplicate, partial/comment-only
    submission) plus a new `needs_openai_key` test in
    `test_chat_routes.py` proving a real turn's `done.message_id` round-trips
    through a feedback submission and back out of `GET /chat/threads/{id}`.
    Frontend changes pass `tsc --noEmit`, `next lint`, and `next build`
    clean; a live-browser (Playwright) check of the actual control was
    deliberately skipped this round to avoid writing more test data into
    the shared local dev Postgres mid-cleanup (see below) -- still open if
    wanted later.
  - **Local dev DB now has a disposable test-DB path (2026-09-01, by a peer
    session working this same repo concurrently):** `docker compose
    --profile test up -d` brings up `orthopedics-postgres-test` (`:5433`)
    and `orthopedics-neo4j-test` (`:7688` bolt);
    `backend/tests/conftest.py` points `DATABASE_URL`/`NEO4J_URI` at those
    instead of the dev instances whenever `CI` isn't set, running
    `alembic upgrade head` + Neo4j `ensure_constraints()` against the test
    DB at session start. `pytest` no longer touches local Postgres/Neo4j at
    `:5432`/`:7687` at all. Does **not** cover a running dev server
    (`uvicorn` on `:8000`, e.g. for a Playwright check) -- that still reads
    `.env` directly and hits dev Postgres/Neo4j for real.
