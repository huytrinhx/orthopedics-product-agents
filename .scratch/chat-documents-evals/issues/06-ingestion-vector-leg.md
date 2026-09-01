# 06: Ingestion vector leg (chunking + embedding + pgvector)

**What to build:** An uploaded document actually becomes searchable. This
wires up `backend/ingestion/chunking.py`, `backend/ingestion/embedding.py`,
and `backend/retrieval/vector_store.py` for real, and makes the
`vector_search` tool return real hybrid (vector + full-text) results
instead of raising `NotImplementedError`. Design settled via a grilling
session against the *actual* current state of the pipeline (see "Design"
below) — not just the stub signatures.

**Blocked by:** 01 (Add Postgres migration tooling and a users table), 04
(Document upload + list with status). **Parallel to 07** (ingestion graph
leg, already shipped) — both legs plug into the same
`backend/documents/service.py`/`backend/ingestion/pipeline.py` seam, but
neither depends on the other's output (see `pipeline.py`'s own docstring).
This ticket also extends work from an interim ticket not in this numbered
list: the manual **Index/Reindex** button (`pending` → `queued` →
`processing` → `done`/`failed`, `POST /documents/{id}/index`) already
exists and is what actually drives `process_document` today — this ticket
does not need to build that trigger, only make the pipeline it triggers do
real work.

**Status:** done

**Cross-reference (from ticket 07's schema design):** `backend/evals/unite-
master-csv.txt` (the real master item file) is ingested by a separate
one-off script straight into the graph, not through this chunking/embedding
path or ticket 04's upload flow — don't expect it to show up in
`vector_search` results or the documents list.

## Design

### Current state this ticket builds on (read before starting)

- `backend/documents/service.py`'s `process_document` today does its own
  crude text handling: `path.read_text(encoding="utf-8")`, catching
  `UnicodeDecodeError` to silently **skip** the graph leg for any binary
  upload (a real PDF) while still marking the document `done`. This ticket
  replaces that placeholder with real extraction (below) — which also means
  PDFs will finally reach the graph leg's LLM extraction, a side effect of
  adding shared extraction infra, not scope creep.
- `backend/ingestion/pipeline.py`'s `ingest_document(document_id, text, *,
  filename, system, doc_type)` is the **graph leg only** (ticket 07). Do not
  rename or fold vector-leg logic into it — add a sibling function instead
  (see "Pipeline wiring" below), matching the "parallel, not dependent"
  relationship its own docstring already documents.
- No `chunks` table, no `CREATE EXTENSION vector`, and no
  PDF/token/text-splitting dependency exists anywhere yet. `pgvector` (the
  Python client) and `psycopg[binary]` are already installed; `pdfplumber`
  and `tiktoken` are new.
- `get_embeddings_model()` (`backend/config/llm_clients.py`) already wraps
  OpenAI `text-embedding-3-small` (1536-dim), cached singleton — reuse this,
  don't add a second embeddings client.
- The golden-dataset format (`backend/evals/golden_datasets/README.md`)
  cites `"doc-id#chunk-id"` — chunk identity has to support that string
  directly.

### Text extraction (new, shared by both legs)

A new module (e.g. `backend/ingestion/text_extraction.py`) dispatches on
file extension:

- `.pdf` → `pdfplumber` (needed for its per-character font-size data, which
  chunking also uses — see below)
- `.txt` / `.md` → plain decode
- anything else → a clear, typed error

`service.py`'s `process_document` calls this **once**, then passes the
result to both `ingestion.pipeline.ingest_document` (graph leg, unchanged
call signature) and the new vector-leg entry point (below). An extraction
failure (corrupt PDF, unsupported extension) now fails the document
(`status = "failed"`, readable reason) instead of today's silent skip —
this is an intentional behavior change from the current placeholder.

### Chunking (`backend/ingestion/chunking.py`)

Heading-aware, not fixed-size or purely paragraph-based:

- **PDF-sourced text**: a line is a heading if its font size exceeds the
  *page's median body-text font size* by more than 1.15x, using
  `pdfplumber`'s per-character layout data. (Bold was considered and
  rejected — technical documents bold inline implant/product names
  constantly, which would misfire as false section breaks.)
- **`.md`-sourced text**: literal Markdown heading syntax (`#`, `##`, …) —
  no layout data needed or available.
- **`.txt`-sourced text**: no layout signal exists at all; fall back to
  blank-line paragraph boundaries as the closest available approximation of
  "structure." (This fallback wasn't a separate grilled decision — it
  follows directly from "heading-aware where the signal exists, otherwise
  don't fabricate one.")
- Text between two headings/sections packs into ~800-token windows
  (`tiktoken`, matching `text-embedding-3-small`'s token accounting) with a
  100-token overlap. A single section that alone exceeds the window budget
  gets the same token-windowing applied within it.
- Each returned chunk carries `chunk_index` (sequential per document,
  starting at 0) and `section_title` (nullable — the detected heading text,
  or `None` for the paragraph-fallback path). `section_title` is persisted
  on the chunk row for future citation/debugging display, even though
  nothing reads it yet.

### Chunk identity and citations

- Chunk rows get a UUID primary key (consistent with how `documents` itself
  is keyed — normal FK/delete semantics, no special-casing).
- The citation-facing identity is `f"{document_id}#{chunk_index}"`, matching
  the golden-dataset format exactly.

### Schema (new migration, `down_revision` = `a796a69506ed`)

- `CREATE EXTENSION IF NOT EXISTS vector` (production Postgres is Supabase,
  pgvector-enabled — see README — so this isn't a deployment risk).
- `chunks` table:
  - `id` UUID PK, `gen_random_uuid()` default
  - `document_id` UUID, FK → `documents(id)`, **`ON DELETE CASCADE`** — a
    document delete must not orphan its chunks, and a DB-level cascade
    can't be forgotten by some future direct-delete code path the way an
    application-level cleanup call could
  - `chunk_index` int, `content` text, `section_title` text nullable
  - `embedding vector(1536)`, plus a vector index (HNSW with
    `vector_cosine_ops` if the installed pgvector version supports it,
    IVFFlat otherwise) for the cosine-distance leg of hybrid search
  - a generated `tsvector` column (`to_tsvector('english', content)`) with a
    GIN index, for the full-text leg
  - `system_id`, `document_type_id` — nullable, **denormalized** (copied
    from the parent `documents` row at ingest time, not joined at query
    time) so filtering doesn't need a join back to `documents`. A re-tag
    already fully re-triggers ingestion (existing `PATCH /tags` behavior),
    so these stay in sync without extra bookkeeping.
  - `created_at` timestamptz, `now()` default

### Retrieval (`backend/retrieval/vector_store.py`)

- `VectorStoreClient.upsert_chunks(document_id, chunks)`: **delete-then-insert**
  — delete all existing rows for `document_id`, then bulk-insert the fresh
  set, inside one transaction. Makes the Reindex button idempotent with no
  stale trailing chunks; the momentary zero-chunk window inside the
  transaction doesn't matter yet since nothing queries chunks concurrently
  with ingestion (no live agent/query path exists yet — see below).
- `VectorStoreClient.hybrid_search(query, vector, top_k=8, filters=None)`:
  real **Reciprocal Rank Fusion**. Each leg (cosine-distance ORDER BY
  `embedding <=> vector`, and `ts_rank`/`plainto_tsquery` on the generated
  `tsvector` column) pulls `2 * top_k` candidates before fusion — zero
  oversampling would give RRF nothing extra to promote from either leg,
  defeating the point of fusing two ranked lists. Fusion constant `k = 10`
  (small, not the RRF paper's `k=60`, which was tuned for large web-scale
  rank lists — `top_k` here is single digits).
- `filters`: a small typed structure (dataclass/TypedDict, not a raw
  `dict`, matching the pydantic-model convention already used elsewhere in
  this codebase — e.g. `documents/models.py`'s `SetDocumentTagsRequest`):
  `RetrievalFilters(system_id: uuid.UUID | None = None, document_type_id:
  uuid.UUID | None = None)`, `None` fields meaning unfiltered. One param so
  a third filter dimension can be added later without another signature
  change.
- `vector_search` (`backend/agents/tools/vector_search.py`): real thin
  wrapper — `embed_texts([query])[0]`, call `hybrid_search`, shape results
  with the `document_id#chunk_index` citation string. **Filters are not
  wired here** — nothing calls this tool yet (the ReAct graph that would,
  `backend/agents/workflows/react_agent.py`, is itself still
  `NotImplementedError`, tickets 08/09's territory), so filter plumbing
  exists in `hybrid_search` without a caller exercising it yet. Add the
  tool-level filter args once there's a real caller to validate them
  against.

### Embedding (`backend/ingestion/embedding.py`)

- Batches at 2048 texts per call (OpenAI's documented per-request item cap
  for embeddings) — fewer round-trips than an arbitrary smaller batch size
  for a long technical document with many chunks.
- Retries transient/rate-limit errors up to 3x with exponential backoff
  (1s / 2s / 4s). The retry cost of resending a large batch on a transient
  429 is small next to the round-trip savings of batching at the real
  limit.

### Pipeline wiring

- `backend/ingestion/pipeline.py` gains a **sibling** function, e.g.
  `ingest_document_vectors(document_id: str, text: str, *, system_id:
  uuid.UUID | None, document_type_id: uuid.UUID | None) -> None`: chunk →
  embed → `VectorStoreClient.upsert_chunks`. Kept separate from the
  existing graph-leg `ingest_document`, not merged into it — the two legs
  take different auxiliary params (`system`/`doc_type` *names* for the
  graph leg's node properties vs. `system_id`/`document_type_id` *UUIDs*
  for the chunk table's denormalized filter columns) and fail
  independently.
- `backend/documents/service.py`'s `process_document`: extract text once
  (above), then call both `ingest_document` (graph, existing) and
  `ingest_document_vectors` (new). Either raising propagates to the
  existing `try/except` → `status = "failed"` with the exception message —
  no new error-handling shape needed there.

### Testing (mirrors ticket 07's established convention)

Ticket 07 split its tests exactly along "needs a real LLM call" vs.
"doesn't": `test_ingestion_pipeline.py`/`test_graph_client.py` run in CI
with no secret against real Postgres/Neo4j; `test_entity_extraction.py`
gates its LLM-calling tests behind an `OPENAI_API_KEY`-checked `skipif`.
Mirror that split here rather than skipping automated coverage entirely:

- Real-Postgres, no-secret-needed, run in CI: chunking boundary logic
  (heading detection, token windowing, oversized-section fallback), the
  `chunks` table round-trip, `upsert_chunks`'s delete-then-insert
  reindex behavior, and `hybrid_search`'s RRF ranking/filtering — using
  small **deterministic fake vectors** (not real embeddings) so the
  storage/ranking SQL gets real coverage without an API key, exactly as
  ticket 07 exercises real graph writes/queries without needing the LLM.
- `OPENAI_API_KEY`-gated `skipif`: only `embed_texts`'s actual call to
  OpenAI needs this — everything downstream of "we have a vector" doesn't.
- Manual/CLI check (per this ticket's original acceptance criteria,
  unchanged): upload a real PDF, confirm chunks + embeddings land, confirm
  `vector_search`/`hybrid_search` ranks relevant chunks above irrelevant
  ones, and confirm the Reindex button replaces rather than duplicates that
  document's chunks.

### Implementation notes (found while building, not in the original spec)

- `ingest_document_vectors` gained the same `if not os.environ.get("OPENAI_API_KEY"): return` guard `ingest_document` (the graph leg) already had — without it, every existing test/environment that doesn't configure an OpenAI key (CI, this repo's own `test_documents.py`/`test_tags.py`) would have started failing documents instead of leaving them at `done`, since the vector leg (unlike the graph leg) has no other guard that would short-circuit first.
- `psycopg`/pgvector gotcha: a plain Python list bound as a query parameter dumps as `double precision[]`. Postgres accepts that into an `INSERT` on a `vector` column via an implicit assignment cast, but the `<=>` operator has no `vector <=> double precision[]` overload — the parameter needs an explicit `::vector` cast in the `ORDER BY` clause.
- `tiktoken.get_encoding("cl100k_base")` fetches its merge table from Azure Blob Storage over HTTPS on first use if not already cached locally. This sandbox's Python doesn't trust the same CA store curl/the system do (the same class of issue already documented in this file for `next/font`), and depending on outbound network access for a static, versioned file on every cold container start is fragile regardless of that. Fixed by vendoring the file at `backend/ingestion/.tiktoken_cache/` (hash-verified against tiktoken's own expected SHA-256) and pointing `TIKTOKEN_CACHE_DIR` at it from `chunking.py` — zero network calls anywhere, verified by sabotaging `socket.socket.connect` in-process and confirming `chunk_document` still works.
- Full manual/CLI check ran against a real generated PDF (two headings, `pdfplumber`+`fpdf2`) through the actual `process_document` orchestration (not just the vector-leg function directly): landed on `status="done"`, wrote 2 real chunks with real `text-embedding-3-small` embeddings, and `hybrid_search` correctly ranked a torque-related query above an unrelated sterilization chunk. Reindex idempotency (delete-then-insert) and cascade-delete-on-document-delete are covered by `test_vector_store.py` against real Postgres.

### Explicitly out of scope for this ticket

- **DOCX support.** PDF and plain text (`.txt`/`.md`) cover the realistic
  case; add DOCX if a real IFU shows up in that format.
- **Query-time filter args on the `vector_search` tool itself.** The
  `chunks` schema and `hybrid_search` support filtering now; wiring
  `system_id`/`document_type_id` into the tool's own call signature waits
  for a real caller (tickets 08/09's ReAct graph) to validate the API
  shape against.
- **Persisted `section_title` being read/displayed anywhere.** Stored now
  because it's free once headings are detected; using it in a citation UI
  is a future ticket.
- **`backend/evals/unite-master-csv.txt`.** Ingested straight into the
  graph by ticket 07's one-off seed script, not through this
  chunking/embedding path — it won't show up in `vector_search` results.

## Checklist

- [x] Shared text extraction module: PDF via `pdfplumber`, plain text via
      `.txt`/`.md`; unsupported extensions raise a clear, typed error.
      Replaces `service.py`'s current `read_text`/`UnicodeDecodeError`
      placeholder; both ingestion legs consume its output.
- [x] `chunk_document` real implementation: heading-aware for PDF text
      (font size >1.15x page-median body size), Markdown-heading-aware for
      `.md`, paragraph-fallback for `.txt`. Packs section text into
      ~800-token windows (100-token overlap, `tiktoken`); an oversized
      section is token-windowed the same way. Returns `chunk_index` +
      `section_title` per chunk.
- [x] `embed_texts` real implementation: 2048-item batches, retry with
      exponential backoff (1s/2s/4s) on transient/rate-limit errors.
- [x] Migration: `CREATE EXTENSION IF NOT EXISTS vector`; `chunks` table
      (UUID PK, `document_id` FK **ON DELETE CASCADE**, `chunk_index`,
      `content`, `section_title` nullable, `embedding vector(1536)` +
      vector index, generated `tsvector` + GIN index, denormalized nullable
      `system_id`/`document_type_id`, `created_at`).
- [x] `VectorStoreClient.upsert_chunks`: delete-then-insert per
      `document_id`, one transaction.
- [x] `VectorStoreClient.hybrid_search`: real Reciprocal Rank Fusion
      (`k=10`, `2*top_k` candidates per leg), optional typed
      `RetrievalFilters` (`system_id`/`document_type_id`).
- [x] `vector_search` agent tool: real thin wrapper (embed query → hybrid
      search → `document_id#chunk_index`-cited results); no tool-level
      filter args yet (no caller to validate them against).
- [x] `backend/ingestion/pipeline.py` gains `ingest_document_vectors` as a
      sibling to (not a merge into) the existing graph-leg `ingest_document`.
- [x] `backend/documents/service.py`'s `process_document` calls the shared
      extractor once, then both ingestion legs; an extraction failure now
      fails the document instead of silently skipping the graph leg (a
      behavior change from today's binary-upload handling).
- [x] Tests mirror ticket 07's split: real-Postgres coverage (with
      deterministic fake vectors) for chunking/storage/hybrid-search/reindex
      runs in CI with no secret; `embed_texts`'s real OpenAI call is behind
      an `OPENAI_API_KEY`-gated `skipif`.
- [x] Manual/CLI check against a real uploaded PDF confirms the full
      chain end-to-end, including that Reindex replaces rather than
      duplicates chunks.
