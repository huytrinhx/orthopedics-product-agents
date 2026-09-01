# Context: Orthopedics Product Agents

Single-context repo — see `docs/agents/domain.md` for how this file and
`docs/adr/` should be consumed.

## Glossary

- **System** — one orthopedic implant/instrument product line (e.g. `MIS`,
  `REFLEX`). Golden-dataset queries and intent detection are scoped to a
  system; retrieval and citations stay within it. Not to be confused with
  "workflow" (below) or a computer system. Same granularity as the graph's
  `Product Family` node and the document-level System tag
  (`backend/documents/models.py`) — see `Tray` below for the finer
  granularity actually used inside the master item file.
- **Product Family** — the graph node (`backend/retrieval/graph_client.py`)
  for a "System" (above) — `MIS`, `REFLEX`. Distinct from `Tray`, one level
  finer.
- **Tray** — one specific implant/instrument set, at the granularity the
  master item file (`backend/evals/unite-master-csv.txt`) actually lists
  under its `System` column (e.g. `REFLEX® HYBRID Implant System`, `REFLEX®
  MINI Nitinol Staple System`) — one level below `Product Family`/System tag.
  A `Procedure` (below) requires a specific `Tray`, not just a `Product
  Family` — this is what answers "if I'm doing a hybrid MTP fusion, do I only
  need the hybrid tray?"
- **Master item file** — `backend/evals/unite-master-csv.txt`, the real
  catalog of every `Part` (SKU, description, spec columns, and a plate↔screw
  compatibility matrix). The authoritative source for `Part`/`Tray`/`Product
  Family` graph nodes, seeded by a one-off script — distinct from the
  per-document prose extraction that runs on brochures/technique guides at
  upload time and only attaches facts to `Part`s that already exist here.
- **Part** — a single catalog item, identified by SKU (the master item
  file's `Item No.`) and only that — no name-based or fuzzy matching. A
  mention in a document's prose without a resolvable SKU does not become a
  `Part` node.
- **Procedure** — a named surgical procedure (bunion, Akin, lapidus, MTP
  fusion, ...) as a graph entity, distinct from "Technique/procedural"
  question types in the golden datasets, which are about the same concept
  but expressed as free text rather than a graph node.
- **Workflow** — one agent architecture (deterministic pipeline, ReAct
  agent, supervisor/multi-agent), registered by name in
  `backend/agents/registry.py`. Selected per-request (`/chat/<workflow>/stream`)
  and by the eval harness — not the same concept as "system."
- **Judge** — the shared LLM-based scorer (`backend/agents/judge.py`)
  producing four axis scores: faithfulness, relevance, style, citation. Used
  identically by the inline self-eval retry loop, the offline eval harness,
  and the human feedback UI.
- **Synonym resolution** — expanding a query against the canonical
  entity/synonym graph in Neo4j/AuraDB before retrieval (e.g. "ACL repair"
  -> also matches variant phrasings within the same system). Distinct from
  the vector/full-text retrieval step itself.
- **Canonical term** — the anchor graph node for a cluster of equivalent
  terms from `backend/evals/synonyms-map.csv`, resolved via
  `synonym_resolve`. Covers two distinct relationships to the canonical
  node: many-to-many synonym clusters (e.g. `guidepin`/`guidewire`/`wire`/
  `pin`) and 1:1 abbreviation expansions (e.g. `FT` -> `Full thread`) — both
  resolve identically from a caller's perspective.
- **Golden dataset** — a JSONL file of real, human-reviewed query/expected-
  answer/expected-citation examples for one system, generated from
  `feedback-notes.csv` via `build_dataset.py`. Consumed by
  `backend/evals/harness.py`.
- **Chunk** — a retrieval-sized slice of an ingested document
  (`backend/ingestion/chunking.py`), embedded and stored in the pgvector
  index (`backend/retrieval/vector_store.py`).

## Where decisions live

- `docs/adr/` — architectural decision records for this repo's
  infrastructure choices (deployment, retrieval backend, LLM provider,
  service topology, document storage).
- `agents.md` — standing technical conventions that aren't ADR-worthy on
  their own but shouldn't be silently reversed.
- `product.md` — what's actually known about product behavior, distinct
  from infrastructure.
