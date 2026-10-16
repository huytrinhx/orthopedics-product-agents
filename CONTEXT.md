# Context: Orthopedics Product Agents

Single-context repo — see `docs/agents/domain.md` for how this file and
`docs/adr/` should be consumed.

## Glossary

- **Product Family** — the top-level container of the catalog hierarchy
  (Product Family > Tray > Part), one per value of the master item file's
  `Family` column (e.g. `MIS - Foot Recon System`, `REFLEX HYBRID`,
  `Cannulated Screws`; 18 in the Oct-26 catalog). A graph node
  (`backend/retrieval/graph_client.py`) mirrored into Postgres
  (`product_families`, `backend/catalog/`) with an admin-editable display
  name. There is no brand level above it — the old `MIS`/`REFLEX` families
  are retired.
- **Tray** — one specific implant/instrument set within exactly one
  `Product Family`, named by the master item file's `Tray` column minus the
  `UNITE® Foot & Ankle | ` brand segment (e.g. `REFLEX® HYBRID Implant
  System`, `Soft Tissue | Comprehensive Instrument Tray`). Mirrored into
  Postgres (`trays`) alongside its family. A `Procedure` (below) requires a
  specific `Tray`, not just a `Product Family` — this is what answers "if
  I'm doing a hybrid MTP fusion, do I only need the hybrid tray?"
- **System** *(legacy)* — the document-level System tag
  (`backend/documents/models.py`, the Postgres `systems` table, e.g. `MIS -
  Foot Recon`) that predates Product Families, and the old name of the
  master item file's `Tray` column. Golden-dataset queries and intent
  detection are still scoped by it. Family-scoped graph lookups translate a
  System tag to its Product Family through an explicit map
  (`backend/catalog/legacy_systems.py`) until documents are re-tagged
  against families. Not to be confused with "workflow" (below) or a
  computer system.
- **Master item file** — `backend/evals/master-product-detail-oct-26.xlsx`
  (the stakeholders' "Master Product Detail File (Oct-26) - Do Not Edit",
  committed as-is), the real catalog of every `Part` (family, tray, SKU,
  description, spec columns, and a plate↔screw compatibility matrix). The
  authoritative source for `Product Family`/`Tray`/`Part`, seeded by a
  one-off script — distinct from the per-document prose extraction that runs
  on brochures/technique guides at upload time and only attaches facts to
  `Part`s that already exist here. Typos in it (e.g. `Ankle Fracture Plating
  Sytem`) are fixed through display names, not by editing the file.
- **Part** — a single catalog item, identified by SKU (the master item
  file's `Item No.`) and only that — no name-based or fuzzy matching. A
  mention in a document's prose without a resolvable SKU does not become a
  `Part` node.
- **Procedure** — a named surgical procedure (bunion, Akin, lapidus, MTP
  fusion, ...) as a graph entity, distinct from "Technique/procedural"
  question types in the golden datasets, which are about the same concept
  but expressed as free text rather than a graph node.
- **Question type** — what kind of thing a rep's question asks about
  (`product_characteristics`, `technique_procedural`, `features_benefits`,
  ..., `unclassified`), classified per turn by the deterministic workflow.
  The slugs are code (`backend/agents/question_types.py`).
- **Doc-type hierarchy** — for each question type, the ranked list of
  document types retrieval filters to, most authoritative first. It's data
  (`question_type_document_types`, edited in Settings) keyed by document
  type id. An empty list, which `unclassified` always has, means no filter.
  A document type that no question type lists is **unmapped**: it's only
  reachable on an unfiltered turn. See
  `docs/adr/0006-doc-type-hierarchy-is-data.md`.
- **Workflow** — one agent architecture (deterministic pipeline, ReAct
  agent, supervisor/multi-agent), registered by name in
  `backend/agents/registry.py`. Selected per-request (`/chat/<workflow>/stream`)
  and by the eval harness — not the same concept as "system."
- **Judge** — the shared LLM-based scorer (`backend/agents/judge.py`)
  producing four axis scores: faithfulness, relevance, style, citation. Used
  identically by the offline eval harness and the human feedback UI.
- **Fact check** — the deterministic per-turn check (`backend/agents/
  fact_check.py`) every workflow's `self_eval` node runs on a draft answer:
  cited passage ids, SKUs, and thread types verified against that turn's
  own sources. A failed check triggers one **correction pass** — the draft
  regenerated with the specific problems listed.
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
