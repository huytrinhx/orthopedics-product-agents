# Context: Orthopedics Product Agents

Single-context repo — see `docs/agents/domain.md` for how this file and
`docs/adr/` should be consumed.

## Glossary

- **System** — one orthopedic implant/instrument product line (e.g. `MIS`,
  `REFLEX`). Golden-dataset queries and intent detection are scoped to a
  system; retrieval and citations stay within it. Not to be confused with
  "workflow" (below) or a computer system.
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
