# 07: Ingestion graph leg (entity extraction + Neo4j)

**What to build:** Entities and synonyms get extracted from uploaded documents
into the Neo4j/AuraDB graph, making `graph_query` and `synonym_resolve` return
real data instead of raising `NotImplementedError`. Two separate ingestion
paths feed this graph — a one-off deterministic seed from the master item
file, and per-document LLM prose extraction that attaches to it — see
"Schema" below. This design was worked out against the real fixtures in
`backend/evals/` (`unite-master-csv.txt`, `doctype-hierarchy.csv`,
`synonyms-map.csv`) and the golden-dataset question types, not just the
literal text of the graph_client stubs.

**Blocked by:** 01 (Add Postgres migration tooling and a users table), 04
(Document upload + list with status)

**Status:** done

## Schema

### Entities

| Node | Identity key | Notes |
|---|---|---|
| `Document` | doc id | carries `doc_type`, `system`/tray tags inherited from ticket 05 (not re-inferred by the LLM) |
| `ProductFamily` | name (`MIS`, `REFLEX`) | top-level; matches ticket 05's System tag and the golden datasets' `system` field |
| `Tray` | master-file `System` column value (e.g. `REFLEX HYBRID Implant System`) | finer than `ProductFamily` — see note below |
| `TraySection` | tray + section label | physical position within a tray (e.g. "top level" vs "bottom level"); best-effort, likely sparse until tray-overhead-guide extraction exists — don't block this ticket on solving diagram/image extraction |
| `Part` | **SKU (`Item No.`) only** | no name-based or fuzzy-matched identity — two mentions only merge if they share a SKU. Properties: description, item type (Implant/Capital/Disposable/Limited Use/Biologic), head style, construct, thread, color, guidewire/pre-drill/driver spec text |
| `Procedure` | name (bunion, Akin, lapidus, MTP fusion, ...) | |
| `CanonicalTerm` | canonical name | anchors both synonym and abbreviation clusters from `synonyms-map.csv` |

**Note on `Tray` vs ticket 05's System tag:** `unite-master-csv.txt`'s `System`
column has ~27 distinct values (tray/instrument-set granularity, e.g. `REFLEX®
HYBRID Implant System`, `REFLEX® MINI Nitinol Staple System`), one level below
ticket 05's `MIS`/`REFLEX` document tag. Model both levels (`Tray` under
`ProductFamily`) rather than flattening — this is what lets `Procedure
--REQUIRES--> Tray` answer "if I'm doing a hybrid MTP fusion, do I only need
the hybrid tray?" correctly. This is a possible future refinement to ticket
05's tag granularity, not something to change there now.

### Relationships

- `Part -[:BELONGS_TO_TRAY]-> Tray -[:BELONGS_TO_FAMILY]-> ProductFamily`
- `Part -[:LOCATED_IN]-> TraySection` (best-effort, may be sparse)
- `Part -[:COMPATIBLE_WITH]-> Part` — e.g. plate ↔ screw-family. Resolved via
  **exact SKU-prefix match**: a compatibility-matrix column header like `2.7mm
  Polyaxial Locking (MPSL27xx)` names a real Item-No. prefix; any screw whose
  `Item No.` matches gets the edge. Deterministic, no LLM/fuzzy matching.
- `Part -[:REQUIRES_TOOL]-> Part` — e.g. screw ↔ guidewire/driver/drill-bit.
  The `Guidewire`/`Pre-Drill Diameter`/`Driver` columns are free text with no
  SKU reference, even though matching guidewire/driver/drill-bit SKUs exist
  elsewhere in the same file — text-to-SKU resolution here is acceptable
  **only** inside the one-off master-catalog script (see below), not as a
  general policy.
- `Part -[:DIFFERENTIATES_FROM]-> Part` (+ an explanation text property) —
  answers "what's the difference between the two 1.4mm wires" / "how do I
  tell the three 26mm TETRA staples apart" style questions
- `Procedure -[:REQUIRES]-> Tray`
- `Part|Procedure -[:SOURCED_FROM]-> Document`
- `Term -[:ALIAS_OF]-> CanonicalTerm` for true many:many synonym clusters
  (`guidepin`/`guidewire`/`wire`/`pin`), `Term -[:ABBREVIATION_OF]->
  CanonicalTerm` for 1:1 abbreviation expansions (`FT`→`Full thread`) — both
  resolve identically through `synonym_resolve`, callers don't need to
  distinguish. Explicit exclusion notes in `synonyms-map.csv` (e.g. "caddy is
  NOT a synonym" of tray/system/set) are **not** modeled as a negative edge —
  just omit the edge and keep the note as a `CanonicalTerm` property for
  human audit.

### Two ingestion paths

1. **One-off master-catalog seed** — a new script (e.g.
   `backend/ingestion/seed_master_catalog.py`) parses
   `backend/evals/unite-master-csv.txt` directly and writes the authoritative
   `Part`/`Tray`/`ProductFamily` catalog plus `COMPATIBLE_WITH` and
   `REQUIRES_TOOL` edges. Deterministic parsing (SKU-prefix matching for the
   compatibility matrix, text-to-SKU resolution for tool requirements — both
   acceptable here since this is a controlled, reviewable batch job over a
   known-clean source, not general document ingestion). Run once/occasionally
   by hand, reading the CSV from disk — **not** part of the per-document
   background ingestion task, and the file is not uploaded through ticket
   04's flow.
2. **Per-document prose extraction** — `backend/ingestion/entity_extraction.py`
   (LLM-based, this is what the existing checklist bullets below describe)
   runs on brochures/surgical-technique guides/launch presentations/etc. at
   upload time. It only **attaches** facts (`DIFFERENTIATES_FROM` text,
   `SOURCED_FROM` provenance, procedure mentions) to `Part`/`Procedure` nodes
   that already exist from path 1 — it must not mint new unmerged,
   non-SKU'd `Part` nodes from a bare name mention (e.g. "the MIS
   screwdriver" with no SKU in that sentence doesn't become a node). Path 1
   should run before path 2 produces anything useful, since path 2 has
   nothing to attach to otherwise.

### Explicitly out of scope for this ticket

- **Doctype-priority ranking / cross-source confirmation.** `doctype-
  hierarchy.csv` ranks DocTypes P1–P4 per prompt-type and wants ≥2 sources
  confirmed before answering. `graph_query` just returns each fact's
  `SOURCED_FROM` document(s) with `doc_type` available on the `Document`
  node — applying the priority table and the confirmation rule is the
  calling chat workflow's job (ticket 08+), since it also has to reconcile
  against `vector_search` results from ticket 06, not something `graph_query`
  can decide alone.
- **Ordered procedural step sequences** (e.g. exact back-table setup order,
  assembly step ordering). LLM extraction of *ordered* steps from prose is
  much more failure-prone than flat entity/relationship extraction; rely on
  ticket 06's vector search for these for now. Revisit as a future ticket if
  that proves insufficient.

## Checklist

- [x] `backend/ingestion/entity_extraction.py` extracts entities,
      `DIFFERENTIATES_FROM` relationships, and provenance from a document's
      text, and only attaches to existing SKU'd `Part`/`Procedure` nodes (no
      new non-SKU'd nodes)
- [x] A new one-off script parses `backend/evals/unite-master-csv.txt` into
      the `Part`/`Tray`/`ProductFamily` catalog plus `COMPATIBLE_WITH`
      (SKU-prefix matched) and `REQUIRES_TOOL` (text-to-SKU resolved) edges
- [x] Extracted entities and edges are written to Neo4j via
      `backend/retrieval/graph_client.py`
- [x] `graph_query(entity, relationship)` returns real related entities for a
      term that exists in the graph, including at minimum `COMPATIBLE_WITH`,
      `DIFFERENTIATES_FROM`, and `REQUIRES` (procedure→tray) relationships
- [x] `synonym_resolve(term)` returns real canonical synonyms/aliases for a
      term seeded from `synonyms-map.csv` (both `ALIAS_OF` and
      `ABBREVIATION_OF` clusters), e.g. an aliased implant-system term
- [x] Manual/CLI checks against real data confirm: a compatibility query
      (e.g. `graph_query("MPPA100L", "COMPATIBLE_WITH")` against the real
      seeded master catalog returns 133 real screws) returns real graph
      data. `attach_differentiation`/the `DIFFERENTIATES_FROM` query path is
      verified end-to-end against real Neo4j (`test_graph_client.py`), but
      only with synthetic parts — verifying it against a *real* LLM-extracted
      differentiation (e.g. "difference between the two 1.4mm wires") needs
      `OPENAI_API_KEY`, which isn't configured in this environment; see the
      `needs_openai_key`-skipped tests in `test_entity_extraction.py`
- [x] This (path 2, prose extraction) runs as part of the same background
      ingestion task as ticket 06 (or a parallel step), and also gates the
      document's "done" status. The master-catalog seed (path 1) does not.
