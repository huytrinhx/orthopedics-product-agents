# 24: Graph-first query enrichment + doctype-priority reranking

**What to build:** Requested directly (2026-09-04), not pre-scoped via a
grilling session -- captured here after the fact for traceability, matching
this session's convention for 20-23.

Two related refinements to `deterministic`'s retrieval leg:

1. **Resolve graph entities before vector search, not just after.**
   `resolve_skus` (ticket 22) only ever ran *after* `hybrid_retrieve`/
   `rerank`, so the vector query itself was still just the rep's raw
   wording -- "wire" stayed "wire" in the search text even after the graph
   correctly resolved it to real guidepin SKUs. A new node,
   `resolve_query_entities`, runs the same graph lookup immediately after
   `resolve_synonyms` (before any retrieval), and its result both enriches
   `hybrid_retrieve`'s search text with the resolved parts' own
   descriptions (closer to actual inventory-table chunk wording than the
   rep's phrasing ever is) and is reused, not recomputed, by `resolve_skus`
   for the spec/SKU path.
2. **Doctype-hierarchy-aware reranking.** `doctype-hierarchy.csv` (Master
   Item File > Inventory Control Form > Tray Overhead Guide > Brochure/
   Surgical Technique, for Spec questions; Surgical Technique > Launch
   Presentation, for Procedural) has existed in the repo since Phase 2's
   scaffold and was **never wired into any code** -- confirmed by grep
   before starting this ticket. `rerank` now adds a small additive bonus
   per candidate passage based on its `document_type` and
   `resolved_question_type`, using this priority order (Priority 1,
   "Master Item File", isn't a document at all -- it's the graph, already
   covered by ticket 20/22 -- so this ranks the remaining P2-P4 tiers as a
   tiebreak, not an override of genuine semantic relevance).

**Why now:** directly motivated by the still-open typo-propagation finding
(Q04/Q10's "50mm" vs. the correct 150mm) from the eval reflections --
`rerank` had no way to prefer the structured Inventory Control Form data
over the OCR-damaged PDF chunk it was tied with on pure semantic
similarity.

**Blocked by:** 20 (graph-grounded Part lookup), 22 (resolve-then-aggregate
pipeline) -- extends both directly.

**Status:** done (2026-09-04), verified against the live DB (not just unit
tests) -- re-running Q10 twice after this landed shows Inventory Control
chunks now dominating the reranked list (combined scores ~1.0/0.2/0.2 vs.
Surgical Technique's ~0.05/0.05) and the answer correctly stating 150mm/
200mm with correct PT/FT labels, matching the graph's `Part` record, on
both runs.

- [x] `resolve_query_entities` resolves the raw query to candidate Part
      nodes before `hybrid_retrieve` runs
- [x] `hybrid_retrieve`'s search text is enriched with resolved parts'
      descriptions (capped at 5, to sharpen rather than flood the query)
- [x] `resolve_skus` reuses `resolve_query_entities`'s result for the
      spec/SKU path instead of re-querying the graph -- net effect: the
      common case now costs the *same or fewer* Neo4j round-trips than
      before this ticket, not more, despite the new node
- [x] `chunks`' `document_type` is joined into `hybrid_search`'s results
      (`retrieval/vector_store.py`) rather than requiring a separate
      per-chunk lookup
- [x] `rerank` applies a doctype-priority bonus keyed by
      `resolved_question_type`, and the `reranked` passages' own `score`
      field reflects the combined (relevance + priority) value actually
      used to rank them, not the stale pre-rerank RRF score
- [x] `RetrievedPassage` (shared state schema) carries `document_type`;
      all three construction sites (`deterministic.hybrid_retrieve`,
      `deterministic.self_eval`'s synthetic catalog-facts passage,
      `react_agent._passages_from_scratchpad`'s synthetic tool-result
      passages) updated for consistency

**Explicitly out of scope, not done here:** `react_agent` gets none of
this -- it doesn't have a fixed `hybrid_retrieve`/`rerank` pair to modify,
and its own tool-calling already does something organically similar
(calling `part_lookup` before `vector_search` when it chooses to). Also
out of scope: actually parsing `doctype-hierarchy.csv` at runtime -- its
priority order is hand-encoded as a Python dict (`_DOCTYPE_PRIORITY`)
matching the CSV's intent in this system's real `document_type` names,
not read from the file itself.
