# 21: Fix resolve_synonyms to resolve per-term, not whole-query

**What to build:** `resolve_synonyms` (`backend/agents/workflows/deterministic.py`)
currently calls `synonym_resolve.ainvoke({"term": search_query})` with the
**entire** question string. `synonym_resolve` (ticket 07) does a graph
lookup keyed on a `Term` node's exact name, so a multi-word sentence almost
never matches anything -- the deterministic workflow's own docstring already
says this ("most turns are a no-op here"). Extract candidate terms from the
query (word-level, matching `backend/evals/compare_retrieval.py`'s
`_candidate_terms` helper, which already exists as a diagnostic script for
exactly this) and resolve each one against the graph instead.

**Why this belongs on the graph-traversal list, not "build a synonym
layer":** verified directly against Neo4j while triaging the MIS eval
reflection (2026-09-03) -- **the synonym data already exists and is
already correct**:

```
"pin" -[:ALIAS_OF]-> "guidepin"
"guidewire" -[:ALIAS_OF]-> "guidepin"
"wire" -[:ALIAS_OF]-> "guidepin"
"FT" -[:ABBREVIATION_OF]-> "Full thread"
"PT" -[:ABBREVIATION_OF]-> "Partial thread"
```

This is exactly the mapping the original human feedback (`feedback-notes.csv`)
flagged as missing ("'Guidepin' vs. 'Wire' ... probably because the bot
isn't programmed to understand synonyms yet") and that the eval reflection's
original "Reconsider" bucket assumed might need new engineering. It
doesn't -- the graph has it. The gap is purely that `resolve_synonyms`
never asks it the right question.

**Blocked by:** 07 (ingestion graph leg, done -- `ALIAS_OF`/`ABBREVIATION_OF`
edges already exist and are populated)

**Status:** done (2026-09-03).

- [x] `resolve_synonyms` extracts candidate terms from `search_query` --
      `_candidate_terms` promoted out of `compare_retrieval.py` into
      `backend/agents/tools/term_extraction.py`'s `extract_candidate_terms`,
      shared by both that script and this node (no duplicated stopword
      list)
- [x] Each candidate term is resolved via `synonym_resolve`, folded into
      `hybrid_retrieve`'s `search_text` (unchanged wiring, now actually
      populated)
- [x] Bug found and fixed during implementation, not anticipated in the
      original scope: a plural term ("wires") never matches the graph's
      singular `Term` node ("wire") as an exact name, so the fix above
      alone was still a near-total no-op. Added
      `term_extraction.with_singular_variants` (a crude trailing-s strip,
      not real stemming) and applied it in both this node and ticket 22's
      `resolve_skus`, which has the identical problem matching Part
      descriptions.
- [x] Asking about "wires" in an MIS context retrieves/answers using the
      same passages a "guidepins" phrasing would (evidenced by
      `mis.jsonl`'s Q02/Q03 vs. Q04/Q10/Q07 pairs, which ask the same thing
      with different terminology)
