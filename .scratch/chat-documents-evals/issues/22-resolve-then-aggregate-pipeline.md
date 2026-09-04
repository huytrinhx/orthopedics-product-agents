# 22: Wire a resolve-then-aggregate pipeline into `deterministic`

**What to build:** Extend `backend/agents/workflows/deterministic.py` with a
fixed (not agentic) two-phase pipeline for spec/SKU/pairing-shaped answers,
branching by `resolved_question_type`:

```
detect_intent (system + question_type, unchanged)
  -> question_type == "Technique/procedural":
       retrieve_procedure (vector_search over the system's technique doc,
       same as today's hybrid_retrieve) -> extract_terms (from the
       retrieved passage text)
  -> otherwise (spec/SKU/pairing question types):
       extract_terms (from the raw query directly)
  -> resolve_skus (ticket 21's per-term extraction + ticket 20's graph
     Part-node lookup, same mechanism for both paths above -- only the
     *input text* differs)
  -> aggregate_facts (pull every matched Part's properties -- sku,
     description, thread, guidewire_spec, driver_spec, pre_drill_spec,
     head_style, construct -- into generation context)
  -> generate (unchanged node, now grounded in aggregated Part facts
     alongside/instead of raw vector prose)
```

This is the design worked out in the 2026-09-03 grilling session on
whether to build `react_agent.py`: the actual need described (narrow to
system + SKU(s), pull every fact about those SKUs, aggregate into context;
the procedural path reaches the same goal via a different first step) is
fixed control flow with a type branch, not a case where the model needs to
decide what to do next. See ticket 23 for the *other* half of that
decision -- a true agentic workflow, kept deliberately separate so its
result isn't confounded with this one.

**Design decisions already made (grilling session, don't re-litigate):**
- **Multiple candidate SKUs are not disambiguated up front.** When
  resolution turns up more than one plausible Part (e.g. "the 1.4mm wire"
  legitimately matches two real Foot Recon guidepins), pass all matched
  candidates' facts through to `generate`, labeled, and let it decide what
  the question actually needs. Do not `interrupt()` on this -- that would
  fire constantly for ordinary questions, unlike `detect_intent`'s system
  ambiguity, which is genuinely rare.
- **The procedural path does not use graph `Procedure` nodes.** Checked
  directly against the real graph DB: the mechanism to write them exists
  (`graph_client.py`'s `attach_procedure`) but zero nodes exist, because
  (a) the ingested surgical-technique document narrates steps ("Insert a
  burr through the percutaneous hole...") rather than "Procedure X
  requires Tray Y" sentences, and (b) even if it did, the document's
  instrument groupings ("MIS Bunion Instruments") don't match the graph's
  whole-system-level `Tray` node names. Fixing that is a real, separate
  project (defining what a procedure is as structured data, re-ingesting
  against that shape) -- not a silent prerequisite here. The procedural
  path stays vector-search-first, same as `deterministic` already does.
- **Term extraction is one mechanism, two input sources.** Reuse ticket
  21's extract-then-resolve logic for both paths -- the spec/SKU path
  feeds it the raw query, the procedural path feeds it retrieved passage
  text. Building separate extraction logic per path is exactly the kind of
  premature special-casing that left `resolve_synonyms` broken before
  ticket 21.

**Blocked by:** 20 (graph-grounded Part lookup), 21 (per-term synonym
resolution) -- this ticket is the pipeline that wires both of them
together into a real answer path; building it before either lands means
`resolve_skus` has nothing working to call.

**Status:** done (2026-09-03), verified by hand (ticket 13's harness still
doesn't exist) against three full re-runs of `mis.jsonl` during
implementation -- final tally moved from 2 good / 6 partial / 3 miss
(pre-ticket-20/21/22 baseline) to 7 good / 3 partial / 1 miss. Full
before/after evidence, per-question verdicts, and the pipeline traces are
in `artifacts/evals-deterministic.html`.

- [x] `extract_terms` is one shared function
      (`term_extraction.extract_candidate_terms` +
      `with_singular_variants`), called with either the raw query
      (spec/SKU path) or reranked passage text (procedural path) inside
      `resolve_skus` -- not a separate node, since both paths converge
      immediately into the same resolution call
- [x] `resolve_skus` resolves each extracted term (plus anything
      `resolve_synonyms` already resolved this turn) to candidate `Part`
      nodes via ticket 20's `find_parts`/`part_lookup`
- [x] `aggregate_facts` formats every resolved Part's properties into a
      "Known catalog facts" block in `generate`'s context, alongside (not
      replacing) the vector-retrieved passages
- [x] Multiple candidate SKUs are passed through to `generate` labeled, not
      collapsed or routed through `interrupt()` -- verified: Q06 ("part
      number for the MIS 1.4 wire") correctly surfaces both real 1.4mm
      guidepins rather than guessing one
- [x] Re-ran `mis.jsonl`'s Q02, Q04, Q06, Q07, Q08, Q10 -- all improved,
      several (Q02, Q10) went from wrong/incomplete to fully correct. Q05
      did **not** improve: still open, see below.
- [x] Verified by hand via a trace-instrumented runner (same pattern as the
      artifact's Section 05), not by trusting self-eval judge scores alone
      -- good thing too, since a real bug in `self_eval` itself was found
      this way (below)

**Bugs found and fixed during implementation** (none anticipated in the
original scope -- direct testing at each step caught all of them before
they shipped silently):
- `find_parts` returned `{"part": {...}, "product_family": ...}` (nested);
  `resolve_skus`/`_format_part` expected a flat dict with `sku` at the top
  level, so every match was silently dropped (`part.get("sku")` always
  `None`). Fixed by flattening in `find_parts` itself.
- `detect_intent`'s classifier occasionally put the literal string
  `"null"` *into* the `system` string field instead of leaving it unset,
  which then propagated as a bogus `ProductFamily` scope
  (`_product_family_for_system("null")` -> `"null"`, matching zero real
  families) and silently zeroed out `resolve_skus` for the whole turn.
  Normalized in `detect_intent` right after the classifier call.
- `self_eval` judged the draft answer against `reranked` passages only,
  never `aggregated_facts` -- so a claim correctly grounded in the graph
  but absent from (or contradicting) a vector passage read as "unfaithful"
  to a judge that never saw the graph data, penalizing exactly the fix
  ticket 20 exists to make. Now passes a synthesized `catalog-facts`
  passage alongside `reranked`.

**Still open, not fixed here -- a real, diagnosed gap:** Q05's back-table
answer still names Partial-Thread screw SKUs for what should be a
Full-Thread bunion screw. Root cause confirmed via trace: the procedural
path's `extract_terms` only reads `reranked` passage text, and the
Full-Thread screw inventory chunk isn't reliably in that top-5 -- so
`resolve_skus` never sees an `MSCF...` token to resolve, regardless of how
good the resolution mechanism is. A likely fix (not implemented): also
extract terms from the raw query for the procedural path (it literally
says "a 4.0 and 3.5 screw for their bunion") rather than passage text
alone. Worth a small follow-up, not a new ticket on its own.
      (see that artifact's Finding 05 for why the judge alone isn't
      trustworthy yet)
