# 27: `supervisor.py` multi-agent workflow -- considered, deferred

**What was considered:** replacing `supervisor.py`'s original stub premise
(a router dispatching to domain specialists like "clinical-guidelines" vs
"product-catalog" -- a split that was never grounded in this codebase; there
is one knowledge domain and four shared tools) with a role-based design:

- **Inventory specialist** -- `part_lookup` + `synonym_resolve` (+
  `graph_query`, contested), grounded strictly in catalog-fact tools
- **Surgeon specialist** -- `vector_search` only, framed around foot
  anatomy/procedure expertise
- **Synthesizer** -- merges both specialists' context into one draft answer
- **Product Manager** -- checks the synthesizer's answer's citation claims
  against the retrieved context, potentially as a hard retry gate (like
  `deterministic`'s `self_eval` -> `reformulate` loop) rather than
  scoring-only

Motivation given: "clear roles and boundaries would dramatically scale to
additional product systems and provide more accurate answers."

**Why deferred (2026-09-04 grilling session):**

1. **The scaling claim doesn't hold up.** Inventory-vs-Surgeon is a split by
   *question framing* (catalog fact vs. clinical/procedural), orthogonal to
   product *system* (MIS, REFLEX, future systems). `detect_intent`'s
   existing system classification already handles the system axis today,
   in both other workflows, for free. Adding a system doesn't add a role
   under this design -- so "scales to more systems" isn't actually what this
   split buys.

2. **The concrete failure evidence doesn't support the core hypothesis.**
   Checked directly against `artifacts/evals-react-agent.html` (the only
   real failure data that exists) rather than reasoning abstractly:
   - The 1 miss (Q03, "wires: top vs. bottom of tray") is a same-agent
     routing-consistency problem -- Q04 asks the identical underlying fact
     ("guidepins: top vs. bottom") and got it right; Q03's different
     phrasing sent the same agent down a different, confused tool path. An
     Inventory/Surgeon split doesn't address phrasing-dependent routing
     within a single framing.
   - The richest partial (Q05, "back table setup") still mislabels a screw's
     thread type (says "Compression PT," the graph record says
     `thread: Full`) -- and **`deterministic` makes the identical mistake,
     on the identical question, through a completely different
     architecture and tool sequence** (see ticket 22's own "still open, not
     fixed here" note: `resolve_skus`'s procedural path only extracts terms
     from reranked passage text, missing the raw query's explicit "4.0 and
     3.5 screw" mention). That's a diagnosed tool/data-layer grounding gap,
     not a routing or reasoning-architecture one -- two unrelated
     architectures already hit it identically, so a third architecture
     doesn't have a principled reason to expect a different outcome.
   - Neither failure on record looks like "needed an inventory-fact framing
     and a clinical/anatomy framing simultaneously and only got one" -- the
     specific gap this design was built to close doesn't show up in the
     data that exists to check it against.

3. **The one piece with real evidentiary support is separable.** A
   citation-verifying Product Manager that actually blocks and forces a
   retry (not scoring-only, which would just be `judge_answer` renamed) is
   plausibly useful independent of any multi-specialist split -- Q05's
   thread-type claim is exactly the shape of unsupported claim such a gate
   would catch. If this gets built, it doesn't need Inventory/Surgeon as a
   prerequisite.

**Not resolved, left open for whenever this is revisited:**
- Hard vs. soft tool boundary between specialists (if a split is ever
  built, a soft boundary -- same tools, different persona prompt -- is
  architecturally indistinguishable from `react_agent` run twice, and
  wouldn't test anything `react_agent`'s own eval hasn't already covered)
- Always-parallel dispatch vs. gated on `detect_intent`'s `question_type`
  (compound multi-type questions are 1 of 30 across
  `mis.jsonl`/`reflex.jsonl` -- gating, not always-parallel, matches the
  actual data)
- PM retry-loop shape and budget (mirror `deterministic`'s
  `MAX_RETRIEVAL_LOOPS`, or something new)

**Revisit when:** multi-agent orchestration is needed for a reason this
session's evidence doesn't yet show -- e.g. a real second knowledge domain
appears (not just a reframing of the existing one), or a future eval run
turns up a failure pattern that actually looks like "needed two
irreconcilable framings and only got one," which nothing on record today
does.

**Status:** deferred, not built. `supervisor.py` stays a registered,
non-functional stub (`functional=False`) -- see its own docstring for a
pointer back to this ticket.
