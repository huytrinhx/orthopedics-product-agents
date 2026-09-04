# 23: Build `react_agent.py` as a true agentic workflow

**What to build:** Actually implement `backend/agents/workflows/react_agent.py`
(currently `raise NotImplementedError`) as a real ReAct-style loop: the
model chooses which tool to call (`vector_search`, `graph_query`,
`synonym_resolve`) and when it has enough to answer, rather than following
a fixed node sequence. Bind all three tools, give it a step budget, and let
it iterate.

**Why this is its own ticket, not folded into ticket 22:** decided in the
2026-09-03 grilling session on this exact question. Ticket 22 builds fixed
control flow (a type branch, not model-decided steps) because the concrete
need described there doesn't require an agent deciding what to do next.
This ticket exists to test a different, real question: does letting the
model choose its own tool sequence add anything *beyond* what better
resolution logic already gets you? Building both and comparing them is the
only way to answer that -- building only one would leave "agentic choice
helped" and "the tools/data got better" permanently confounded.

**Design decisions already made (grilling session):**
- **Evaluate manually first, not gated on ticket 13.** No automated
  judge/harness exists yet, and that's fine for this pass -- score it the
  same way `deterministic` was evaluated (a trace-instrumented run over
  `mis.jsonl`/`reflex.jsonl`, read by hand, published as an artifact), not
  a reason to wait.
- **Build on today's tools, not gated on tickets 20/21.** This is
  deliberate, not an oversight: running `react_agent` against the same
  imperfect `vector_search`/`graph_query`/`synonym_resolve` that
  `deterministic` has known bugs against is itself informative -- if
  agentic tool choice naturally routes around some of those bugs (e.g. by
  re-querying differently instead of committing to one bad rerank), that's
  a real signal about the architecture's value independent of the data
  fixes. Note explicitly when reporting results: a win here might mean
  "agentic choice helps," or might just mean "got a lucky retrieval draw,"
  the same way one `deterministic` eval run did -- don't over-claim from a
  single pass.
- **"Raw power," not production-shaped.** Give it a generous step budget
  (6-8 tool calls) rather than tuning it down to `deterministic`'s
  `MAX_RETRIEVAL_LOOPS=2` preemptively. The point of this ticket is to find
  the capability ceiling, not to ship a cost-optimized default -- that's a
  later decision, made after there's something to compare against.
- **Reuse the clarification mechanism**, not reinvent it: `detect_intent`'s
  `interrupt()`/resume pattern (ticket 09) already works and is
  checkpointer-agnostic; there's no reason this workflow needs its own
  disambiguation mechanism.

**Blocked by:** 08 (chat baseline workflow, for the registry/checkpointer
plumbing this shares), 09 (interrupt/resume mechanism to reuse). Explicitly
**not** blocked by 20, 21, or 22 -- see above.

**Status:** done (2026-09-03). Full results, findings, and a direct
side-by-side comparison against `deterministic`:
`artifacts/evals-react-agent.html`.

- [x] `build_graph` binds `vector_search`, `part_lookup`, `graph_query`, and
      `synonym_resolve` (added `part_lookup` beyond the original scope --
      it's the tool that actually returns useful MIS data; `graph_query`'s
      `COMPATIBLE_WITH` traversal is empty for MIS, per ticket 20) with an
      8-tool-call budget, no fixed node sequence
- [x] Reuses `detect_intent`'s `interrupt()` pattern directly (the same
      function, imported from `deterministic.py`) rather than reimplementing
      it
- [x] `register("react_agent", build_graph, functional=True)` -- the admin
      picker's "Coming soon" badge is gone; 3 tests that asserted the old
      "only deterministic is functional" reality were updated
- [ ] **Not done as scoped:** the eval runner stayed a workflow-specific
      throwaway script (now 3 of them across this reflection, not
      generalized to a `--workflow` argument). Worth doing before a fourth
      workflow needs evaluating.
- [x] A full `mis.jsonl` run published as an eval reflection artifact,
      including a side-by-side verdict table against `deterministic`

**Real bugs found running this for real, fixed at the tool layer (not the
agent's prompt) so `deterministic` benefits too:**
- `find_parts`'s `product_family` required an exact Neo4j ProductFamily
  name; the agent naturally passed `detect_intent`'s more specific
  Postgres system name (`"MIS - Foot Recon"`) instead, matching zero real
  families. Normalized inside `find_parts` itself.
- `find_parts` only matched a search term as one contiguous substring; the
  agent composed natural multi-word phrases ("1.4 guidepin") that never
  appear contiguously in a compound catalog description. Added a
  same-word fallback.
- `synonym_resolve` never singularized its input -- `deterministic.py`'s
  `resolve_synonyms` node did this before calling the tool, but the tool
  itself didn't, so the agent's raw `synonym_resolve("guidepins")` call
  returned nothing where `"guidepin"` resolves fine. Fixed in the tool.
- `judge_answer`'s citation-axis instructions never mentioned the
  inline-SKU citation convention both workflows' own prompts established,
  so any answer built entirely from `part_lookup` (correct, by design)
  scored a hard 0 on citation. Fixed in `judge.py`.

These four fixes took the measured result from 3 good / 4 partial / 4 miss
(first pass) to 8 good / 2 partial / 1 miss (second pass) -- see the
artifact for the full before/after.
