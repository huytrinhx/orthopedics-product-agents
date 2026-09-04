# 20: Ground structured spec/SKU/pairing facts in the graph, not vector prose alone

**What to build:** For a question whose answer is really "look up a specific
Part's properties" (a SKU, a thread type, a guidewire/driver pairing, a
style descriptor) rather than "synthesize prose from retrieved passages",
`hybrid_retrieve`/`generate` (`backend/agents/workflows/deterministic.py`)
should ground the answer in a direct Neo4j `Part` node lookup (via
`backend/agents/tools/graph_query.py`, extended if needed) instead of
relying solely on `vector_search` + rerank + free-text generation.

**Why this, not a bigger "wire in the ReAct/graph leg" project:** verified
directly against the real graph DB (`docker exec orthopedics-neo4j
cypher-shell`) while triaging the MIS eval reflection
(2026-09-03) -- this is cheaper and more precise than it first looked from
the eval alone:

- `COMPATIBLE_WITH` edges (30,567 of them) exist for the UNITE plating
  systems, but **zero** for MIS parts -- the pairing data for MIS lives as
  plain properties directly on each screw's own `Part` node
  (`guidewire_spec`, `driver_spec`, `pre_drill_spec`, `head_style`,
  `construct`, `thread`, `sku`), e.g. `MSCF3526` ("MIS HV CHAMFER FT, 3.5 X
  26", `thread: "Full"`) carries `guidewire_spec: "Ø1.4 x 200mm (Cobalt
  Chrome)"` directly. So this is a single-node property read for MIS, not a
  relationship traversal -- simpler than "route through graph_query's
  COMPATIBLE_WITH" sounded in the original matrix.
- This one fix resolves several eval findings at once, all traceable to the
  same root cause (a structured fact only available in prose the vector
  leg didn't reliably surface or the generation step didn't reliably read
  correctly):
  - **Q05** (back-table setup): the generated answer labeled the 4.0mm/3.5mm
    MIS bunion screws "Partial Thread" and paired them with 150mm
    guidewires; the graph says `thread: "Full"` and `guidewire_spec: "Ø1.6
    x 200mm"` / `"Ø1.4 x 200mm"` respectively -- a direct property read
    would have gotten this right deterministically.
  - **Q04/Q10** ("50mm" vs. 150mm guidepin length): the vector-retrieved
    chunk (`0bb5d399…#19`, from the Inventory Control Form PDF) has a
    genuine OCR/extraction artifact -- `"MGT14150 Ø1.4 x 50mm Guidepin"` --
    confirmed by reading the chunk directly from Postgres. The graph's
    `Part` node for `MGT14150` (built from structured data, not PDF OCR)
    correctly has `"GUIDEPIN, MIS 3.5 PT, 1.4 X 150MM"`. Cross-checking
    against the graph before answering would have caught this.
  - **Q06/Q07** (SKU lookup flipped between miss and correct across two eval
    runs, depending on vector rerank luck): a direct `Part` lookup by
    resolved description/SKU is deterministic where vector rerank isn't.
  - **Fill-in: style descriptors** (Q01 dropped "headless, chamfered,
    cannulated"): `head_style`/`construct` are already Part properties --
    free once a question resolves to a Part.
  - **Fill-in: redirect to real neighboring SKUs** (Q08, asked about a
    nonexistent 1.5mm wire): a Part query for the same description family
    near the asked-for diameter would let generation proactively list the
    real 1.2/1.4/1.6mm options with SKUs instead of stopping at "not
    found".

**Blocked by:** 07 (ingestion graph leg, done -- `graph_query`/`Part` nodes
already exist and are populated for MIS)

**Status:** done (2026-09-03), implemented as part of ticket 22's
resolve_skus/aggregate_facts nodes -- see that ticket for the shared
mechanism and the 2026-09-03 eval reflection artifact
(`artifacts/evals-deterministic.html`) for verified before/after evidence.

- [x] A way to resolve a question (or its reranked passages) to one or more
      candidate `Part` nodes -- `backend/retrieval/graph_client.py`'s
      `find_parts` (SKU-exact + description-CONTAINS, scoped to
      ProductFamily), wrapped as `backend/agents/tools/part_lookup.py`
- [x] `generate` is grounded in those Part properties via a new "Known
      catalog facts" context block (`aggregate_facts` node), with the
      vector leg staying the source for prose/context
- [x] Re-running `mis.jsonl`'s Q04, Q06, Q07, Q10 shows the graph-backed
      answer overriding the vector-only one where they disagreed -- Q10's
      "50mm" (an OCR artifact in the vector chunk) now correctly reads
      150mm, matching the graph's `Part` record. **Q05 still doesn't**:
      the procedural path's term extraction only sees `reranked` passage
      text, and the Full-Thread screw inventory chunk isn't always in that
      top-5 -- a real, diagnosed gap, not yet fixed. See ticket 22's
      status note.
- [x] Q08-shaped "this exact SKU doesn't exist" questions get a real
      redirect (nearby SKUs, from the graph) instead of a bare "not found"
      -- verified: now lists real 1.2mm/1.6mm alternatives with SKUs
