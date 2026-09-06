# 25: Tray-layout visual ingestion (Part -[:LOCATED_IN]-> TraySection)

**What to build:** Populate the graph's already-scaffolded-but-unused `Part -[:LOCATED_IN]-> TraySection` relationship (ticket 07: "shape only; unpopulated until tray-overhead-guide extraction exists") by extracting real tray-region assignments from the tray photos embedded in ingested documents -- a rep asking "what level/side is the 4.0mm screw on" currently has no data-backed answer at all.

Requested directly (2026-09-05), scoped via a grilling session -- captured here for traceability, matching this session's convention for tickets not pre-scoped in the original 17.

**Blocked by:** 07 (Ingestion graph leg, done -- this extends its deliberately-deferred `LOCATED_IN` piece), 20 (graph-grounded Part lookup, done -- this needs the same "known parts for this system" pattern).

**Status:** ready-for-agent

## Ground truth (found live, 2026-09-05)

Rendered a real page from `Medline Unite MIS Foot Recon System - Customer Presentation.pdf` (a `Launch Presentation`-tagged document) to look at an actual tray photo before scoping this. Confirmed:
- A tray photo splits into labeled regions (e.g. "Top Level" / "Bottom Level" × "Left Side" / "Middle" / "Right Side"), with caption bullets alongside naming categories in prose ("Screw Prep Instruments (Middle)").
- The caption text is already sitting in `ingestion/text_extraction.py`'s output today (pdfplumber's text layer) -- no new extraction needed to get it.
- The photo itself carries *additional* information the caption text doesn't repeat: finer per-column labels printed directly on the tray (exact diameter+thread codes like "3.0mm FT" vs "3.5mm PT"), which is what actually distinguishes which specific known SKUs sit in a region. This is the concrete reason vision is worth the cost here, not just parsing the caption text.
- `get_chat_model()` already defaults to `gpt-4o`, which is multimodal -- no new provider/model config needed, just a differently-shaped message (image content, not just text).

## Design (settled via grilling)

1. **No document-type gating.** `document_types` are free-text admin tags (ticket 05), not a fixed enum -- a user can name one anything, so trusting an exact tag string (even a list of "the usual suspects": Surgical Technique, Setup Guide, Launch Presentation, Tray Layout) is fragile and would silently skip a mistagged or oddly-named document. Instead, a cheap **content heuristic** runs on every ingested document regardless of its tag:
   - Signal A: the page has multiple large embedded images (`pdfplumber` `page.images`, already available at no extra parsing cost; size above a threshold).
   - Signal B: the page's already-extracted text contains a tray/level keyword ("Top Level", "Bottom Level", "Tray Design", "Left Side", "Right Side", or similar -- tune against the real example above).
   - Only pages passing **both** signals go to vision. Neither signal costs an LLM call; the combination was true on the one real example checked and should stay highly precise against false positives (a pure inventory-table page, for instance, has neither).
2. **Extraction stays coarse**, matching ticket 07's original scope exactly -- a tray name + level + region label (e.g. "Top Level", "Left Side"), not pixel coordinates or bounding boxes. Vision's job is reading the region's own printed labels well enough to correctly sort which known SKUs belong there, not producing a spatial map.
3. **Ingestion-only scope for this ticket.** Populate `LOCATED_IN` and verify it's correct against real re-indexed documents. Wiring it into `aggregate_facts`/chat answers (e.g. "what level is the 4.0mm screw on") is a small, separate follow-up once there's real data to point at -- don't bundle the retrieval-side change into the same ticket.
4. **New module**: `ingestion/tray_layout_extraction.py`, sibling to `entity_extraction.py` (graph leg) and `chunking.py`/`embedding.py` (vector leg) -- a third independent ingestion leg, triggered from the same place `ingest_document` already is (`documents/service.py`, on upload/index/re-tag). Reuses `entity_extraction.py`'s established convention: given the real "known parts for this system" list (`graph_client.list_parts_for_family`), the vision call may only reference SKUs from that list -- anything it names that isn't a real known SKU is dropped, never merged as a fuzzy match or a new node.
5. **`TraySection.key` is scoped through the tray name**: `"{tray_name}::{level}::{region}"` (e.g. `"MIS Bunion Instruments Tray::Top Level::Right Side"`) -- the existing constraint (`tray_section_key`, ticket 07) is a *global* uniqueness constraint on `key`, so an unscoped label like `"Top Level"` alone would collide across different systems' trays.
6. **Idempotent on re-ingestion**: re-indexing a document deletes that tray's existing `LOCATED_IN` edges before writing new ones (matching `seed_master_catalog.py`'s delete-then-insert convention), so re-running after a fix self-corrects instead of leaving two disagreeing answers for the same slot.
7. **Capped cost**: a hard limit of 20 heuristic-passing pages processed per document (mirrors `MAX_TERMS_PER_TURN`-style safety caps elsewhere in this codebase), logged as a warning if hit rather than failing the document -- generous against every real document seen so far (one tray-photo page per document, typically), a guard against a malformed/unusually large PDF, not a real expected case.

## Acceptance criteria

- [ ] `ingestion/tray_layout_extraction.py`: a page-selection function implementing the two-signal heuristic (large embedded images + tray/level keyword in extracted text) over a document's already-extracted per-page text/images.
- [ ] A vision extraction call (via `get_chat_model()`, image + nearby caption text + the system's known-parts list as input) that returns candidate `{region_label, skus: [...]}` groupings, validated against the known-parts list before writing anything (hallucinated/unknown SKUs dropped, matching `entity_extraction.py`'s convention).
- [ ] `retrieval/graph_client.py` gains a method to write `Part -[:LOCATED_IN]-> TraySection` edges (MERGE on `TraySection.key`, delete-then-recreate that tray's edges per re-ingestion run) -- `TraySection` nodes carry at least `key`, `tray` (name), `level`, and `region`.
- [ ] Wired into the existing ingestion trigger point (`documents/service.py`) as a third leg alongside the vector/graph legs, capped at 20 heuristic-passing pages per document, best-effort (a document with no tray photo at all is a silent no-op, not a failure).
- [ ] Verified live against a real re-indexed document (the MIS Foot Recon `Customer Presentation.pdf` / `System Overview and Surgical Technique.pdf` are the known real examples) -- direct Neo4j query confirming real `LOCATED_IN` edges landed with sensible SKU-to-region assignments, cross-checked by eye against the actual tray photo.

**Explicitly out of scope:** using the extracted `LOCATED_IN` data in chat answers (a follow-up ticket once this data exists and is verified); pixel/bounding-box-level position; any change to how `document_types` tags work.
