# Orthopedics Product Agents — Product

## What it's for

A rep, surgeon, or internal user asks a question about a specific
orthopedic implant/instrument system — specs, SKU/ordering info, surgical
technique, how to tell similar parts apart — and gets a cited answer drawn
from that system's real documentation (inventory control forms,
brochures, surgical technique guides), not a generic or invented one.

This is inferred from the golden eval datasets
(`backend/evals/golden_datasets/`) and `feedback-notes.csv`, since the
retrieval/agent logic itself is still stubbed out (see Status in
`README.md`) — treat this section as the best current evidence of intent,
not a finalized spec, and update it once real behavior lands.

## What the golden datasets show

- **Per-system routing.** Queries are scoped to one product system at a
  time (e.g. `MIS`, `REFLEX` today — see `intent_detection.jsonl`). Intent
  detection picks the system before retrieval runs, and can flag
  `expects_clarification: true` rather than guessing when it's ambiguous.
- **Every answer traces to real source material.** `expected_citations` in
  `mis.jsonl` / `reflex.jsonl` point at specific documents (e.g. "MIS
  Inventory Control Form", "REFLEX TETRA" brochure) — answers aren't
  free-associated.
- **Question types are varied but system-specific**: spec lookups
  (dimensions, thread/length options), SKU/ordering info, part
  differentiation ("how do I tell X apart from Y"), and surgical
  workflow/setup questions.
- **Feedback loop.** `feedback-notes.csv` (human-reviewed prompt / provided
  answer / preferred answer / content+formatting feedback) is the source
  of truth that `build_dataset.py` turns into the JSONL golden sets — new
  product-system coverage or corrections start there, not by hand-editing
  JSONL.

## Open / not yet decided

Everything below is genuinely undecided, not a documented guardrail — don't
treat these as settled:

- Whether out-of-scope or unsupported-system questions get an explicit
  "not covered" response or something else.
- Whether recommendations enforce a whitelist of real systems/parts the way
  a mature version of this product might (analogous to, but not copied
  from, patterns seen in adjacent retrieval products).
- Any account/auth model, guest-vs-persisted-history distinction, or
  redaction requirement — none of this exists in the code yet.

See `agents.md` for the technical (not product) standing decisions.
