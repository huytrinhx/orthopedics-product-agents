# 0006 — Doc-type hierarchy is data, question types are code

## Status

Accepted

## Context

The deterministic workflow classifies every question into one or more
question types (`backend/agents/question_types.py`), then uses them twice
during retrieval. `hybrid_retrieve` hard-filters the retrieval pool to the
document types relevant to that question type (untagged chunks always get
through), and `rerank` adds a small bonus by how authoritative a chunk's
document type is for that question type. Both read a priority mapping that
lived in code (`_DOCTYPE_PRIORITY` in `deterministic.py`): a ranked list of
document type name fragments per question type, matched against real
document type names by substring.

Document types are a closed, admin-curated list edited from Settings (issue
#16), so the code-side mapping went stale without anyone seeing it. A
renamed type stopped matching its fragment and quietly dropped out of
retrieval. A newly added one wasn't reachable by any filtered question type
until a deploy. Nothing showed which document types the mapping covered.

## Decision

Split the two halves (issue #17):

- **Question types stay in code**: slug and description. The deterministic
  workflow branches on the slugs (`technique_procedural` widens SKU
  resolution, for example), and the classifier's structured-output enum is
  built from them. Changing one is a behaviour change that needs code and
  review anyway.
- **The ranked document types per question type are data**: the
  `question_type_document_types` table (`backend/doc_type_hierarchy/`),
  keyed by document type **id**, edited from the Settings page's "Question
  types → doc types hierarchy" section. Retrieval's filter and rerank bonus
  read it each turn. The substring matching is gone.
- **`unclassified` is the escape hatch.** It's always present, it can't be
  edited, and it maps to no filter (a check constraint stops it getting
  rows). Any question type with an empty list also retrieves unfiltered.
  Settings badges a document type that no question type lists as
  **unmapped**: right now it's only reachable on an unfiltered turn.
- A document type merge (#16) moves the merged type's hierarchy rows to the
  type it was merged into. Deleting a document type drops its rows.

The migration seeded the table from the old in-code mapping, using the same
substring match against the document types that existed then. It also added
`features_benefits` from the stakeholders' hierarchy CSV
(`backend/evals/doctype-hierarchy.csv`). That CSV's "Master Item File" tier
isn't a document type (it's the graph), so it was skipped.

## Consequences

- Wiring a new or renamed document type into retrieval is a Settings edit,
  with no deploy. Renaming never breaks the mapping, because the mapping
  is keyed by id.
- Adding, renaming, or removing a *question type* still needs a code change.
  A stale slug left in the table matches nothing the classifier can emit,
  so it's inert.
- Retrieval does one more small query per turn (the hierarchy read in
  `hybrid_retrieve`; `rerank` gets the bonuses through graph state).
- The seeded filter matches the old one exactly. The rerank order is
  preserved too, but the bonus sizes come out identical only when each old
  name fragment matched exactly one document type (tested). That's because
  the bonus now scales with the number of ranked document types, not the
  number of fragments: a fragment that matched nothing no longer takes up a
  rank, and two types that matched the same fragment now get consecutive
  ranks instead of tying.
