"""Canonical question-type taxonomy: what kind of thing a rep's question is
asking about (a spec lookup, a procedure walkthrough, a compatibility
check, ...). Shared across any workflow that classifies a question this way
-- currently only `agents/workflows/deterministic.py`'s `detect_intent` --
so a future workflow reuses this list instead of carrying its own drifting
copy.

Canonicalized 2026-09-05 into short, stable slugs. The original source
(`evals/golden_datasets/feedback-notes.csv`'s free-text "Question Type"
column, carried through by `evals/golden_datasets/build_dataset.py`) was
never an enum -- rows read like "Specs - system contents; SKU/ordering
info" -- which is why `deterministic.py` used to carry a whole fuzzy
case/whitespace-matching normalizer just to coerce a classifier's echoed-
back string onto one of four hand-typed values. With short slugs enforced
directly by the classifier's structured-output schema (a `Literal` built
from `QUESTION_TYPE_NAMES`), that normalizer is redundant and has been
removed.

`description` is real input to the classifier's prompt (not just a
maintainer comment) -- a bare slug like "compatibility_lookup" doesn't
self-explain the way the old verbose CSV wording did, so losing the
description at classification time would likely hurt accuracy.

NOT YET RECONCILED: `evals/golden_datasets/build_dataset.py`'s
`expected_question_type` field still carries the CSV's raw wording
("Technique/procedural", "Pull resource", ...), not these slugs, and the
CSV has no source column for the new `compatibility_lookup` type at all.
Nothing currently scores against `expected_question_type` (`evals/
harness.py` doesn't read `question_type`), so this is a latent mismatch,
not an active bug -- but whoever builds that scoring needs a translation
step (and a decision on what backfills `compatibility_lookup`) first.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class QuestionType:
    name: str
    description: str


QUESTION_TYPES: list[QuestionType] = [
    QuestionType("product_characteristics", "materials, dimensions, thread types, color, etc."),
    QuestionType("system_contents", "what's included in a set/system, SKUs, levels in the tray"),
    QuestionType(
        "technique_procedural",
        "step-by-step surgical technique, procedural walkthrough, applications of products or "
        "parts, etc.",
    ),
    QuestionType("documents_lookup", "fast pull of a document by name, SKU, or other identifier"),
    QuestionType(
        "compatibility_lookup",
        "which products/parts are compatible with which others, e.g. screws with plates, "
        "implants with instruments, etc.",
    ),
]

QUESTION_TYPE_NAMES: list[str] = [qt.name for qt in QUESTION_TYPES]
