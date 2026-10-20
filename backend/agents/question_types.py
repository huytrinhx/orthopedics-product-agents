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

The golden datasets use these slugs too: `evals/golden_datasets/
build_dataset.py` translates the CSV's raw wording ("Technique/
procedural", "Pull resource", ...) to them. No CSV row is labelled
`compatibility_lookup` or `features_benefits`, so nothing in the datasets
expects those two yet.

Slugs and descriptions are code, but which document types each one
retrieves from is data (issue #17, docs/adr/0006-doc-type-hierarchy-is-
data.md): the ranked list per question type lives in the
`question_type_document_types` table (backend/doc_type_hierarchy/),
edited from the Settings page. The deterministic workflow branches on
these slugs and the classifier's structured-output enum is built from
them, so adding or renaming one is a code change; wiring a new doc type
into one is not.

`unclassified` is the escape hatch: always present, never editable, and
it maps to no doc-type filter at all -- so a doc type nobody has mapped
yet is still reachable on a turn the classifier can't place.
"""
from dataclasses import dataclass

UNCLASSIFIED = "unclassified"
# The deterministic workflow answers this one without retrieval or a model
# call when one document clearly matches (its lookup_document node).
DOCUMENTS_LOOKUP = "documents_lookup"


@dataclass(frozen=True)
class QuestionType:
    name: str
    description: str


QUESTION_TYPES: list[QuestionType] = [
    QuestionType("product_characteristics", "materials, dimensions, thread types, color, etc."),
    QuestionType("system_contents", "what's included in a set/system, SKUs, levels in the tray"),
    QuestionType("technique_procedural", "step-by-step surgical technique, procedural walkthrough, applications of products or parts, etc.",),
    QuestionType(
        DOCUMENTS_LOOKUP,
        "pull up a whole document rather than ask about its contents, named by its title, "
        "document type, product family, or tray (e.g. 'the REFLEX HYBRID tray overhead guide')",
    ),
    QuestionType("compatibility_lookup", "which products/parts are compatible with which others, e.g. screws with plates, implants with instruments, etc.",),
    QuestionType("features_benefits", "product features and benefits, selling points, why choose this product"),
    QuestionType(UNCLASSIFIED, "none of the other question types clearly applies"),
]

QUESTION_TYPE_NAMES: list[str] = [question_type.name for question_type in QUESTION_TYPES]
