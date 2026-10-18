"""Converts feedback-notes.csv (human-reviewed chatbot Q&A feedback) into the
JSONL golden datasets consumed by evals/harness.py: one QA/citation file per
Product Family (mis-foot-recon-system.jsonl, reflex-foot-recon-systems.jsonl,
reflex-hybrid.jsonl) and intent_detection.jsonl (family routing, question
type and clarification eval, ticket 09 / issues #18 and #19) -- the latter
also gets a few hand-authored rows appended after the CSV-derived ones:
family-clarification cases (_AMBIGUOUS_FIXTURES) and extra documents_lookup
examples (_DOCUMENTS_LOOKUP_FIXTURES, the rarest question type in the CSV).
Rerun this after new rows land in feedback-notes.csv.

feedback-notes.csv stays exactly as reviewers wrote it, so its "Indexed
System" column still says MIS or REFLEX. product_family_for() translates
that to the Product Family each question is actually about, and
question_types_for() translates its free-text "Question Type" column to the
slugs in agents/question_types.py, so every label here is a value the
deterministic workflow can actually resolve to.
"""
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

DATASET_DIRECTORY = Path(__file__).parent
CSV_PATH = DATASET_DIRECTORY / "feedback-notes.csv"

COL_SYSTEM = 0
COL_PROMPT = 1
COL_PROVIDED_ANSWER = 2
COL_PREFERRED_ANSWER = 3
COL_CONTENT_FEEDBACK = 4
COL_FORMATTING_FEEDBACK = 5
COL_QUESTION_TYPE = 6
COL_CITATIONS = 7

MIS_FOOT_RECON = "MIS - Foot Recon System"
REFLEX_FOOT_RECON = "REFLEX Foot Recon Systems"
REFLEX_HYBRID = "REFLEX HYBRID"
CANNULATED_SCREWS = "Cannulated Screws"

# The CSV's free-text question types -> agents/question_types.py slugs.
# "Pull resource" became documents_lookup. Nothing in the CSV was labelled
# compatibility_lookup or features_benefits, so no row expects them.
_QUESTION_TYPE_SLUGS: dict[str, str] = {
    "Specs - product characteristics": "product_characteristics",
    "Specs - system contents; SKU/ordering info": "system_contents",
    "Technique/procedural": "technique_procedural",
    "Technique/procedural steps": "technique_procedural",
    "Pull resource": "documents_lookup",
}


def split_lines(value: str) -> list[str]:
    return [line.strip() for line in value.split("\n") if line.strip()]


def strip_asides(citation: str) -> str:
    return re.sub(r"\s*\([^)]*\)", "", citation).strip()


def product_family_for(indexed_system: str, prompt: str) -> str:
    """The Product Family a CSV row's question is about. MIS rows are all
    about the MIS Foot Recon tray (screws, guidepins, the screwdriver, back
    table setup), never MIS Power. REFLEX spans two families: questions
    naming HYBRID belong to REFLEX HYBRID, and the rest (TETRA, MINI,
    Dynamic Disc, the reamers and drills in "the reflex tray") belong to
    REFLEX Foot Recon Systems. Raises on any other value so a new system in
    the CSV gets a deliberate mapping instead of a silent guess.
    """
    if indexed_system == "MIS":
        return MIS_FOOT_RECON
    if indexed_system == "REFLEX":
        return REFLEX_HYBRID if "hybrid" in prompt.lower() else REFLEX_FOOT_RECON
    raise ValueError(f"No Product Family mapping for indexed system {indexed_system!r}")


def question_types_for(raw_question_types: str) -> list[str]:
    slugs: dict[str, None] = {}
    for raw_question_type in split_lines(raw_question_types):
        if raw_question_type not in _QUESTION_TYPE_SLUGS:
            raise ValueError(f"No question type slug for {raw_question_type!r}")
        slugs.setdefault(_QUESTION_TYPE_SLUGS[raw_question_type], None)
    return list(slugs)


def load_rows() -> list[list[str]]:
    with open(CSV_PATH, newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))
    # rows[0] is a "Table 1" title row, rows[1] is the header
    return rows[2:]


def build_qa_record(row: list[str]) -> dict:
    prompt = row[COL_PROMPT].strip()
    return {
        "query": prompt,
        "expected_answer": row[COL_PREFERRED_ANSWER].strip(),
        "expected_citations": [
            strip_asides(citation) for citation in split_lines(row[COL_CITATIONS])
        ],
        "product_family": product_family_for(row[COL_SYSTEM].strip(), prompt),
        "question_type": question_types_for(row[COL_QUESTION_TYPE]),
        "provided_answer": row[COL_PROVIDED_ANSWER].strip(),
        "content_feedback": row[COL_CONTENT_FEEDBACK].strip(),
        "formatting_feedback": row[COL_FORMATTING_FEEDBACK].strip(),
    }


def build_intent_record(row: list[str]) -> dict:
    prompt = row[COL_PROMPT].strip()
    question_types = question_types_for(row[COL_QUESTION_TYPE])
    return intent_record(
        prompt, product_family_for(row[COL_SYSTEM].strip(), prompt), question_types
    )


def intent_record(prompt: str, product_family: str | None, question_types: list[str]) -> dict:
    """expects_family_clarification follows from having no family (issue
    #18 asks whenever the classifier can't name one), and
    expects_question_type_clarification from having more than one type
    (issue #19 asks whenever several are detected).
    """
    return {
        "turns": [{"role": "user", "content": prompt}],
        "expected_product_family": product_family,
        "expected_question_type": question_types,
        "expects_family_clarification": product_family is None,
        "expects_question_type_clarification": len(question_types) > 1,
    }


# feedback-notes.csv is real human-reviewed chat feedback, so every row in it
# is a query someone actually asked -- every single one confidently names or
# implies one family, meaning expects_family_clarification is False for all
# 30 CSV rows above. Ticket 09 (chat clarification flow) needs at least one
# real family-clarification case to verify the interrupt()/resume path
# against, which the CSV has no example of -- these are hand-authored
# instead, appended after the CSV-derived records rather than mixed into
# feedback-notes.csv itself (that file stays human-reviewed-only). A query
# that names no product and gives no other family-distinguishing signal is
# the realistic shape of "can't confidently tell."
_AMBIGUOUS_FIXTURES: list[dict] = [
    intent_record("What screws are in the set?", None, ["system_contents"]),
    intent_record("How do I set up the back table for this case?", None, ["technique_procedural"]),
]

# documents_lookup (fetch-a-document-by-name, not an open question; the CSV's
# "Pull resource") is the rarest question type in feedback-notes.csv -- only
# 1 of 30 real rows. These extra hand-authored examples give the eval more
# than a single data point to check question-type classification against.
# Each names a family: "Cannulated screw reference" names the Cannulated
# Screws family, which didn't exist as a routing target when this fixture
# was first written as an ambiguous case.
#
# NOT YET IMPLEMENTED, flagging for whoever picks this up: a query
# classified as documents_lookup could skip most of the deterministic
# pipeline in agents/workflows/deterministic.py. It's a direct "fetch this
# document" lookup, not a question needing synonym expansion, reranking,
# generation or a fact check. That needs a routing branch after
# clarify_question_type, which is a graph-structure decision, not a
# fixture-data one -- these rows exist so that work has something real to
# test against once it lands.
_DOCUMENTS_LOOKUP_FIXTURES: list[dict] = [
    intent_record("MIS procedure guide", MIS_FOOT_RECON, ["documents_lookup"]),
    intent_record("Cannulated screw reference", CANNULATED_SCREWS, ["documents_lookup"]),
    intent_record("MIS inventory control", MIS_FOOT_RECON, ["documents_lookup"]),
]


def dataset_file_name(product_family: str) -> str:
    """"MIS - Foot Recon System" -> "mis-foot-recon-system.jsonl"."""
    return re.sub(r"[^a-z0-9]+", "-", product_family.lower()).strip("-") + ".jsonl"


def write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as file:
        file.writelines(json.dumps(record, ensure_ascii=False) + "\n" for record in records)


def main() -> None:
    rows = load_rows()

    by_family: dict[str, list[dict]] = defaultdict(list)
    intent_records = []
    for row in rows:
        record = build_qa_record(row)
        by_family[record["product_family"]].append(record)
        intent_records.append(build_intent_record(row))

    for product_family, records in by_family.items():
        out_path = DATASET_DIRECTORY / dataset_file_name(product_family)
        write_jsonl(out_path, records)
        print(f"wrote {len(records)} examples to {out_path.name}")

    intent_records.extend(_AMBIGUOUS_FIXTURES)
    intent_records.extend(_DOCUMENTS_LOOKUP_FIXTURES)
    intent_path = DATASET_DIRECTORY / "intent_detection.jsonl"
    write_jsonl(intent_path, intent_records)
    print(f"wrote {len(intent_records)} examples to {intent_path.name}")


if __name__ == "__main__":
    main()
