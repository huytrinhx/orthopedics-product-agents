"""Converts feedback-notes.csv (human-reviewed chatbot Q&A feedback) into the
JSONL golden datasets consumed by evals/harness.py: mis.jsonl, reflex.jsonl
(QA/citation eval, one file per product system) and intent_detection.jsonl
(system-routing + clarification eval, ticket 09) -- the latter also gets a
few hand-authored rows appended after the CSV-derived ones: expects_
clarification: True cases (_AMBIGUOUS_FIXTURES) and extra "Pull resource"
examples (_PULL_RESOURCE_FIXTURES, the rarest question_type in the CSV).
Rerun this after new rows land in feedback-notes.csv.
"""
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

DATASET_DIR = Path(__file__).parent
CSV_PATH = DATASET_DIR / "feedback-notes.csv"

COL_SYSTEM = 0
COL_PROMPT = 1
COL_PROVIDED_ANSWER = 2
COL_PREFERRED_ANSWER = 3
COL_CONTENT_FEEDBACK = 4
COL_FORMATTING_FEEDBACK = 5
COL_QUESTION_TYPE = 6
COL_CITATIONS = 7


def split_lines(value: str) -> list[str]:
    return [line.strip() for line in value.split("\n") if line.strip()]


def strip_asides(citation: str) -> str:
    return re.sub(r"\s*\([^)]*\)", "", citation).strip()


def load_rows() -> list[list[str]]:
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    # rows[0] is a "Table 1" title row, rows[1] is the header
    return rows[2:]


def build_qa_record(row: list[str]) -> dict:
    return {
        "query": row[COL_PROMPT].strip(),
        "expected_answer": row[COL_PREFERRED_ANSWER].strip(),
        "expected_citations": [
            strip_asides(c) for c in split_lines(row[COL_CITATIONS])
        ],
        "system": row[COL_SYSTEM].strip(),
        "question_type": split_lines(row[COL_QUESTION_TYPE]),
        "provided_answer": row[COL_PROVIDED_ANSWER].strip(),
        "content_feedback": row[COL_CONTENT_FEEDBACK].strip(),
        "formatting_feedback": row[COL_FORMATTING_FEEDBACK].strip(),
    }


def build_intent_record(row: list[str]) -> dict:
    return {
        "turns": [{"role": "user", "content": row[COL_PROMPT].strip()}],
        "expected_system": row[COL_SYSTEM].strip(),
        "expected_question_type": split_lines(row[COL_QUESTION_TYPE]),
        "expects_clarification": False,
    }


# feedback-notes.csv is real human-reviewed chat feedback, so every row in it
# is a query someone actually asked -- every single one confidently names or
# implies one system, meaning expects_clarification is False for all 30 CSV
# rows above. Ticket 09 (chat clarification flow) needs at least one real
# expects_clarification: True case to verify the interrupt()/resume path
# against, which the CSV has no example of and build_intent_record has no way
# to derive -- these are hand-authored instead, appended after the CSV-derived
# records rather than mixed into feedback-notes.csv itself (that file stays
# human-reviewed-only). Modeled on the CSV's one genuinely double-labeled row
# (row 25, "Technique/procedural steps\nSpecs - product characteristics") --
# a query that names neither MIS nor REFLEX and gives no other system-
# distinguishing signal is the realistic shape of "can't confidently tell."
_AMBIGUOUS_FIXTURES: list[dict] = [
    {
        "turns": [{"role": "user", "content": "What screws are in the set?"}],
        "expected_system": None,
        "expected_question_type": ["Specs - system contents; SKU/ordering info"],
        "expects_clarification": True,
    },
    {
        "turns": [{"role": "user", "content": "How do I set up the back table for this case?"}],
        "expected_system": None,
        "expected_question_type": ["Technique/procedural"],
        "expects_clarification": True,
    },
]

# "Pull resource" (fetch-a-document-by-name, not an open question) is the
# rarest question type in feedback-notes.csv -- only 1 of 30 real rows (see
# build_qa_record's CSV-derived output). These extra hand-authored examples
# give the eval more than a single data point to check question_type
# classification against; two name a system explicitly (so
# expects_clarification stays False, same logic as the CSV rows), one names
# only a part family with no system context and so is ambiguous, same
# reasoning as _AMBIGUOUS_FIXTURES above.
#
# NOT YET IMPLEMENTED, flagging for whoever picks this up: a query
# classified as "Pull resource" should skip the full
# detect_intent -> resolve_synonyms -> hybrid_retrieve -> rerank -> generate
# -> self_eval (-> reformulate retry loop) pipeline in
# agents/workflows/deterministic.py entirely. It's a direct "fetch this
# document" lookup, not a question needing synonym expansion, LLM reranking,
# generation, or faithfulness/relevance self-eval -- running the full
# reasoning loop for it is pure latency/cost with no accuracy benefit. This
# needs an actual routing branch out of detect_intent (or a dedicated fast
# node) once question_type is used for more than observability, which is a
# graph-structure decision, not a fixture-data one -- these rows exist so
# that work has something real to test against once it lands.
_PULL_RESOURCE_FIXTURES: list[dict] = [
    {
        "turns": [{"role": "user", "content": "MIS procedure guide"}],
        "expected_system": "MIS",
        "expected_question_type": ["Pull resource"],
        "expects_clarification": False,
    },
    {
        "turns": [{"role": "user", "content": "Cannulated screw reference"}],
        "expected_system": None,
        "expected_question_type": ["Pull resource"],
        "expects_clarification": True,
    },
    {
        "turns": [{"role": "user", "content": "MIS inventory control"}],
        "expected_system": "MIS",
        "expected_question_type": ["Pull resource"],
        "expects_clarification": False,
    },
]


def write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(record, ensure_ascii=False) + "\n" for record in records)


def main() -> None:
    rows = load_rows()

    by_system: dict[str, list[dict]] = defaultdict(list)
    intent_records = []
    for row in rows:
        system = row[COL_SYSTEM].strip()
        by_system[system].append(build_qa_record(row))
        intent_records.append(build_intent_record(row))

    for system, records in by_system.items():
        out_path = DATASET_DIR / f"{system.lower()}.jsonl"
        write_jsonl(out_path, records)
        print(f"wrote {len(records)} examples to {out_path.name}")

    intent_records.extend(_AMBIGUOUS_FIXTURES)
    intent_records.extend(_PULL_RESOURCE_FIXTURES)
    intent_path = DATASET_DIR / "intent_detection.jsonl"
    write_jsonl(intent_path, intent_records)
    print(f"wrote {len(intent_records)} examples to {intent_path.name}")


if __name__ == "__main__":
    main()
