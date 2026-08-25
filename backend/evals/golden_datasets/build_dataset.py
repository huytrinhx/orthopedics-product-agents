"""Converts feedback-notes.csv (human-reviewed chatbot Q&A feedback) into the
JSONL golden datasets consumed by evals/harness.py: mis.jsonl, reflex.jsonl
(QA/citation eval, one file per product system) and intent_detection.jsonl
(system-routing eval). Rerun this after new rows land in feedback-notes.csv.
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
        "expects_clarification": False,
    }


def write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


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

    intent_path = DATASET_DIR / "intent_detection.jsonl"
    write_jsonl(intent_path, intent_records)
    print(f"wrote {len(intent_records)} examples to {intent_path.name}")


if __name__ == "__main__":
    main()
