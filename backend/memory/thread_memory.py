"""Thread memory (issue #7, Ticket A): what a turn knows about the earlier
turns of the same conversation, at a fixed and small token cost.

Each finished turn leaves one small TurnRecord in the `turn_memory` state
channel (agents/state.py; written by the deterministic workflow's finalize),
built purely from state the graph already has -- no model call. generate
renders the records into one "Earlier in this conversation" block
(render_thread_memory): the last RECENT_TURNS_IN_FULL turns as full
records, older ones collapsed to their SKUs and document ids, the whole
block capped at THREAD_MEMORY_TOKEN_CAP by memory/budget.py, oldest dropped
first. That block replaces sending every earlier message verbatim, which
grew the prompt by a whole Q&A pair per turn; `messages` stays the
transcript the UI shows.

Records are summaries, never transcripts: the answer is cut to
ANSWER_TOKEN_LIMIT and stripped of citation brackets (its cited chunk ids
are kept separately), and parts are just a SKU and its one-line
description.
"""
import re

from typing_extensions import (
    TypedDict,  # pydantic v2 needs this, not typing.TypedDict, on Python <3.12
)

from memory.budget import fit_to_budget, truncate_to_tokens

THREAD_MEMORY_TOKEN_CAP = 1500
RECENT_TURNS_IN_FULL = 3
# Per-record caps, so one long turn can't crowd the others out of the
# block on its own.
QUESTION_TOKEN_LIMIT = 80
ANSWER_TOKEN_LIMIT = 150
DESCRIPTION_TOKEN_LIMIT = 16
MAX_PARTS_PER_TURN = 8
MAX_CITED_CHUNKS_PER_TURN = 5

THREAD_MEMORY_HEADER = (
    "Earlier in this conversation (oldest first). Use it to understand what "
    "a follow-up refers to. Parts listed here are catalog records from those "
    "turns. Cite only this turn's context passages, never a source listed here."
)


class RememberedPart(TypedDict):
    sku: str
    description: str


class TurnRecord(TypedDict):
    question: str
    answer: str
    parts: list[RememberedPart]
    cited_chunk_ids: list[str]
    # The turn's resolved Product Family display name (issue #18), or None
    # for an unscoped turn. Shown in the block only: the family a follow-up
    # stays on is still detect_intent's carry-over (issue #22), not this.
    family: str | None


def remembered_parts(answer: str, resolved_parts: list[dict], query_parts: list[dict]) -> list[dict]:
    """The turn's Parts worth remembering: the resolved Parts the answer
    actually names, or, when it names none, the Parts the rep's own words
    matched. Not every resolved Part -- a procedural turn resolves dozens
    from passage text, and most never reach the answer.
    """
    named = [
        part
        for part in resolved_parts
        if part.get("sku") and re.search(rf"\b{re.escape(part['sku'])}\b", answer)
    ]
    candidates = named or [part for part in query_parts if part.get("sku")]
    return list({part["sku"]: part for part in candidates}.values())[:MAX_PARTS_PER_TURN]


def build_turn_record(
    *, question: str, answer: str, parts: list[dict], cited_chunk_ids: list[str], family: str | None
) -> TurnRecord:
    """One turn's record, every field already cut to its cap. `answer`
    should come without citation brackets; `parts` are catalog Part records
    (at least `sku`), already chosen by remembered_parts.
    """
    return {
        "question": truncate_to_tokens(" ".join(question.split()), QUESTION_TOKEN_LIMIT),
        "answer": truncate_to_tokens(" ".join(answer.split()), ANSWER_TOKEN_LIMIT),
        "parts": [
            {
                "sku": part["sku"],
                "description": truncate_to_tokens(part.get("description") or "", DESCRIPTION_TOKEN_LIMIT),
            }
            for part in parts[:MAX_PARTS_PER_TURN]
        ],
        "cited_chunk_ids": list(dict.fromkeys(cited_chunk_ids))[:MAX_CITED_CHUNKS_PER_TURN],
        "family": family,
    }


def _turn_label(turn_number: int, record: TurnRecord) -> str:
    return f"Turn {turn_number} ({record['family']})" if record.get("family") else f"Turn {turn_number}"


def _document_ids(record: TurnRecord) -> list[str]:
    return list(dict.fromkeys(chunk_id.split("#")[0] for chunk_id in record.get("cited_chunk_ids") or []))


def _full_entry(turn_number: int, record: TurnRecord) -> str:
    lines = [
        f"{_turn_label(turn_number, record)}:",
        f"Rep asked: {record['question']}",
        f"Answer given: {record['answer']}",
    ]
    if record.get("parts"):
        lines.append(
            "Parts: " + "; ".join(f"{part['sku']} ({part['description']})" for part in record["parts"])
        )
    if record.get("cited_chunk_ids"):
        lines.append("Sources cited: " + ", ".join(record["cited_chunk_ids"]))
    return "\n".join(lines)


def _collapsed_entry(turn_number: int, record: TurnRecord) -> str | None:
    details = []
    if record.get("parts"):
        details.append("parts " + ", ".join(part["sku"] for part in record["parts"]))
    if _document_ids(record):
        details.append("documents " + ", ".join(_document_ids(record)))
    # A turn with neither has nothing left to say once collapsed.
    if not details:
        return None
    return f"{_turn_label(turn_number, record)}: " + "; ".join(details)


def render_thread_memory(records: list[TurnRecord]) -> tuple[str, int]:
    """The "Earlier in this conversation" block for `records` (oldest
    first), and its token count. Empty, at zero tokens, on a thread's first
    turn.
    """
    older_count = max(len(records) - RECENT_TURNS_IN_FULL, 0)
    entries = [
        entry
        for turn_number, record in enumerate(records[:older_count], start=1)
        if (entry := _collapsed_entry(turn_number, record))
    ]
    entries += [
        _full_entry(turn_number, record)
        for turn_number, record in enumerate(records[older_count:], start=older_count + 1)
    ]
    return fit_to_budget(THREAD_MEMORY_HEADER, entries, THREAD_MEMORY_TOKEN_CAP)


def remembered_part_lines(records: list[TurnRecord]) -> str:
    """Every remembered Part as "SKU: description" lines -- for the fact
    check's source text, since a SKU the block showed the model is a
    catalog fact, not an invented one.
    """
    return "\n".join(
        f"{part['sku']}: {part['description']}" for record in records for part in record.get("parts") or []
    )
