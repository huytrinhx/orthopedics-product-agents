"""Step -> SKU set eval (issue #28): scores the techniques ingestion stored
against golden_datasets/technique_steps/step_skus.jsonl, hand-built from
the 2026-10-18 evaluation of production's Surgical Technique documents.

Each row names a family, a technique (words of its name, or null for any),
a step (words of its label, its printed number, or neither for the whole
technique) and the surgeon's parameters, and lists the SKUs that step (or
technique) uses. The prediction is what `techniques.service.technique_steps`
returns for the same family, technique and parameters -- the lookup an
answer will be built from (issue #29), not a copy of it -- unioned over
the matched steps' items. A row scores set precision and recall, and the
report aggregates them per document (summed over its rows) and overall.

It also checks that every citation the matched steps carry is a chunk the
document has, the acceptance's "resolve to real citation ids".

No model call: it reads what indexing stored. So it needs a DATABASE_URL
whose documents were indexed after this shipped, and Neo4j for the
catalog descriptions. Run it through evals/harness.py's `steps` command.
"""
import json
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from documents.repository import list_chunks
from ingestion.headings import step_number
from techniques.models import TechniqueOut, TechniqueStepOut
from techniques.service import technique_steps

DEFAULT_STEP_DATASET = Path(__file__).parent / "golden_datasets" / "technique_steps" / "step_skus.jsonl"

# Called like techniques.service.technique_steps.
StepLookup = Callable[..., Awaitable[list[TechniqueOut]]]
# Document id -> its chunk indexes.
ChunkIndexLoader = Callable[[uuid.UUID], Awaitable[set[int]]]


@dataclass
class StepRowResult:
    id: str
    document: str
    product_family: str
    expected_skus: list[str]
    predicted_skus: list[str] = field(default_factory=list)
    # The stored document and technique the row matched; None when no
    # technique (or no step in it) matched.
    document_id: str | None = None
    filename: str | None = None
    technique: str | None = None
    steps: list[str] = field(default_factory=list)
    # Other techniques that also matched the row; the first was scored.
    other_candidates: int = 0
    precision: float | None = None
    recall: float | None = None
    missing: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)
    citations: int = 0
    dangling_citations: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class DocumentScore:
    document: str
    rows: int
    precision: float | None
    recall: float | None


@dataclass
class StepReport:
    rows: int
    matched_rows: int
    precision: float | None
    recall: float | None
    dangling_citations: int
    documents: list[DocumentScore]


def load_step_dataset(dataset_path: str | Path) -> list[dict]:
    with open(dataset_path, encoding="utf-8") as dataset_file:
        return [json.loads(line) for line in dataset_file if line.strip()]


def _words(text: str) -> set[str]:
    return set("".join(character if character.isalnum() else " " for character in text.lower()).split())


def matching_steps(technique: TechniqueOut, row: dict) -> list[TechniqueStepOut]:
    """The technique's steps the row means: every step when it names none,
    else those whose printed number or label words match."""
    if row.get("step_number") is None and not row.get("step"):
        return list(technique.steps)
    return [
        step
        for step in technique.steps
        if (row.get("step_number") is None or step_number(step.step_label) == row["step_number"])
        and (not row.get("step") or _words(row["step"]) <= _words(step.step_label))
    ]


def _fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


async def _chunk_indexes(document_id: uuid.UUID) -> set[int]:
    return {chunk.chunk_index for chunk in await list_chunks(document_id)}


async def evaluate_row(
    row: dict, lookup: StepLookup = technique_steps, load_chunk_indexes: ChunkIndexLoader = _chunk_indexes
) -> StepRowResult:
    expected = sorted(set(row["expected_skus"]))
    result = StepRowResult(
        id=row["id"],
        document=row.get("document") or row["product_family"],
        product_family=row["product_family"],
        expected_skus=expected,
    )
    techniques = await lookup(row["product_family"], row.get("technique"), row.get("parameters") or {})
    candidates = [(technique, matching_steps(technique, row)) for technique in techniques]
    candidates = [(technique, steps) for technique, steps in candidates if steps]
    if not candidates:
        result.error = (
            "no stored technique matched"
            if not techniques
            else f"{len(techniques)} technique(s) matched, but no step did"
        )
        result.recall, result.missing = 0.0, expected
        return result

    technique, steps = candidates[0]
    result.other_candidates = len(candidates) - 1
    result.document_id = str(technique.document_id)
    result.filename = technique.filename
    result.technique = technique.name
    result.steps = [step.step_label for step in steps]
    predicted = sorted({sku.sku for step in steps for item in step.items for sku in item.skus})
    result.predicted_skus = predicted
    overlap = set(predicted) & set(expected)
    result.precision = _fraction(len(overlap), len(predicted))
    result.recall = _fraction(len(overlap), len(expected))
    result.missing = sorted(set(expected) - overlap)
    result.extra = sorted(set(predicted) - overlap)

    citations = list(
        dict.fromkeys(
            [citation for step in steps for citation in step.citations]
            + [citation for step in steps for item in step.items for citation in item.citations]
        )
    )
    known_chunks = await load_chunk_indexes(technique.document_id)
    result.citations = len(citations)
    result.dangling_citations = [
        citation for citation in citations if int(citation.rsplit("#", 1)[1]) not in known_chunks
    ]
    return result


async def run_step_eval(
    rows: list[dict],
    lookup: StepLookup = technique_steps,
    load_chunk_indexes: ChunkIndexLoader = _chunk_indexes,
) -> list[StepRowResult]:
    return [await evaluate_row(row, lookup, load_chunk_indexes) for row in rows]


def _set_scores(results: list[StepRowResult]) -> tuple[float | None, float | None]:
    """Precision and recall summed over the rows' sets (micro-averaged), so
    a big step counts for more than a small one."""
    overlap = sum(len(set(result.predicted_skus) & set(result.expected_skus)) for result in results)
    predicted = sum(len(result.predicted_skus) for result in results)
    expected = sum(len(result.expected_skus) for result in results)
    return _fraction(overlap, predicted), _fraction(overlap, expected)


def summarize(results: list[StepRowResult]) -> StepReport:
    precision, recall = _set_scores(results)
    documents: dict[str, list[StepRowResult]] = {}
    for result in results:
        documents.setdefault(result.document, []).append(result)
    return StepReport(
        rows=len(results),
        matched_rows=sum(1 for result in results if result.error is None),
        precision=precision,
        recall=recall,
        dangling_citations=sum(len(result.dangling_citations) for result in results),
        documents=[
            DocumentScore(document, len(document_results), *_set_scores(document_results))
            for document, document_results in documents.items()
        ],
    )


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def format_report(report: StepReport, results: list[StepRowResult]) -> str:
    lines = [
        f"Rows: {report.rows} ({report.matched_rows} matched a stored step)",
        f"Set precision: {_percent(report.precision)}   recall: {_percent(report.recall)}",
        f"Citations to chunks the document doesn't have: {report.dangling_citations}",
        "",
        "Per document:",
    ]
    for document in report.documents:
        lines.append(
            f"  {document.document}: {document.rows} row(s), precision "
            f"{_percent(document.precision)}, recall {_percent(document.recall)}"
        )
    lines += ["", "Rows:"]
    for result in results:
        if result.error:
            lines.append(f"  {result.id}: {result.error}")
            continue
        lines.append(
            f"  {result.id}: precision {_percent(result.precision)}, recall "
            f"{_percent(result.recall)} -- {result.technique!r} > {'; '.join(result.steps)} "
            f"({result.filename})"
        )
        if result.other_candidates:
            lines.append(f"    {result.other_candidates} other technique(s) also matched")
        if result.missing:
            lines.append(f"    missing: {', '.join(result.missing)}")
        if result.extra:
            lines.append(f"    extra: {', '.join(result.extra)}")
        if result.dangling_citations:
            lines.append(f"    dangling citations: {', '.join(result.dangling_citations)}")
    return "\n".join(lines)


def write_results(output_path: str | Path, report: StepReport, results: list[StepRowResult]) -> None:
    Path(output_path).write_text(
        json.dumps(
            {"report": asdict(report), "rows": [asdict(result) for result in results]}, indent=2
        ),
        encoding="utf-8",
    )
