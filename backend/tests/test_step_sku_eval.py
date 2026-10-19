"""evals/step_sku_eval.py's scoring (issue #28), on a fake lookup: no
database, no model."""
import uuid

from evals import harness, step_sku_eval
from evals.step_sku_eval import evaluate_row, load_step_dataset, run_step_eval, summarize
from techniques.models import StepItemOut, StepSkuOut, TechniqueOut, TechniqueStepOut

DOCUMENT_ID = uuid.uuid4()


def _item(*skus: str, chunk_index: int = 2) -> StepItemOut:
    return StepItemOut(
        mention="item",
        item_type="drill bit",
        condition=None,
        optional=False,
        specs={},
        applies_when={},
        resolution="in_document",
        skus=[StepSkuOut(sku=sku, description=None) for sku in skus],
        citations=[f"{DOCUMENT_ID}#{chunk_index}"],
    )


def _technique(name: str, steps: list[TechniqueStepOut]) -> TechniqueOut:
    return TechniqueOut(
        document_id=DOCUMENT_ID,
        filename="lisfranc.pdf",
        product_family="Ancillary Foot Recon System",
        name=name,
        parameters=[],
        required_trays=[],
        unbound_parameters=[],
        steps=steps,
    )


_TECHNIQUES = [
    _technique(
        "Lisfranc Homerun",
        [
            TechniqueStepOut(
                step_order=1, step_label="Step 1. Approach", citations=[f"{DOCUMENT_ID}#1"],
                items=[_item("MSG14150")],
            ),
            TechniqueStepOut(
                step_order=2,
                step_label="Step 5. Drill",
                citations=[f"{DOCUMENT_ID}#2"],
                items=[_item("MSN10015"), _item("MSN20014", "MSN20013", chunk_index=40)],
            ),
        ],
    )
]


async def _lookup(product_family, technique=None, parameters=None):
    return [record for record in _TECHNIQUES if technique is None or technique in record.name.lower()]


async def _chunk_indexes(document_id):
    return {1, 2}


async def test_a_step_row_scores_set_precision_and_recall_and_dangling_citations():
    result = await evaluate_row(
        {
            "id": "lisfranc-step-5-3.7",
            "product_family": "Ancillary Foot Recon System",
            "technique": "homerun",
            "step_number": 5,
            "parameters": {"screw_diameter": "3.7"},
            "expected_skus": ["MSN10015", "MSN10016", "MSN20014"],
        },
        _lookup,
        _chunk_indexes,
    )

    assert result.steps == ["Step 5. Drill"]
    assert result.predicted_skus == ["MSN10015", "MSN20013", "MSN20014"]
    assert (result.precision, result.recall) == (2 / 3, 2 / 3)
    assert (result.missing, result.extra) == (["MSN10016"], ["MSN20013"])
    assert result.dangling_citations == [f"{DOCUMENT_ID}#40"]


async def test_a_technique_row_unions_every_step_and_an_unmatched_row_scores_zero_recall():
    results = await run_step_eval(
        [
            {"id": "whole", "document": "Lisfranc", "product_family": "F", "technique": "homerun",
             "expected_skus": ["MSG14150", "MSN10015"]},
            {"id": "label", "document": "Lisfranc", "product_family": "F", "step": "drill",
             "expected_skus": ["MSN10015"]},
            {"id": "missing", "document": "Other", "product_family": "F", "technique": "intercuneiform",
             "expected_skus": ["MSN10017"]},
        ],
        _lookup,
        _chunk_indexes,
    )

    whole, label, missing = results
    assert whole.recall == 1.0 and whole.precision == 2 / 4
    assert label.steps == ["Step 5. Drill"]
    assert (missing.error, missing.recall, missing.missing) == (
        "no stored technique matched", 0.0, ["MSN10017"]
    )

    report = summarize(results)
    assert report.matched_rows == 2
    assert [(document.document, document.rows) for document in report.documents] == [
        ("Lisfranc", 2), ("Other", 1)
    ]
    assert report.recall == 3 / 4


def test_the_golden_step_dataset_loads_and_names_its_parameters():
    rows = load_step_dataset(step_sku_eval.DEFAULT_STEP_DATASET)

    assert {row["id"] for row in rows} >= {"mis-bunion-proximal-screw", "reflex-tetra-ultra-lapidus-30mm"}
    assert all(row["expected_skus"] and row["product_family"] for row in rows)


def test_the_steps_command_runs_without_an_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    ran = []

    async def fake_run_steps(dataset_path, output_path):
        ran.append(dataset_path)

    monkeypatch.setattr(harness, "_run_steps", fake_run_steps)
    harness.main(["steps"])

    assert ran == [step_sku_eval.DEFAULT_STEP_DATASET]
