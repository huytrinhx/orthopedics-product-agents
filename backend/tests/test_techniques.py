"""Issue #28's technique/step extraction and storage
(ingestion/technique_extraction.py, backend/techniques/).

The model is never called: get_chat_model is swapped for a fake that
returns a fixed outline and records the prompt. The storage tests run
against real Postgres, like test_document_skus.py, with the catalog read
and graph writes swapped for a fixture so they don't depend on what Neo4j
holds.
"""
import uuid

import pytest

from auth.repository import create_user
from catalog.repository import list_families_with_trays, upsert_families_and_trays
from document_skus.repository import DocumentSkuRow, replace_document_skus
from documents.repository import create_document, delete_document, set_tags
from ingestion import technique_extraction
from ingestion.technique_extraction import (
    NamedValue,
    StepItemReference,
    Technique,
    TechniqueOutline,
    TechniqueStep,
    extract_technique_outline,
    step_bearing_chunks,
)
from retrieval.vector_store import get_vector_store
from techniques import repository, service
from techniques.resolution import CatalogPart
from techniques.service import build_technique_rows, technique_steps

ANCILLARY = "Ancillary Foot Recon System"
CORE_FOOT = "Core Foot Plating System"

_FIXTURE_PARTS = [
    CatalogPart("MSDL3720", "SCREW, LISFRANC, SOLID, 3.7 X 20MM", frozenset({ANCILLARY}),
                "Implant", "Full", "Ø1.4x150mm", "Ø2.6mm", "T15"),
    CatalogPart("MSDL4120", "SCREW, LISFRANC, SOLID, 4.1 X 20MM", frozenset({ANCILLARY}),
                "Implant", "Full", "Ø1.4x150mm", "Ø2.8mm", "T15"),
    CatalogPart("MSN10015", "DRILL BIT, CANN, AO/QC, 2.6MM, 3.7 PILOT", frozenset({ANCILLARY})),
    CatalogPart("MSN10017", "DRILL BIT, CANN, AO/QC, 2.8MM, 4.1 PILOT", frozenset({ANCILLARY})),
    CatalogPart("MSN20014", "COUNTERSINK, CANN, LISFRANC", frozenset({ANCILLARY})),
    CatalogPart("MPP2100U", "PLATE, LAPIDUS STANDARD", frozenset({CORE_FOOT})),
]
_TRAYS_BY_FAMILY = {
    ANCILLARY: ["Ancillary Foot Recon Plating System"],
    CORE_FOOT: ["Core Foot Recon Plating System"],
}


def _named(values: dict[str, str]) -> list[NamedValue]:
    return [NamedValue(name=name, value=value) for name, value in values.items()]


def _outline(technique_name: str = "Lisfranc Homerun") -> TechniqueOutline:
    return TechniqueOutline(
        techniques=[
            Technique(
                name=technique_name,
                parameters=["screw_diameter"],
                tray_references=["Core Foot Recon Plating System"],
                steps=[
                    # Out of order, and one step points only at a chunk it
                    # wasn't given: sorted, and dropped.
                    TechniqueStep(
                        step_label="Step 5. Drill",
                        step_order=5,
                        source_chunk_indexes=[2, 99],
                        items=[
                            StepItemReference(
                                mention="Ø2.6mm drill bit for Ø3.7mm screws",
                                item_type="drill bit",
                                specs=_named({"for_screw_diameter": "3.7mm"}),
                                applies_when=_named({"screw_diameter": "3.7mm"}),
                            ),
                            StepItemReference(
                                mention="Ø2.8mm drill bit for Ø4.1mm screws",
                                item_type="drill bit",
                                specs=_named({"for_screw_diameter": "4.1mm"}),
                                applies_when=_named({"screw_diameter": "4.1mm"}),
                            ),
                            StepItemReference(mention="countersink", item_type="countersink"),
                        ],
                    ),
                    TechniqueStep(
                        step_label="Step 1. Approach", step_order=1, source_chunk_indexes=[1]
                    ),
                    TechniqueStep(
                        step_label="Step 9. Invented", step_order=9, source_chunk_indexes=[42]
                    ),
                ],
            )
        ]
    )


class _FakeChatModel:
    def __init__(self, outline: TechniqueOutline):
        self.outline = outline
        self.prompts: list = []

    def with_structured_output(self, schema):
        assert schema is TechniqueOutline
        model = self

        class _Runnable:
            async def ainvoke(self, prompt):
                model.prompts.append(prompt)
                return model.outline

        return _Runnable()


def _use_fake_model(monkeypatch, outline: TechniqueOutline) -> _FakeChatModel:
    fake = _FakeChatModel(outline)
    monkeypatch.setattr(technique_extraction, "get_chat_model", lambda *arguments: fake)
    return fake


# ---- Step chunks and the outline call ----


def test_only_chunks_with_a_step_heading_in_their_path_are_step_bearing():
    chunks = [
        (0, [], "Cover"),
        (1, ["Lisfranc Homerun", "Step 1. Approach"], "Incise."),
        (2, ["Implant Removal"], "Use the T15 driver."),
        (3, ["Optional | Plate bending"], "Bend."),
        (4, None, "Indexed before heading paths existed."),
    ]

    assert [chunk_index for chunk_index, _, _ in step_bearing_chunks(chunks)] == [1, 3]


async def test_no_step_chunks_means_no_model_call(monkeypatch):
    def no_model(*arguments):
        raise AssertionError("no chunks, no call")

    monkeypatch.setattr(technique_extraction, "get_chat_model", no_model)

    assert await extract_technique_outline([]) == TechniqueOutline()


async def test_the_outline_call_sees_each_chunk_with_its_number_and_path(monkeypatch):
    fake = _use_fake_model(monkeypatch, _outline())

    outline = await extract_technique_outline(
        [(2, ["Lisfranc Homerun", "Step 5. Drill"], "Drill with the Ø2.6mm drill bit.")]
    )

    assert outline == _outline()
    [(_, user_message)] = [message for message in fake.prompts[0] if message[0] == "user"]
    assert "[chunk 2] Lisfranc Homerun > Step 5. Drill\nDrill with the Ø2.6mm drill bit." in user_message


def test_rows_keep_only_given_chunks_and_resolve_items_in_code():
    rows = build_technique_rows(
        _outline(),
        given_chunk_indexes={1, 2},
        catalog={part.sku: part for part in _FIXTURE_PARTS},
        document_family=ANCILLARY,
        document_sku_chunks={"MSN10015": [7], "MSN10017": [7], "MSN20014": [7], "MSDL3720": [6]},
        trays_by_family=_TRAYS_BY_FAMILY,
    )

    [technique] = rows
    assert technique.required_trays == ["Core Foot Recon Plating System"]
    assert [step.step_label for step in technique.steps] == ["Step 1. Approach", "Step 5. Drill"]
    drill_step = technique.steps[1]
    assert drill_step.source_chunk_indexes == [2]
    assert [(item.resolved_skus, item.resolution) for item in drill_step.items] == [
        (["MSN10015"], "in_document"),
        (["MSN10017"], "in_document"),
        (["MSN20014"], "in_document"),
    ]
    assert drill_step.items[0].applies_when == {"screw_diameter": "3.7mm"}
    # The step's chunk, then the tray table the index found the drill in.
    assert drill_step.items[0].source_chunk_indexes == [2, 7]


# ---- Storage and the lookup, against real Postgres ----


# What replace_document_procedures was last given, per document.
_WRITTEN_PROCEDURES: dict[str, list[dict]] = {}


class _FixtureGraphClient:
    async def list_catalog_parts(self, skus=None):
        return [
            {
                "sku": part.sku,
                "description": part.description,
                "families": sorted(part.families),
                "item_type": part.item_type,
                "thread": part.thread,
                "guidewire_spec": part.guidewire_spec,
                "pre_drill_spec": part.pre_drill_spec,
                "driver_spec": part.driver_spec,
            }
            for part in _FIXTURE_PARTS
            if skus is None or part.sku in skus
        ]

    async def list_trays_by_family(self):
        return _TRAYS_BY_FAMILY

    async def replace_document_procedures(self, document_id, procedures):
        _WRITTEN_PROCEDURES[document_id] = procedures


def _unit_vector(index: int) -> list[float]:
    vector = [0.0] * 1536
    vector[index] = 1.0
    return vector


async def _create_tagged_document() -> tuple[uuid.UUID, str]:
    suffix = uuid.uuid4().hex[:10]
    family_name = f"{ANCILLARY} {suffix}"
    await upsert_families_and_trays({family_name: [f"Tray {suffix}"]})
    family = next(family for family in await list_families_with_trays() if family.name == family_name)
    user = await create_user(f"user-{suffix}@example.com", None, False, True)
    document = await create_document(
        filename=f"lisfranc-{suffix}.pdf", storage_path=f"/tmp/storage-{suffix}.pdf", uploaded_by=user.id
    )
    await set_tags(document.id, document_type_id=None, product_family_id=family.id, tray_ids=[])
    return document.id, family_name


async def _write_document(document_id: uuid.UUID) -> None:
    async with get_vector_store() as store:
        await store.upsert_chunks(
            document_id,
            [
                {
                    "chunk_index": chunk_index,
                    "page_number": chunk_index + 1,
                    "section_title": heading_path[-1] if heading_path else None,
                    "heading_path": heading_path,
                    "content": content,
                    "embedding": _unit_vector(chunk_index),
                }
                for chunk_index, heading_path, content in [
                    (0, [], "Cover page."),
                    (1, ["Lisfranc Homerun", "Step 1. Approach"], "Make the incision."),
                    (2, ["Lisfranc Homerun", "Step 5. Drill"], "Drill with the Ø2.6mm drill bit."),
                    (3, ["Instruments"], "MSN10015 Drill\nMSN10017 Drill\nMSN20014 Countersink"),
                ]
            ],
        )
    await replace_document_skus(
        document_id,
        [
            DocumentSkuRow(3, sku, sku, "exact", True, None, f"{sku} row")
            for sku in ("MSN10015", "MSN10017", "MSN20014")
        ],
    )


@pytest.fixture
def stubbed_ingestion(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(service, "get_graph_client", _FixtureGraphClient)
    return monkeypatch


async def test_indexing_stores_the_outline_and_the_lookup_reads_it(stubbed_ingestion):
    fake = _use_fake_model(stubbed_ingestion, _outline())
    document_id, family = await _create_tagged_document()
    try:
        await _write_document(document_id)

        await service.index_document_techniques(document_id, product_family=family)

        # Only the two step chunks went to the model.
        [(_, user_message)] = [message for message in fake.prompts[0] if message[0] == "user"]
        assert "[chunk 1]" in user_message and "[chunk 2]" in user_message
        assert "[chunk 0]" not in user_message and "[chunk 3]" not in user_message
        assert _WRITTEN_PROCEDURES[str(document_id)] == [
            {"name": "Lisfranc Homerun", "trays": ["Core Foot Recon Plating System"]}
        ]

        [technique] = await technique_steps(family, "homerun", {"screw_diameter": "3.7"})
        assert technique.document_id == document_id
        assert technique.unbound_parameters == []
        assert [step.step_label for step in technique.steps] == ["Step 1. Approach", "Step 5. Drill"]
        drill_step = technique.steps[1]
        assert drill_step.citations == [f"{document_id}#2"]
        assert [[sku.sku for sku in item.skus] for item in drill_step.items] == [
            ["MSN10015"], ["MSN20014"]
        ]
        assert drill_step.items[0].skus[0].description == "DRILL BIT, CANN, AO/QC, 2.6MM, 3.7 PILOT"
        assert drill_step.items[0].citations == [f"{document_id}#2", f"{document_id}#3"]

        # Without the parameter, every option comes back, and it's flagged.
        [unbound] = await technique_steps(family, "homerun")
        assert unbound.unbound_parameters == ["screw_diameter"]
        assert len(unbound.steps[1].items) == 3
        assert await technique_steps(family, "intercuneiform") == []
    finally:
        await delete_document(document_id)


async def test_reindex_replaces_techniques_and_delete_removes_them(stubbed_ingestion):
    _use_fake_model(stubbed_ingestion, _outline("First Name"))
    document_id, family = await _create_tagged_document()
    await _write_document(document_id)
    await service.index_document_techniques(document_id, product_family=family)

    _use_fake_model(stubbed_ingestion, _outline("Second Name"))
    await service.index_document_techniques(document_id, product_family=family)

    assert [record.name for record in await repository.list_techniques(document_id=document_id)] == [
        "Second Name"
    ]
    assert _WRITTEN_PROCEDURES[str(document_id)][0]["name"] == "Second Name"

    await delete_document(document_id)
    assert await repository.list_techniques(document_id=document_id) == []


async def test_a_document_with_no_step_chunks_clears_its_techniques(stubbed_ingestion):
    _use_fake_model(stubbed_ingestion, _outline())
    document_id, family = await _create_tagged_document()
    try:
        await _write_document(document_id)
        await service.index_document_techniques(document_id, product_family=family)

        async with get_vector_store() as store:
            await store.upsert_chunks(
                document_id,
                [{"chunk_index": 0, "page_number": 1, "section_title": None, "heading_path": [],
                  "content": "A brochure.", "embedding": _unit_vector(0)}],
            )

        def no_model(*arguments):
            raise AssertionError("no step chunks, no call")

        stubbed_ingestion.setattr(technique_extraction, "get_chat_model", no_model)
        await service.index_document_techniques(document_id, product_family=family)

        assert await repository.list_techniques(document_id=document_id) == []
        assert _WRITTEN_PROCEDURES[str(document_id)] == []
    finally:
        await delete_document(document_id)
