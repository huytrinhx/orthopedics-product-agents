"""A document's surgical techniques and their steps, with each step's SKU
sets (issue #28).

`index_document_techniques` runs at ingestion, after the vector leg and
the SKU index (backend/document_skus/), because it reads both:

1. The step-bearing chunks (those whose heading path has a step heading)
   go to one outline call (backend/ingestion/technique_extraction.py).
2. The outline is checked against what was given: a step's chunk numbers
   must be ones it was sent, and a step left with none is dropped, so
   every stored step and item cites a real `{document_id}#{chunk_index}`.
3. Each technique's tray references are matched to catalog trays, and the
   document's scope is its family, the families of the SKUs its index
   found, and the families of those trays (techniques/resolution.py).
4. Each item resolves to a SKU set in code (resolution.py), and the whole
   document's techniques replace what was stored (repository.py). The
   graph gets one Procedure per technique, keyed by document and name,
   REQUIRES-linked to its trays.

`technique_steps(family, technique, parameters)` is the read an answer is
built from (issue #29): the ordered steps, each item's SKU set for the
given parameters with catalog descriptions, and citation ids.
"""
import logging
import os
import re
import uuid
from collections.abc import Mapping

from document_skus.repository import list_sku_chunk_indexes
from documents.repository import list_chunks
from ingestion.technique_extraction import (
    TechniqueOutline,
    extract_technique_outline,
    step_bearing_chunks,
)
from model_assignments.runtime import configured_slot_model, log_slot_models
from model_assignments.slots import Slot
from retrieval.graph_client import get_graph_client
from techniques import repository
from techniques.models import StepItemOut, StepSkuOut, TechniqueOut, TechniqueStepOut
from techniques.repository import StepItemRow, StepRow, TechniqueRecord, TechniqueRow
from techniques.resolution import (
    UNRESOLVED,
    CatalogPart,
    ResolutionContext,
    applies_to,
    normalize_parameter_name,
    resolve_step_items,
    resolve_tray_references,
    scope_families,
)

logger = logging.getLogger(__name__)


async def load_resolution_catalog() -> dict[str, CatalogPart]:
    return {
        part["sku"]: CatalogPart(
            sku=part["sku"],
            description=part.get("description") or "",
            families=frozenset(part.get("families") or []),
            item_type=part.get("item_type"),
            thread=part.get("thread"),
            guidewire_spec=part.get("guidewire_spec"),
            pre_drill_spec=part.get("pre_drill_spec"),
            driver_spec=part.get("driver_spec"),
        )
        for part in await get_graph_client().list_catalog_parts()
        if part.get("sku")
    }


def build_technique_rows(
    outline: TechniqueOutline,
    *,
    given_chunk_indexes: set[int],
    catalog: Mapping[str, CatalogPart],
    document_family: str | None,
    document_sku_chunks: Mapping[str, list[int]],
    trays_by_family: Mapping[str, list[str]],
) -> list[TechniqueRow]:
    """The outline, checked and resolved, as the rows to store. Pure."""
    family_by_tray = {tray: family for family, trays in trays_by_family.items() for tray in trays}
    required_trays = [
        resolve_tray_references(technique.tray_references, trays_by_family)
        for technique in outline.techniques
    ]
    context = ResolutionContext(
        catalog=catalog,
        scope=scope_families(
            document_family,
            document_sku_chunks,
            catalog,
            {family_by_tray[tray] for trays in required_trays for tray in trays},
        ),
        document_sku_chunks=document_sku_chunks,
    )

    rows: list[TechniqueRow] = []
    for technique, trays in zip(outline.techniques, required_trays):
        steps: list[StepRow] = []
        for step in sorted(technique.steps, key=lambda step: step.step_order):
            chunk_indexes = list(
                dict.fromkeys(index for index in step.source_chunk_indexes if index in given_chunk_indexes)
            )
            if not chunk_indexes:
                continue
            resolved_items = resolve_step_items(step.items, context, chunk_indexes)
            steps.append(
                StepRow(
                    step_label=step.step_label.strip(),
                    source_chunk_indexes=chunk_indexes,
                    items=[
                        StepItemRow(
                            mention=item.mention,
                            item_type=item.item_type,
                            literal_sku=item.sku or None,
                            specs=item.spec_values(),
                            condition=item.condition,
                            optional=item.optional,
                            applies_when=item.parameter_values(),
                            resolved_skus=list(resolved.skus),
                            resolution=resolved.resolution,
                            source_chunk_indexes=list(resolved.source_chunk_indexes),
                        )
                        for item, resolved in zip(step.items, resolved_items)
                    ],
                )
            )
        if steps and technique.name.strip():
            rows.append(
                TechniqueRow(
                    name=technique.name.strip(),
                    parameters=list(dict.fromkeys(technique.parameters)),
                    required_trays=trays,
                    steps=steps,
                )
            )
    return rows


async def index_document_techniques(
    document_id: uuid.UUID | str, *, product_family: str | None
) -> None:
    """Rebuilds one document's techniques from the chunks and SKU index it
    has stored now. A document with no step-bearing chunks makes no model
    call and ends up with no techniques."""
    if not os.environ.get("OPENAI_API_KEY"):
        # The same environment-config gap the other model-calling legs skip
        # on (e.g. CI without the secret).
        return
    chunks = step_bearing_chunks(
        (chunk.chunk_index, chunk.heading_path, chunk.content)
        for chunk in await list_chunks(uuid.UUID(str(document_id)))
    )
    if chunks:
        # Read once per document (issue #24), like the other extraction legs.
        slot_model = await configured_slot_model(Slot.ENTITY_EXTRACTION)
        log_slot_models(
            "ingestion_models",
            {Slot.ENTITY_EXTRACTION.value: slot_model},
            document_id=str(document_id),
        )
        outline = await extract_technique_outline(chunks, slot_model=slot_model)
        graph = get_graph_client()
        rows = build_technique_rows(
            outline,
            given_chunk_indexes={chunk_index for chunk_index, _, _ in chunks},
            catalog=await load_resolution_catalog(),
            document_family=product_family,
            document_sku_chunks=await list_sku_chunk_indexes(document_id),
            trays_by_family=await graph.list_trays_by_family(),
        )
    else:
        rows = []

    await repository.replace_document_techniques(document_id, rows)
    await get_graph_client().replace_document_procedures(
        str(document_id), [{"name": row.name, "trays": row.required_trays} for row in rows]
    )

    items = [item for row in rows for step in row.steps for item in step.items]
    unresolved = sorted({item.mention for item in items if item.resolution == UNRESOLVED})
    logger.info(
        "technique_extraction document_id=%s step_chunks=%d techniques=%d steps=%d items=%d "
        "unresolved=%d",
        document_id,
        len(chunks),
        len(rows),
        sum(len(row.steps) for row in rows),
        len(items),
        len(unresolved),
        extra={
            "event": "technique_extraction",
            "document_id": str(document_id),
            "step_chunks": len(chunks),
            "techniques": len(rows),
            "steps": sum(len(row.steps) for row in rows),
            "items": len(items),
            "unresolved": unresolved,
        },
    )


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def technique_matches(name: str, query: str | None) -> bool:
    """Every word of `query` is a word of the technique's name, case
    ignored: "tetra ultra" finds "TETRA Staple and ULTRA staple"."""
    return query is None or _words(query) <= _words(name)


def _citations(document_id: uuid.UUID, chunk_indexes: list[int]) -> list[str]:
    return [f"{document_id}#{chunk_index}" for chunk_index in chunk_indexes]


def _technique_out(
    record: TechniqueRecord, parameters: Mapping[str, str], descriptions: Mapping[str, str | None]
) -> TechniqueOut:
    given = {normalize_parameter_name(name) for name in parameters}
    depended_on = [normalize_parameter_name(name) for name in record.parameters] + [
        normalize_parameter_name(name)
        for step in record.steps
        for item in step.items
        for name in item.applies_when
    ]
    return TechniqueOut(
        document_id=record.document_id,
        filename=record.filename,
        product_family=record.product_family,
        name=record.name,
        parameters=record.parameters,
        required_trays=record.required_trays,
        unbound_parameters=[name for name in dict.fromkeys(depended_on) if name not in given],
        steps=[
            TechniqueStepOut(
                step_order=step_order,
                step_label=step.step_label,
                citations=_citations(record.document_id, step.source_chunk_indexes),
                items=[
                    StepItemOut(
                        mention=item.mention,
                        item_type=item.item_type,
                        condition=item.condition,
                        optional=item.optional,
                        specs=item.specs,
                        applies_when=item.applies_when,
                        resolution=item.resolution,
                        skus=[
                            StepSkuOut(sku=sku, description=descriptions.get(sku))
                            for sku in item.resolved_skus
                        ],
                        citations=_citations(record.document_id, item.source_chunk_indexes),
                    )
                    for item in step.items
                    if applies_to(item.applies_when, parameters)
                ],
            )
            for step_order, step in enumerate(record.steps, start=1)
        ],
    )


async def _descriptions(records: list[TechniqueRecord]) -> dict[str, str | None]:
    skus = sorted(
        {sku for record in records for step in record.steps for item in step.items for sku in item.resolved_skus}
    )
    if not skus:
        return {}
    return {
        part["sku"]: part.get("description")
        for part in await get_graph_client().list_catalog_parts(skus=skus)
    }


async def technique_steps(
    product_family: str | None,
    technique: str | None = None,
    parameters: Mapping[str, str] | None = None,
) -> list[TechniqueOut]:
    """The stored techniques of every document tagged `product_family` (its
    catalog name) whose name has every word of `technique`, each with its
    ordered steps and, per step, only the items that apply to
    `parameters` (e.g. {"screw_diameter": "3.7"}). An item bound to a
    parameter that isn't given is kept, and the parameter is listed in
    `unbound_parameters`."""
    parameters = dict(parameters or {})
    records = [
        record
        for record in await repository.list_techniques(product_family=product_family)
        if technique_matches(record.name, technique)
    ]
    descriptions = await _descriptions(records)
    return [_technique_out(record, parameters, descriptions) for record in records]


async def list_document_techniques(document_id: uuid.UUID) -> list[TechniqueOut]:
    """One document's techniques with every item, for review."""
    records = await repository.list_techniques(document_id=document_id)
    descriptions = await _descriptions(records)
    return [_technique_out(record, {}, descriptions) for record in records]
