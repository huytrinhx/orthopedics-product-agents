"""Raw SQL against the `techniques`, `technique_steps` and
`technique_step_items` tables
(backend/migrations/versions/a5c8e2f17b94_create_technique_tables.py).

A document's techniques are written as a whole and replaced as a whole:
replace_document_techniques deletes them (steps and items cascade) and
inserts the new extraction in one transaction, the document_skus pattern.
"""
import uuid
from dataclasses import dataclass, field

from psycopg.types.json import Jsonb

from config.db import get_connection


@dataclass(frozen=True)
class StepItemRow:
    mention: str
    item_type: str
    literal_sku: str | None
    specs: dict[str, str]
    condition: str | None
    optional: bool
    applies_when: dict[str, str]
    resolved_skus: list[str]
    resolution: str
    source_chunk_indexes: list[int]


@dataclass(frozen=True)
class StepRow:
    step_label: str
    source_chunk_indexes: list[int]
    items: list[StepItemRow] = field(default_factory=list)


@dataclass(frozen=True)
class TechniqueRow:
    name: str
    parameters: list[str]
    required_trays: list[str]
    steps: list[StepRow] = field(default_factory=list)


@dataclass
class TechniqueRecord:
    """A stored technique with its steps (in order) and the document it's
    from. Step and item order are list positions, from the stored
    step_order/item_order."""

    document_id: uuid.UUID
    filename: str
    product_family: str | None
    technique_order: int
    name: str
    parameters: list[str]
    required_trays: list[str]
    steps: list[StepRow] = field(default_factory=list)


async def replace_document_techniques(
    document_id: uuid.UUID | str, techniques: list[TechniqueRow]
) -> None:
    connection = await get_connection()
    try:
        async with connection.cursor() as cursor:
            await cursor.execute("DELETE FROM techniques WHERE document_id = %s", (document_id,))
            for technique_order, technique in enumerate(techniques, start=1):
                await cursor.execute(
                    """
                    INSERT INTO techniques (document_id, technique_order, name, parameters, required_trays)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        document_id,
                        technique_order,
                        technique.name,
                        technique.parameters,
                        technique.required_trays,
                    ),
                )
                (technique_id,) = await cursor.fetchone()
                for step_order, step in enumerate(technique.steps, start=1):
                    await cursor.execute(
                        """
                        INSERT INTO technique_steps (technique_id, step_order, step_label, source_chunk_indexes)
                        VALUES (%s, %s, %s, %s)
                        RETURNING id
                        """,
                        (technique_id, step_order, step.step_label, step.source_chunk_indexes),
                    )
                    (step_id,) = await cursor.fetchone()
                    if not step.items:
                        continue
                    await cursor.executemany(
                        """
                        INSERT INTO technique_step_items
                            (step_id, item_order, mention, item_type, literal_sku, specs,
                             condition, optional, applies_when, resolved_skus, resolution,
                             source_chunk_indexes)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        [
                            (
                                step_id,
                                item_order,
                                item.mention,
                                item.item_type,
                                item.literal_sku,
                                Jsonb(item.specs),
                                item.condition,
                                item.optional,
                                Jsonb(item.applies_when),
                                item.resolved_skus,
                                item.resolution,
                                item.source_chunk_indexes,
                            )
                            for item_order, item in enumerate(step.items, start=1)
                        ],
                    )
        await connection.commit()
    finally:
        await connection.close()


async def list_techniques(
    *, document_id: uuid.UUID | str | None = None, product_family: str | None = None
) -> list[TechniqueRecord]:
    """Stored techniques with their steps and items, for one document or
    every document tagged with one family (its catalog name), in document
    then technique order. Both None reads every document's."""
    connection = await get_connection()
    try:
        async with connection.cursor() as cursor:
            await cursor.execute(
                """
                SELECT t.document_id, d.filename, pf.name, t.technique_order, t.name,
                       t.parameters, t.required_trays,
                       s.step_order, s.step_label, s.source_chunk_indexes,
                       i.item_order, i.mention, i.item_type, i.literal_sku, i.specs,
                       i.condition, i.optional, i.applies_when, i.resolved_skus,
                       i.resolution, i.source_chunk_indexes
                FROM techniques t
                JOIN documents d ON d.id = t.document_id
                LEFT JOIN product_families pf ON pf.id = d.product_family_id
                LEFT JOIN technique_steps s ON s.technique_id = t.id
                LEFT JOIN technique_step_items i ON i.step_id = s.id
                WHERE (%(document_id)s::uuid IS NULL OR t.document_id = %(document_id)s::uuid)
                  AND (%(product_family)s::text IS NULL OR pf.name = %(product_family)s::text)
                ORDER BY d.filename, t.document_id, t.technique_order, s.step_order, i.item_order
                """,
                {"document_id": document_id, "product_family": product_family},
            )
            rows = await cursor.fetchall()
    finally:
        await connection.close()

    techniques: dict[tuple, TechniqueRecord] = {}
    steps: dict[tuple, StepRow] = {}
    for row in rows:
        (
            row_document_id, filename, family, technique_order, name, parameters, required_trays,
            step_order, step_label, step_chunks,
            item_order, mention, item_type, literal_sku, specs, condition, optional, applies_when,
            resolved_skus, resolution, item_chunks,
        ) = row
        technique_key = (row_document_id, technique_order)
        if technique_key not in techniques:
            techniques[technique_key] = TechniqueRecord(
                document_id=row_document_id,
                filename=filename,
                product_family=family,
                technique_order=technique_order,
                name=name,
                parameters=list(parameters or []),
                required_trays=list(required_trays or []),
            )
        if step_order is None:
            continue
        step_key = (row_document_id, technique_order, step_order)
        if step_key not in steps:
            steps[step_key] = StepRow(step_label=step_label, source_chunk_indexes=list(step_chunks))
            techniques[technique_key].steps.append(steps[step_key])
        if item_order is None:
            continue
        steps[step_key].items.append(
            StepItemRow(
                mention=mention,
                item_type=item_type,
                literal_sku=literal_sku,
                specs=dict(specs or {}),
                condition=condition,
                optional=optional,
                applies_when=dict(applies_when or {}),
                resolved_skus=list(resolved_skus or []),
                resolution=resolution,
                source_chunk_indexes=list(item_chunks or []),
            )
        )
    return list(techniques.values())
