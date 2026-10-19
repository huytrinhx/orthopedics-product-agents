"""Pydantic shapes for a document's stored techniques (issue #28): the
admin read (GET /documents/{document_id}/techniques) and the
`technique_steps` lookup an answer will be built from (issue #29)."""
import uuid

from pydantic import BaseModel


class StepSkuOut(BaseModel):
    sku: str
    # From the catalog, never the document text.
    description: str | None


class StepItemOut(BaseModel):
    mention: str
    item_type: str
    condition: str | None
    optional: bool
    specs: dict[str, str]
    # The parameter values this item's set is for; empty when it's used
    # whatever the surgeon picks.
    applies_when: dict[str, str]
    # techniques/resolution.py's LITERAL / IN_DOCUMENT / CATALOG_ONLY /
    # UNRESOLVED.
    resolution: str
    skus: list[StepSkuOut]
    # `{document_id}#{chunk_index}`: the step's chunks, then the chunks the
    # document's SKU index places the SKUs in.
    citations: list[str]


class TechniqueStepOut(BaseModel):
    step_order: int
    step_label: str
    citations: list[str]
    items: list[StepItemOut]


class TechniqueOut(BaseModel):
    document_id: uuid.UUID
    filename: str
    product_family: str | None
    name: str
    parameters: list[str]
    required_trays: list[str]
    # Parameters some item depends on that the lookup wasn't given: each
    # option's item is in `steps`, so the caller lists them or asks.
    unbound_parameters: list[str]
    steps: list[TechniqueStepOut]
