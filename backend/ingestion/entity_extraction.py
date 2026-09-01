"""Extracts entities/relationships from chunks and writes them into the
AuraDB graph (backend/retrieval/graph_client.py) — the graph is populated
at ingestion time, not query time.

This is the per-document, LLM-based path (path 2 in backend/retrieval/
graph_client.py's module docstring) — it only *attaches* facts to `Part`/
`Tray` nodes that already exist from the deterministic master-catalog seed
(backend/ingestion/seed_master_catalog.py, path 1). It never mints a new
`Part` node from a bare name mention: the model is given the real SKU
catalog for the document's tagged system and told to reference only SKUs/
trays from that list, and backend/retrieval/graph_client.py's
attach_differentiation/attach_procedure additionally no-op if a referenced
SKU or tray isn't already a real node — a hallucinated reference is silently
dropped rather than merged in some fuzzy way.
"""
from pydantic import BaseModel, Field

from config.llm_clients import get_chat_model

_SYSTEM_PROMPT = """You extract two kinds of facts from orthopedic product \
documentation (brochures, surgical technique guides, launch presentations):

1. Differentiation: when the text explains how to tell two specific parts \
apart (e.g. two similarly-sized wires, screws, or staples with a different \
indication or thread type), extract the SKUs of both parts and a short \
explanation of the difference, using only the provided text.
2. Procedure requirement: when the text says a named surgical procedure \
requires a specific tray/instrument set, extract the procedure name and the \
tray name.

You may ONLY reference a SKU that appears in the "Known parts" list below, \
and a tray name that appears in the "Known trays" list below, exactly as \
given. If the text discusses a part or tray not in these lists, or you \
cannot confidently match a mention to one specific known SKU, omit it \
rather than guessing. Extract nothing if the text contains neither kind of \
fact.
"""


class ExtractedDifferentiation(BaseModel):
    sku_a: str = Field(description="SKU of the first part, copied exactly from the known-parts list")
    sku_b: str = Field(description="SKU of the second part, copied exactly from the known-parts list")
    explanation: str = Field(description="How these two parts differ, based only on the given text")


class ExtractedProcedureRequirement(BaseModel):
    procedure: str = Field(description="Name of the surgical procedure mentioned in the text")
    tray: str = Field(description="Tray name required for it, copied exactly from the known-trays list")


class _ExtractionOutput(BaseModel):
    differentiations: list[ExtractedDifferentiation] = Field(default_factory=list)
    procedure_requirements: list[ExtractedProcedureRequirement] = Field(default_factory=list)


async def extract_entities(
    chunk_text: str,
    *,
    known_parts: list[dict],
    known_trays: list[str],
) -> list[dict]:
    """Returns a list of dicts, each one of:

        {"type": "differentiation", "sku_a": ..., "sku_b": ..., "explanation": ...}
        {"type": "procedure_requirement", "procedure": ..., "tray": ...}

    `known_parts` is `[{"sku": ..., "description": ...}, ...]` and
    `known_trays` a list of tray names — both scoped to the document's
    tagged system (see backend/retrieval/graph_client.py's
    list_parts_for_family/list_trays_for_family), keeping the prompt small
    and giving the model a closed set of valid SKUs/trays to reference.
    Entries referencing anything outside those sets are dropped here as a
    defensive check — the graph write layer also refuses them, but failing
    fast avoids a wasted write attempt.
    """
    if not known_parts:
        return []

    known_skus = {part["sku"] for part in known_parts}
    known_tray_set = set(known_trays)

    catalog_listing = "\n".join(f"- {part['sku']}: {part['description']}" for part in known_parts)
    trays_listing = "\n".join(f"- {tray}" for tray in known_trays)

    model = get_chat_model().with_structured_output(_ExtractionOutput)
    result = await model.ainvoke(
        [
            ("system", _SYSTEM_PROMPT),
            (
                "user",
                (
                    f"Known parts:\n{catalog_listing}\n\n"
                    f"Known trays:\n{trays_listing}\n\n"
                    f"Text:\n{chunk_text}"
                ),
            ),
        ]
    )

    extracted: list[dict] = []
    for diff in result.differentiations:
        if diff.sku_a in known_skus and diff.sku_b in known_skus and diff.sku_a != diff.sku_b:
            extracted.append(
                {
                    "type": "differentiation",
                    "sku_a": diff.sku_a,
                    "sku_b": diff.sku_b,
                    "explanation": diff.explanation,
                }
            )
    for req in result.procedure_requirements:
        if req.tray in known_tray_set:
            extracted.append(
                {"type": "procedure_requirement", "procedure": req.procedure, "tray": req.tray}
            )
    return extracted
