"""Extracts part differentiations from document text and writes them into
the AuraDB graph (backend/retrieval/graph_client.py) — the graph is
populated at ingestion time, not query time.

This is the per-document, LLM-based path (path 2 in backend/retrieval/
graph_client.py's module docstring) — it only *attaches* facts to `Part`
nodes that already exist from the deterministic master-catalog seed
(backend/ingestion/seed_master_catalog.py, path 1). It never mints a new
`Part` node from a bare name mention: the model is given the real SKU
catalog for the document's scope and told to reference only SKUs from that
list, and backend/retrieval/graph_client.py's attach_differentiation
additionally no-ops if a referenced SKU isn't already a real node — a
hallucinated reference is silently dropped rather than merged in some fuzzy
way.

Narrowed by issue #28. The 2026-10-18 evaluation found its output mostly
restated catalog descriptions ("MSCF3020 is MIS FT, MSCP3020 is MIS PT")
and changed entirely between two runs on the same document, so the prompt
now asks only for a difference the text itself explains: a reason or rule
for choosing one part over the other, never an attribute the catalog
already holds. Procedure requirements moved to the technique outline
(backend/ingestion/technique_extraction.py), which keys them per document
and technique.
"""
from pydantic import BaseModel, Field

from config.llm_clients import get_chat_model
from model_assignments.runtime import SlotModel, invoke_with_output_fallback
from model_assignments.slots import Slot

_SYSTEM_PROMPT = """You extract part differentiations from orthopedic \
product documentation (brochures, surgical technique guides, launch \
presentations): places where the text itself explains when to use one \
specific part rather than another, or why they differ (e.g. "use the 1.1mm \
wire sleeve with 1.1mm wires; the 1.4mm sleeve does not hold them"). For \
each, extract the SKUs of both parts and the explanation, using only the \
provided text.

Do NOT extract a difference that only restates what the parts' catalog \
descriptions already say (size, length, thread type, side, material): \
"MSCF3020 is full thread and MSCP3020 is partial thread" is not a \
differentiation. Extract one only when the text gives a reason, rule or \
consequence.

You may ONLY reference a SKU that appears in the "Known parts" list below, \
exactly as given. If you cannot confidently match a mention to one specific \
known SKU, omit it rather than guessing. Extract nothing if the text \
explains no difference.
"""


class ExtractedDifferentiation(BaseModel):
    sku_a: str = Field(description="SKU of the first part, copied exactly from the known-parts list")
    sku_b: str = Field(description="SKU of the second part, copied exactly from the known-parts list")
    explanation: str = Field(description="The reason or rule the text gives, based only on the given text")


class _ExtractionOutput(BaseModel):
    differentiations: list[ExtractedDifferentiation] = Field(default_factory=list)


async def extract_entities(
    chunk_text: str,
    *,
    known_parts: list[dict],
    slot_model: SlotModel | None = None,
) -> list[dict]:
    """Returns a list of dicts, each

        {"type": "differentiation", "sku_a": ..., "sku_b": ..., "explanation": ...}

    `known_parts` is `[{"sku": ..., "description": ...}, ...]`, scoped to
    the document's family plus the families its SKU index found parts of
    (backend/ingestion/pipeline.py), giving the model a closed set of valid
    SKUs to reference. Entries referencing anything outside that set are
    dropped here as a defensive check — the graph write layer also refuses
    them, but failing fast avoids a wasted write attempt.

    `slot_model` is the entity-extraction slot's model (issue #24), which
    ingest_document resolves once per document; None runs the default.
    """
    if not known_parts:
        return []

    known_skus = {part["sku"] for part in known_parts}
    catalog_listing = "\n".join(f"- {part['sku']}: {part['description']}" for part in known_parts)

    prompt = [
        ("system", _SYSTEM_PROMPT),
        ("user", f"Known parts:\n{catalog_listing}\n\nText:\n{chunk_text}"),
    ]
    result = await invoke_with_output_fallback(
        slot_model or SlotModel(Slot.ENTITY_EXTRACTION),
        get_chat_model,
        lambda chat_model: chat_model.with_structured_output(_ExtractionOutput).ainvoke(prompt),
    )

    return [
        {
            "type": "differentiation",
            "sku_a": differentiation.sku_a,
            "sku_b": differentiation.sku_b,
            "explanation": differentiation.explanation,
        }
        for differentiation in result.differentiations
        if differentiation.sku_a in known_skus
        and differentiation.sku_b in known_skus
        and differentiation.sku_a != differentiation.sku_b
    ]
