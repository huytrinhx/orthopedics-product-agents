"""Extracts a document's technique/step outline from its step-bearing
vector chunks (issue #28): one model call per document.

The model only outlines: which techniques the document holds, their steps
in order, the chunks each step is in, and what each step says it uses
("Ø2.6mm drill bit for Ø3.7mm screws"). It never picks a SKU unless the
text literally prints one -- steps almost never name a SKU, they name an
instrument and a spec, and the SKUs sit in tray tables pages later.
Turning an item into a SKU set is backend/techniques/resolution.py's job,
in code, from the catalog and the document's SKU index.

The input is the vector chunks themselves (`chunk_index` kept), never
`_split_for_extraction` or page-marker-stripped text, so every step
points at the chunks an answer cites as `{document_id}#{chunk_index}`.
A chunk is step-bearing when its heading path (issue #26) has a step
heading in it (headings.is_step_heading).

Runs on the entity-extraction slot (issue #24): it's the same kind of
call, structured output over document text at ingestion.
"""
from collections.abc import Iterable

from pydantic import BaseModel, Field

from config.llm_clients import get_chat_model
from ingestion.headings import is_step_heading
from model_assignments.runtime import SlotModel, invoke_with_output_fallback
from model_assignments.slots import Slot

# The parameter names the prompt asks for, so a lookup's parameters and
# the stored items' agree across documents. Others are allowed when none
# fits.
PARAMETER_NAMES = (
    "screw_diameter",
    "screw_thread",
    "screw_length",
    "side",
    "bridge_size",
    "staple_type",
    "plate",
    "wire_diameter",
)
# The spec keys resolution.py reads. Others are kept and matched loosely.
SPEC_KEYS = (
    "diameter",
    "length",
    "width",
    "thread",
    "side",
    "driver",
    "for_screw_diameter",
    "for_screw_thread",
)

_SYSTEM_PROMPT = f"""You outline the surgical techniques in an orthopedic \
Surgical Technique document. You are given the document's step chunks, each \
labelled with its chunk number and heading path (technique > step).

For each technique, give:
- its name, as the document titles it (when a technique has no title in the \
text, a short name from what its steps do);
- `parameters`: the surgeon's choices the instruments and implants depend \
on, using these names where one fits: {", ".join(PARAMETER_NAMES)};
- `tray_references`: trays or systems the text says the technique requires, \
copied as written (e.g. "Core Foot Recon Plating System");
- its steps in order. Each step has its label as printed, its order (from \
1), the chunk numbers its text is in, and the items it uses.

For each item a step uses (instruments, wires, drills, drivers, guides, \
implants):
- `mention`: the words the text uses, copied verbatim;
- `item_type`: one of drill bit, guidewire, guide pin, driver, screw, \
plate, staple, implant, guide, drill guide, targeting guide, sleeve, \
countersink, depth gauge, tissue protector, inserter, tamp, curette, \
handle, arthrotome, drill pin, locator pin, burr, reamer, or another \
short instrument name;
- `sku`: only when the text literally prints the catalog number next to \
it; otherwise null. Never guess a catalog number;
- `specs`: the values the text gives, as name/value pairs, using these \
names where one fits: {", ".join(SPEC_KEYS)} (e.g. diameter = "2.6mm", \
for_screw_diameter = "3.7mm");
- `condition`: when the text limits its use ("if compression is desired", \
"for Ø3.7mm screws"), copied as written;
- `applies_when`: the parameter values this item is for, as name/value \
pairs using the parameter names above (e.g. screw_diameter = "3.7mm"). \
When the text gives one instrument per option ("Ø2.6mm for Ø3.7mm \
screws; Ø2.8mm for Ø4.1mm screws"), give one item per option, each with \
its own applies_when;
- `optional`: whether the text says it's optional.

Use only the given text. Include only steps that are in it, and only chunk \
numbers from the labels given.
"""


class NamedValue(BaseModel):
    """One spec or parameter value. A list of these rather than a dict:
    an open-keyed object isn't a schema every provider's structured
    output accepts."""

    name: str
    value: str


class StepItemReference(BaseModel):
    mention: str = Field(description="The words the step text uses for the item, verbatim")
    item_type: str = Field(description="drill bit | guidewire | driver | screw | plate | staple | guide | ...")
    sku: str | None = Field(default=None, description="Only when the text literally prints one")
    specs: list[NamedValue] = Field(default_factory=list)
    condition: str | None = None
    applies_when: list[NamedValue] = Field(default_factory=list)
    optional: bool = False

    def spec_values(self) -> dict[str, str]:
        return {spec.name: spec.value for spec in self.specs}

    def parameter_values(self) -> dict[str, str]:
        return {parameter.name: parameter.value for parameter in self.applies_when}


class TechniqueStep(BaseModel):
    step_label: str
    step_order: int
    source_chunk_indexes: list[int]
    items: list[StepItemReference] = Field(default_factory=list)


class Technique(BaseModel):
    name: str
    parameters: list[str] = Field(default_factory=list)
    tray_references: list[str] = Field(default_factory=list)
    steps: list[TechniqueStep] = Field(default_factory=list)


class TechniqueOutline(BaseModel):
    techniques: list[Technique] = Field(default_factory=list)


def is_step_chunk(heading_path: list[str] | None) -> bool:
    return any(is_step_heading(heading) for heading in heading_path or [])


def step_bearing_chunks(
    chunks: Iterable[tuple[int, list[str] | None, str]],
) -> list[tuple[int, list[str], str]]:
    """The (chunk_index, heading_path, content) chunks with a step heading
    in their path, in order."""
    return [
        (chunk_index, list(heading_path or []), content)
        for chunk_index, heading_path, content in chunks
        if is_step_chunk(heading_path)
    ]


def _format_chunks(chunks: list[tuple[int, list[str], str]]) -> str:
    return "\n\n".join(
        f"[chunk {chunk_index}] {' > '.join(heading_path)}\n{content}"
        for chunk_index, heading_path, content in chunks
    )


async def extract_technique_outline(
    chunks: list[tuple[int, list[str], str]],
    *,
    slot_model: SlotModel | None = None,
) -> TechniqueOutline:
    """The outline of the given step-bearing chunks (step_bearing_chunks'
    output). No chunks, no call. `slot_model` is the entity-extraction
    slot's model, resolved once per document; None runs the default.

    Only the shape is checked here; techniques/service.py drops chunk
    numbers that weren't given and steps left with none.
    """
    if not chunks:
        return TechniqueOutline()
    prompt = [
        ("system", _SYSTEM_PROMPT),
        ("user", f"Step chunks:\n\n{_format_chunks(chunks)}"),
    ]
    return await invoke_with_output_fallback(
        slot_model or SlotModel(Slot.ENTITY_EXTRACTION),
        get_chat_model,
        lambda chat_model: chat_model.with_structured_output(TechniqueOutline).ainvoke(prompt),
    )
