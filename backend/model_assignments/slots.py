"""The model slots (issue #24): each node that calls a chat model belongs
to exactly one slot, and an admin picks a model and reasoning effort per
slot from Settings. Slots are code, like question types
(agents/question_types.py): the call sites below are what reads them, and
each slot declares what its LangChain call needs from a model, which is
what the picker filters the OpenRouter catalog by (catalog.py) and what the
probe proves on save (probe.py).

An unset slot runs `CHAT_MODEL` at the model's own reasoning default:
`get_chat_model()` with no arguments, exactly as before slots existed.

A slot that accepts decision models (issue #23; only intent
classification) can hold either kind: a chat model, as above, or a
decision model (config/llm_clients.py's DecisionClient), which answers
Choices with probabilities instead of writing structured output. A decision
model's two thresholds are saved with it.
"""
from dataclasses import dataclass
from enum import StrEnum


class Slot(StrEnum):
    INTENT_CLASSIFICATION = "intent_classification"
    DETERMINISTIC_ANSWER = "deterministic_answer"
    REACT_AGENT = "react_agent"
    JUDGE = "judge"
    ENTITY_EXTRACTION = "entity_extraction"
    TRAY_LAYOUT_EXTRACTION = "tray_layout_extraction"


class ModelKind(StrEnum):
    CHAT = "chat"
    DECISION = "decision"


class Need(StrEnum):
    STRUCTURED_OUTPUT = "structured_output"
    STREAMED_TEXT = "streamed_text"
    TOOL_CALLING = "tool_calling"
    IMAGE_INPUT = "image_input"


# What `reasoning_effort` may be set to. "none" turns reasoning off on a
# model that supports it; an unset effort leaves the model's own default.
REASONING_EFFORTS = ("none", "low", "medium", "high")

# A decision model's thresholds when none were saved (issue #23). Starting
# points, not tuned values: tune both on the intent eval
# (`python -m evals.harness intent --model intent_classification=<id>
# --family-confidence-threshold ... --question-type-margin ...`).
# The family pick needs at least this probability, or the rep is asked.
DEFAULT_FAMILY_CONFIDENCE_THRESHOLD = 0.6
# Two question types whose probabilities are within this of each other
# both count as detected, so the rep is asked which they mean (issue #19).
DEFAULT_QUESTION_TYPE_MARGIN = 0.15


@dataclass(frozen=True)
class SlotDefinition:
    slot: Slot
    label: str
    call_sites: str
    needs: tuple[Need, ...]
    # Pre-filled in the effort picker the first time a slot is saved. Not
    # applied to an unset slot, which keeps the model's own default.
    default_reasoning_effort: str
    # Shown beside the slot in Settings.
    warning: str | None = None
    # Whether a decision model may run this slot instead of a chat model.
    accepts_decision_models: bool = False


SLOT_DEFINITIONS: dict[Slot, SlotDefinition] = {
    definition.slot: definition
    for definition in (
        SlotDefinition(
            Slot.INTENT_CLASSIFICATION,
            "Intent classification",
            "deterministic._classify_intent, _classify_question_type_reply",
            (Need.STRUCTURED_OUTPUT,),
            "none",
            accepts_decision_models=True,
        ),
        SlotDefinition(
            Slot.DETERMINISTIC_ANSWER,
            "Deterministic answer",
            "deterministic.generate",
            (Need.STREAMED_TEXT,),
            "low",
        ),
        SlotDefinition(
            Slot.REACT_AGENT,
            "ReAct agent",
            "react_agent.generate",
            (Need.TOOL_CALLING,),
            "medium",
        ),
        SlotDefinition(
            Slot.JUDGE,
            "Judge",
            "agents/judge.py",
            (Need.STRUCTURED_OUTPUT,),
            "low",
            warning=(
                "Changing the judge's model breaks score comparison with runs scored "
                "before the change."
            ),
        ),
        SlotDefinition(
            Slot.ENTITY_EXTRACTION,
            "Entity extraction",
            "ingestion/entity_extraction.py, ingestion/technique_extraction.py",
            (Need.STRUCTURED_OUTPUT,),
            "low",
        ),
        SlotDefinition(
            Slot.TRAY_LAYOUT_EXTRACTION,
            "Tray-layout extraction",
            "ingestion/tray_layout_extraction.py",
            (Need.STRUCTURED_OUTPUT, Need.IMAGE_INPUT),
            "low",
        ),
    )
}

# The slots a chat turn runs, and so the only ones the Eval tab's rerun can
# override. The judge and the extraction slots never run inside a turn.
CHAT_SLOTS = (Slot.INTENT_CLASSIFICATION, Slot.DETERMINISTIC_ANSWER, Slot.REACT_AGENT)
