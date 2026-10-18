"""Which model each slot runs right now, and the runtime output fallback.

A chat turn resolves every chat slot once, at its start (`resolve_turn_slot_models`,
called by api/routes/chat.py): the saved settings through the repository's
30-second cache, then the conversation's admin overrides on top (the Eval
tab's rerun). The result rides in the run's LangGraph config, so a node
reads its slot with `slot_model_for_run` instead of the database. The
judge and ingestion run outside a turn and read theirs with
`configured_slot_model`.

Without `OPENROUTER_API_KEY` every slot runs the default, whatever is
saved: the Settings section is read-only then, and a saved non-OpenAI model
couldn't run on OpenAI direct anyway.
"""
import logging
import os
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, TypeVar

from langchain_core.exceptions import OutputParserException
from langgraph.config import get_config
from pydantic import ValidationError

from config.llm_clients import configured_chat_model
from model_assignments.repository import cached_assignments
from model_assignments.slots import CHAT_SLOTS, Slot

logger = logging.getLogger(__name__)

# The key under the run config's "configurable" that carries a turn's
# resolved slots.
SLOT_MODELS_CONFIG_KEY = "slot_models"

ResultType = TypeVar("ResultType")


@dataclass(frozen=True)
class SlotModel:
    """One slot's model for one call. `model_id` None means the default:
    `CHAT_MODEL` at the model's own reasoning default. `source` says where
    the choice came from: "default", "setting" or "override".
    """

    slot: Slot
    model_id: str | None = None
    reasoning_effort: str | None = None
    source: str = "default"

    @property
    def is_default(self) -> bool:
        return self.model_id is None

    @property
    def chat_model_arguments(self) -> tuple:
        """What to pass to `get_chat_model`. Nothing at all for the default,
        so an unset slot makes exactly the call it made before slots
        existed.
        """
        return () if self.model_id is None else (self.model_id, self.reasoning_effort)

    def as_config(self) -> dict:
        """Plain data for the run config, which LangGraph may copy into
        checkpoint metadata.
        """
        return {
            "model_id": self.model_id,
            "reasoning_effort": self.reasoning_effort,
            "source": self.source,
        }

    def describe(self) -> dict:
        """For logs and the rerun's "ran on" line: the model actually
        called, with the default spelled out as CHAT_MODEL.
        """
        return {
            "model_id": self.model_id or configured_chat_model(),
            "reasoning_effort": self.reasoning_effort,
            "source": self.source,
        }


def gateway_configured() -> bool:
    return bool(os.environ.get("OPENROUTER_API_KEY"))


async def configured_slot_model(slot: Slot) -> SlotModel:
    """A slot's saved setting, or the default when it's unset or there's
    no gateway.
    """
    if not gateway_configured():
        return SlotModel(slot)
    assignment = (await cached_assignments()).get(slot.value)
    if assignment is None:
        return SlotModel(slot)
    return SlotModel(slot, assignment.model_id, assignment.reasoning_effort, "setting")


async def resolve_turn_slot_models(
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, SlotModel]:
    """Every chat slot for one turn: saved settings, then `overrides` (the
    conversation's admin-only per-slot choices, slot -> {model_id,
    reasoning_effort}) on top.
    """
    slot_models = {slot.value: await configured_slot_model(slot) for slot in CHAT_SLOTS}
    if not gateway_configured():
        return slot_models
    for slot_name, choice in (overrides or {}).items():
        slot = Slot(slot_name)
        slot_models[slot.value] = SlotModel(
            slot, choice["model_id"], choice.get("reasoning_effort"), "override"
        )
    return slot_models


def slot_model_for_run(slot: Slot) -> SlotModel:
    """The slot's model for the graph run this is called from. Outside a run
    (a test calling a node directly), or in a run that carries no slots,
    that's the default.
    """
    try:
        configurable = get_config().get("configurable") or {}
    except RuntimeError:
        return SlotModel(slot)
    choice = (configurable.get(SLOT_MODELS_CONFIG_KEY) or {}).get(slot.value)
    if not choice:
        return SlotModel(slot)
    return SlotModel(slot, choice["model_id"], choice["reasoning_effort"], choice["source"])


def slot_models_config(slot_models: Mapping[str, SlotModel]) -> dict:
    """The run-config entry slot_model_for_run reads back."""
    return {
        SLOT_MODELS_CONFIG_KEY: {
            slot_name: slot_model.as_config() for slot_name, slot_model in slot_models.items()
        }
    }


def log_slot_models(event: str, slot_models: Mapping[str, SlotModel], **context: Any) -> None:
    """One structured line naming the model and effort each slot ran on --
    per chat turn, per ingested document.
    """
    described = {slot_name: slot_model.describe() for slot_name, slot_model in slot_models.items()}
    summary = " ".join(
        f"{slot_name}={description['model_id']}@{description['reasoning_effort'] or 'default'}"
        for slot_name, description in described.items()
    )
    context_text = " ".join(f"{key}={value}" for key, value in context.items())
    logger.info(
        "%s %s %s",
        event,
        context_text,
        summary,
        extra={"event": event, **context, "slot_models": described},
    )


# A model output the node can't use: structured output that didn't parse
# or didn't validate against the node's schema.
OUTPUT_FAILURES: tuple[type[Exception], ...] = (OutputParserException, ValidationError)


def _always_usable(result: Any) -> bool:
    return True


async def invoke_with_output_fallback(
    slot_model: SlotModel,
    get_chat_model: Callable[..., Any],
    call: Callable[[Any], Awaitable[ResultType]],
    is_usable: Callable[[ResultType], bool] = _always_usable,
) -> ResultType:
    """Runs `call` on the slot's model, and if its output is unusable --
    an OUTPUT_FAILURES exception, a None result, or `is_usable` saying no --
    runs it once more on `CHAT_MODEL` and logs `slot_output_fallback`. A
    default slot is already on `CHAT_MODEL`, so it gets no retry and fails
    exactly as it did before slots existed.

    `get_chat_model` is the caller's own module-level reference, so a test
    that patches it there patches both calls. This fallback is separate from
    config/llm_clients.py's gateway-outage fallback, which retries the same
    call on OpenAI direct and never looks at the output.
    """
    try:
        result = await call(get_chat_model(*slot_model.chat_model_arguments))
    except OUTPUT_FAILURES as error:
        if slot_model.is_default:
            raise
        reason = f"{type(error).__name__}: {str(error)[:200]}"
    else:
        if slot_model.is_default or (result is not None and is_usable(result)):
            return result
        reason = "no output" if result is None else "unusable output"

    logger.warning(
        "slot_output_fallback slot=%s model=%s reasoning_effort=%s fallback_model=%s reason=%s",
        slot_model.slot.value,
        slot_model.model_id,
        slot_model.reasoning_effort,
        configured_chat_model(),
        reason,
        extra={
            "event": "slot_output_fallback",
            "slot": slot_model.slot.value,
            "model": slot_model.model_id,
            "reasoning_effort": slot_model.reasoning_effort,
            "fallback_model": configured_chat_model(),
            "reason": reason,
        },
    )
    return await call(get_chat_model())
