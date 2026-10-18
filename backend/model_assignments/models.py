"""Pydantic request/response shapes for the per-slot model settings
(api/routes/model_assignments.py) and the chat request's admin-only
overrides (api/routes/chat.py).
"""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

ReasoningEffort = Literal["none", "low", "medium", "high"]
ModelKindValue = Literal["chat", "decision"]


class ModelChoice(BaseModel):
    """A model for one slot: an OpenRouter model id, and a reasoning effort
    or None for the model's own default.

    `model_kind` "decision" (issue #23) makes it a decision model instead,
    which takes no reasoning effort but has two thresholds; None for either
    means slots.py's default. Whether the slot accepts a decision model is
    the route's check, since only it knows the slot.
    """

    model_id: str = Field(min_length=1)
    reasoning_effort: ReasoningEffort | None = None
    model_kind: ModelKindValue = "chat"
    family_confidence_threshold: float | None = Field(default=None, gt=0, le=1)
    question_type_margin: float | None = Field(default=None, ge=0, lt=1)

    @model_validator(mode="after")
    def _fields_match_the_kind(self) -> "ModelChoice":
        if self.model_kind == "decision" and self.reasoning_effort is not None:
            raise ValueError("a decision model takes no reasoning effort")
        if self.model_kind == "chat" and (
            self.family_confidence_threshold is not None or self.question_type_margin is not None
        ):
            raise ValueError("only a decision model takes thresholds")
        return self


class ModelAssignmentOut(BaseModel):
    model_id: str
    reasoning_effort: ReasoningEffort | None
    model_kind: ModelKindValue
    # Set, defaults filled in, only on a decision model.
    family_confidence_threshold: float | None
    question_type_margin: float | None
    updated_by_email: str | None
    updated_at: datetime


class SlotOut(BaseModel):
    slot: str
    label: str
    call_sites: str
    needs: list[str]
    default_reasoning_effort: ReasoningEffort
    warning: str | None
    accepts_decision_models: bool
    # None while the slot runs the default.
    assignment: ModelAssignmentOut | None


class ModelAssignmentsOut(BaseModel):
    # False without OPENROUTER_API_KEY: every slot runs the default and the
    # section is read-only.
    gateway_configured: bool
    # What the default is: CHAT_MODEL at the model's own reasoning default.
    default_model: str
    slots: list[SlotOut]


class CatalogModelOut(BaseModel):
    model_id: str
    name: str
    prompt_price_per_million: float | None
    completion_price_per_million: float | None
    supports_reasoning: bool
    model_kind: ModelKindValue = "chat"


class ProbeResultOut(BaseModel):
    passed: bool
    details: list[str]
