"""Pydantic request/response shapes for the per-slot model settings
(api/routes/model_assignments.py) and the chat request's admin-only
overrides (api/routes/chat.py).
"""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

ReasoningEffort = Literal["none", "low", "medium", "high"]


class ModelChoice(BaseModel):
    """A model for one slot: an OpenRouter model id, and a reasoning effort
    or None for the model's own default.
    """

    model_id: str = Field(min_length=1)
    reasoning_effort: ReasoningEffort | None = None


class ModelAssignmentOut(BaseModel):
    model_id: str
    reasoning_effort: ReasoningEffort | None
    updated_by_email: str | None
    updated_at: datetime


class SlotOut(BaseModel):
    slot: str
    label: str
    call_sites: str
    needs: list[str]
    default_reasoning_effort: ReasoningEffort
    warning: str | None
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


class ProbeResultOut(BaseModel):
    passed: bool
    details: list[str]

