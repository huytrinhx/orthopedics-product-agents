"""Admin-only per-slot model settings (issue #24): which model, and which
reasoning effort, each node that calls a chat model runs on. The slots are
code (model_assignments/slots.py); a saved choice is a row in
`model_assignments`, and an unset slot runs `CHAT_MODEL`.

Saving is a two-layer check. The model must be in OpenRouter's catalog with
metadata covering the slot's needs (model_assignments/catalog.py), and then
pass the slot's live probe (model_assignments/probe.py); a failed probe
rejects the save with the reason. Test runs the same checks without saving.
Without OPENROUTER_API_KEY all of it is read-only.
"""
from fastapi import APIRouter, Depends, HTTPException, status

from auth.dependencies import require_admin
from auth.repository import UserRecord
from config.llm_clients import configured_chat_model
from model_assignments.catalog import CatalogUnavailable, find_catalog_model, models_for_slot
from model_assignments.models import (
    CatalogModelOut,
    ModelAssignmentOut,
    ModelAssignmentsOut,
    ModelChoice,
    ProbeResultOut,
    SlotOut,
)
from model_assignments.probe import ProbeResult, probe_slot
from model_assignments.repository import delete_assignment, list_assignments, save_assignment
from model_assignments.runtime import gateway_configured
from model_assignments.slots import SLOT_DEFINITIONS, Slot

router = APIRouter()


def _slot(slot_name: str) -> Slot:
    try:
        return Slot(slot_name)
    except ValueError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown model slot: {slot_name}")


def _require_gateway() -> None:
    if not gateway_configured():
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "Model settings are read-only until OPENROUTER_API_KEY is set; every slot runs the "
            "default model.",
        )


async def _assignments_out() -> ModelAssignmentsOut:
    configured = gateway_configured()
    # Without the gateway every slot runs the default, so that's what shows.
    saved = await list_assignments() if configured else {}
    slots = []
    for slot, definition in SLOT_DEFINITIONS.items():
        assignment = saved.get(slot.value)
        slots.append(
            SlotOut(
                slot=slot.value,
                label=definition.label,
                call_sites=definition.call_sites,
                needs=[need.value for need in definition.needs],
                default_reasoning_effort=definition.default_reasoning_effort,
                warning=definition.warning,
                assignment=(
                    ModelAssignmentOut(
                        model_id=assignment.model_id,
                        reasoning_effort=assignment.reasoning_effort,
                        updated_by_email=assignment.updated_by_email,
                        updated_at=assignment.updated_at,
                    )
                    if assignment
                    else None
                ),
            )
        )
    return ModelAssignmentsOut(
        gateway_configured=configured, default_model=configured_chat_model(), slots=slots
    )


async def _check_catalog(slot: Slot, choice: ModelChoice) -> None:
    """The static layer: the catalog metadata the picker filters on, checked
    again here so a hand-built request can't skip it.
    """
    try:
        catalog_model = await find_catalog_model(choice.model_id)
    except CatalogUnavailable as error:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(error))
    if catalog_model is None:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, f"{choice.model_id} isn't in OpenRouter's model catalog"
        )
    missing = catalog_model.missing_needs(slot)
    if missing:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"{choice.model_id} doesn't list support for "
            f"{', '.join(need.value.replace('_', ' ') for need in missing)}, which "
            f"{SLOT_DEFINITIONS[slot].label} needs",
        )
    if choice.reasoning_effort is not None and not catalog_model.supports_reasoning:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"{choice.model_id} doesn't support a reasoning effort; leave it unset",
        )


async def _probe(slot: Slot, choice: ModelChoice) -> ProbeResult:
    _require_gateway()
    await _check_catalog(slot, choice)
    return await probe_slot(slot, choice.model_id, choice.reasoning_effort)


@router.get("", response_model=ModelAssignmentsOut)
async def get_model_assignments(admin: UserRecord = Depends(require_admin)) -> ModelAssignmentsOut:
    return await _assignments_out()


@router.get("/{slot_name}/models", response_model=list[CatalogModelOut])
async def list_slot_models(
    slot_name: str, admin: UserRecord = Depends(require_admin)
) -> list[CatalogModelOut]:
    """The picker's options: catalog models whose metadata covers the slot's
    needs, dated ids preferred over their aliases.
    """
    slot = _slot(slot_name)
    _require_gateway()
    try:
        catalog_models = await models_for_slot(slot)
    except CatalogUnavailable as error:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(error))
    return [
        CatalogModelOut(
            model_id=catalog_model.model_id,
            name=catalog_model.name,
            prompt_price_per_million=catalog_model.prompt_price_per_million,
            completion_price_per_million=catalog_model.completion_price_per_million,
            supports_reasoning=catalog_model.supports_reasoning,
        )
        for catalog_model in catalog_models
    ]


@router.post("/{slot_name}/test", response_model=ProbeResultOut)
async def test_model_assignment(
    slot_name: str, body: ModelChoice, admin: UserRecord = Depends(require_admin)
) -> ProbeResultOut:
    """Runs the slot's probe without saving. A failed probe is still a 200:
    the result is the answer.
    """
    result = await _probe(_slot(slot_name), body)
    return ProbeResultOut(passed=result.passed, details=result.details)


@router.put("/{slot_name}", response_model=ModelAssignmentsOut)
async def update_model_assignment(
    slot_name: str, body: ModelChoice, admin: UserRecord = Depends(require_admin)
) -> ModelAssignmentsOut:
    slot = _slot(slot_name)
    result = await _probe(slot, body)
    if not result.passed:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"{body.model_id} failed the {SLOT_DEFINITIONS[slot].label} probe, so it wasn't "
            "saved:\n" + "\n".join(result.details),
        )
    await save_assignment(slot.value, body.model_id, body.reasoning_effort, admin.id)
    return await _assignments_out()


@router.delete("/{slot_name}", response_model=ModelAssignmentsOut)
async def reset_model_assignment(
    slot_name: str, admin: UserRecord = Depends(require_admin)
) -> ModelAssignmentsOut:
    """Puts the slot back on the default."""
    slot = _slot(slot_name)
    _require_gateway()
    await delete_assignment(slot.value)
    return await _assignments_out()
