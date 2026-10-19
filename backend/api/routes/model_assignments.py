"""Admin-only per-slot model settings (issue #24): which model, and which
reasoning effort, each node that calls a chat model runs on. The slots are
code (model_assignments/slots.py); a saved choice is a row in
`model_assignments`, and an unset slot runs `CHAT_MODEL`.

Saving is a two-layer check. The model must be in OpenRouter's catalog with
metadata covering the slot's needs (model_assignments/catalog.py), and then
pass the slot's live probe (model_assignments/probe.py); a failed probe
rejects the save with the reason. Test runs the same checks without saving.
Without OPENROUTER_API_KEY all of it is read-only.

A slot that accepts decision models (issue #23) can also be given one,
`model_kind` "decision": it must be in OpenRouter's decision model listing
and pass the slot's decision probe, and its thresholds are saved with it
(slots.py's defaults when not given). It's always saved under its dated
id, even when picked by another listed id.
"""
import logging

from fastapi import APIRouter, Depends, HTTPException, status

from auth.dependencies import require_admin
from auth.repository import UserRecord
from config.llm_clients import configured_chat_model
from model_assignments.catalog import (
    CatalogModel,
    CatalogUnavailable,
    decision_models_for_slot,
    find_catalog_model,
    find_decision_model,
    models_for_slot,
)
from model_assignments.models import (
    CatalogModelOut,
    ModelAssignmentOut,
    ModelAssignmentsOut,
    ModelChoice,
    ProbeResultOut,
    SlotOut,
)
from model_assignments.probe import ProbeResult, probe_slot
from model_assignments.repository import (
    ModelAssignmentRecord,
    delete_assignment,
    list_assignments,
    save_assignment,
)
from model_assignments.runtime import decision_slot_model, gateway_configured
from model_assignments.slots import SLOT_DEFINITIONS, ModelKind, Slot

logger = logging.getLogger(__name__)

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


def _assignment_out(slot: Slot, assignment: ModelAssignmentRecord) -> ModelAssignmentOut:
    """A saved row, with a decision model's default thresholds spelled out
    so Settings shows what actually runs.
    """
    family_confidence_threshold = question_type_margin = None
    if assignment.model_kind == ModelKind.DECISION.value:
        decision = decision_slot_model(
            slot,
            assignment.model_id,
            "setting",
            assignment.family_confidence_threshold,
            assignment.question_type_margin,
        )
        family_confidence_threshold = decision.family_confidence_threshold
        question_type_margin = decision.question_type_margin
    return ModelAssignmentOut(
        model_id=assignment.model_id,
        reasoning_effort=assignment.reasoning_effort,
        model_kind=assignment.model_kind,
        family_confidence_threshold=family_confidence_threshold,
        question_type_margin=question_type_margin,
        updated_by_email=assignment.updated_by_email,
        updated_at=assignment.updated_at,
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
                accepts_decision_models=definition.accepts_decision_models,
                assignment=_assignment_out(slot, assignment) if assignment else None,
            )
        )
    return ModelAssignmentsOut(
        gateway_configured=configured, default_model=configured_chat_model(), slots=slots
    )


async def _check_decision_catalog(slot: Slot, choice: ModelChoice) -> ModelChoice:
    """A decision model must be one the slot accepts and OpenRouter lists
    as a decision model. Returns the choice under the model's dated id.
    """
    if not SLOT_DEFINITIONS[slot].accepts_decision_models:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, f"{SLOT_DEFINITIONS[slot].label} can't run a decision model"
        )
    try:
        catalog_model = await find_decision_model(choice.model_id)
    except CatalogUnavailable as error:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(error))
    if catalog_model is None:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, f"{choice.model_id} isn't in OpenRouter's decision model catalog"
        )
    return choice.model_copy(update={"model_id": catalog_model.model_id})


async def _listed_decision_model(slot: Slot, model_id: str) -> CatalogModel | None:
    """The decision model listed under `model_id`, for a slot that accepts
    them; None otherwise, or when the decision listing can't be read."""
    if not SLOT_DEFINITIONS[slot].accepts_decision_models:
        return None
    try:
        return await find_decision_model(model_id)
    except CatalogUnavailable:
        return None


async def _check_catalog(slot: Slot, choice: ModelChoice) -> ModelChoice:
    """The static layer: the catalog metadata the picker filters on, checked
    again here so a hand-built request can't skip it. Returns the choice to
    probe and save.
    """
    if choice.model_kind == ModelKind.DECISION.value:
        return await _check_decision_catalog(slot, choice)
    try:
        catalog_model = await find_catalog_model(choice.model_id)
    except CatalogUnavailable as error:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(error))
    if catalog_model is None:
        decision_model = await _listed_decision_model(slot, choice.model_id)
        if decision_model is not None:
            # No id is in both listings, so a decision model sent as a chat
            # one is unambiguous: run it as what OpenRouter says it is,
            # rather than refuse an id the picker itself offered.
            if choice.reasoning_effort is not None:
                raise HTTPException(
                    status.HTTP_400_BAD_REQUEST,
                    f"{choice.model_id} is a decision model, which takes no reasoning effort",
                )
            return choice.model_copy(
                update={"model_id": decision_model.model_id, "model_kind": ModelKind.DECISION.value}
            )
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
    return choice


async def _probe(slot: Slot, choice: ModelChoice) -> tuple[ModelChoice, ProbeResult]:
    _require_gateway()
    choice = await _check_catalog(slot, choice)
    result = await probe_slot(slot, choice.model_id, choice.reasoning_effort, choice.model_kind)
    return choice, result


def _catalog_model_out(catalog_model: CatalogModel, model_kind: ModelKind) -> CatalogModelOut:
    return CatalogModelOut(
        model_id=catalog_model.model_id,
        name=catalog_model.name,
        prompt_price_per_million=catalog_model.prompt_price_per_million,
        completion_price_per_million=catalog_model.completion_price_per_million,
        # A decision model has no reasoning to set an effort for.
        supports_reasoning=model_kind is ModelKind.CHAT and catalog_model.supports_reasoning,
        model_kind=model_kind.value,
    )


@router.get("", response_model=ModelAssignmentsOut)
async def get_model_assignments(admin: UserRecord = Depends(require_admin)) -> ModelAssignmentsOut:
    return await _assignments_out()


@router.get("/{slot_name}/models", response_model=list[CatalogModelOut])
async def list_slot_models(
    slot_name: str, admin: UserRecord = Depends(require_admin)
) -> list[CatalogModelOut]:
    """The picker's options: catalog models whose metadata covers the slot's
    needs, dated ids preferred over their aliases -- then, for a slot that
    accepts them, decision models by dated id. If only the decision listing
    fails, the chat models are still offered, and the failure is logged.
    """
    slot = _slot(slot_name)
    _require_gateway()
    try:
        catalog_models = await models_for_slot(slot)
    except CatalogUnavailable as error:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(error))
    try:
        decision_models = await decision_models_for_slot(slot)
    except CatalogUnavailable as error:
        logger.warning("decision model catalog unavailable for slot=%s: %s", slot.value, error)
        decision_models = []
    return [
        *(_catalog_model_out(catalog_model, ModelKind.CHAT) for catalog_model in catalog_models),
        *(_catalog_model_out(catalog_model, ModelKind.DECISION) for catalog_model in decision_models),
    ]


@router.post("/{slot_name}/test", response_model=ProbeResultOut)
async def test_model_assignment(
    slot_name: str, body: ModelChoice, admin: UserRecord = Depends(require_admin)
) -> ProbeResultOut:
    """Runs the slot's probe without saving. A failed probe is still a 200:
    the result is the answer.
    """
    _, result = await _probe(_slot(slot_name), body)
    return ProbeResultOut(passed=result.passed, details=result.details)


@router.put("/{slot_name}", response_model=ModelAssignmentsOut)
async def update_model_assignment(
    slot_name: str, body: ModelChoice, admin: UserRecord = Depends(require_admin)
) -> ModelAssignmentsOut:
    slot = _slot(slot_name)
    choice, result = await _probe(slot, body)
    if not result.passed:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"{choice.model_id} failed the {SLOT_DEFINITIONS[slot].label} probe, so it wasn't "
            "saved:\n" + "\n".join(result.details),
        )
    if choice.model_kind == ModelKind.DECISION.value:
        # Saved with the defaults filled in, so a later change to slots.py's
        # defaults doesn't move a model someone has already tuned.
        decision = decision_slot_model(
            slot,
            choice.model_id,
            "setting",
            choice.family_confidence_threshold,
            choice.question_type_margin,
        )
        await save_assignment(
            slot.value,
            choice.model_id,
            None,
            admin.id,
            model_kind=decision.model_kind,
            family_confidence_threshold=decision.family_confidence_threshold,
            question_type_margin=decision.question_type_margin,
        )
    else:
        await save_assignment(slot.value, choice.model_id, choice.reasoning_effort, admin.id)
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
