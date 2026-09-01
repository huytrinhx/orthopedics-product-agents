"""Admin-only app settings (ticket 14). Today the only setting is which
registered workflow (agents/registry.py) new chat conversations default to
-- the one place workflow choice is ever exposed; regular chat users never
see it (see api/routes/chat.py's POST /chat/stream, which reads this).
"""
from fastapi import APIRouter, Depends, HTTPException, status

from agents.registry import is_functional, list_workflows
from auth.dependencies import require_admin
from auth.repository import UserRecord
from settings.models import AdminSettingsOut, SetDefaultWorkflowRequest, WorkflowInfoOut
from settings.repository import get_settings, set_default_workflow

router = APIRouter()


async def _settings_out() -> AdminSettingsOut:
    current = await get_settings()
    return AdminSettingsOut(
        default_workflow=current.default_workflow,
        workflows=[
            WorkflowInfoOut(name=name, functional=is_functional(name))
            for name in list_workflows()
        ],
    )


@router.get("/settings", response_model=AdminSettingsOut)
async def get_admin_settings(admin: UserRecord = Depends(require_admin)) -> AdminSettingsOut:
    return await _settings_out()


@router.put("/settings", response_model=AdminSettingsOut)
async def update_admin_settings(
    body: SetDefaultWorkflowRequest, admin: UserRecord = Depends(require_admin)
) -> AdminSettingsOut:
    if body.default_workflow not in list_workflows():
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown workflow: {body.default_workflow}")
    if not is_functional(body.default_workflow):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, f"Workflow '{body.default_workflow}' isn't implemented yet"
        )
    await set_default_workflow(body.default_workflow, admin.id)
    return await _settings_out()
