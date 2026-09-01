"""Pydantic response/request shapes for the admin settings route."""
from pydantic import BaseModel


class WorkflowInfoOut(BaseModel):
    name: str
    # False for a registered-but-stub workflow (see agents/registry.py's
    # `functional` flag) -- the admin picker shows these disabled rather
    # than omitting them, so it's visible that other architectures exist.
    functional: bool


class AdminSettingsOut(BaseModel):
    default_workflow: str
    workflows: list[WorkflowInfoOut]


class SetDefaultWorkflowRequest(BaseModel):
    default_workflow: str
