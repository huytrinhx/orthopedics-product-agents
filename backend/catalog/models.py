"""Pydantic response shapes for the catalog routes."""
import uuid

from pydantic import BaseModel


class TrayOut(BaseModel):
    id: uuid.UUID
    name: str
    display_name: str


class ProductFamilyOut(BaseModel):
    id: uuid.UUID
    name: str
    display_name: str
    trays: list[TrayOut]
