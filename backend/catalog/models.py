"""Pydantic request/response shapes for the catalog routes."""
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


class ProductFamilySummaryOut(BaseModel):
    """A family without its trays -- what a display-name edit returns."""

    id: uuid.UUID
    name: str
    display_name: str


class DisplayNameUpdate(BaseModel):
    # None or blank resets to the catalog name (catalog.repository).
    display_name: str | None = None
