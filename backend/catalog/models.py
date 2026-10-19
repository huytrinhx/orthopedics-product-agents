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
    # What the intent classifiers are told about the family (issue #30):
    # the catalog-generated description, and the admin's replacement for it
    # when one is saved (None otherwise).
    generated_description: str
    description_override: str | None = None


class ProductFamilySummaryOut(BaseModel):
    """A family without its trays -- what a display-name edit returns."""

    id: uuid.UUID
    name: str
    display_name: str


class DisplayNameUpdate(BaseModel):
    # None or blank resets to the catalog name (catalog.repository).
    display_name: str | None = None


class FamilyDescriptionUpdate(BaseModel):
    # None or blank goes back to the generated description.
    description: str | None = None


class FamilyDescriptionOut(BaseModel):
    id: uuid.UUID
    description_override: str | None
