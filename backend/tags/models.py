"""Pydantic response/request shapes for the tags routes."""
import uuid

from pydantic import BaseModel


class TagOut(BaseModel):
    id: uuid.UUID
    name: str


class CreateTagRequest(BaseModel):
    name: str


class RenameTagRequest(BaseModel):
    name: str


class MergeTagRequest(BaseModel):
    # The canonical document type the stray one (the path id) folds into.
    into_id: uuid.UUID
