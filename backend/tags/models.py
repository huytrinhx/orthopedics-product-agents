"""Pydantic response/request shapes for the tags routes."""
import uuid

from pydantic import BaseModel


class TagOut(BaseModel):
    id: uuid.UUID
    name: str


class CreateTagRequest(BaseModel):
    name: str
