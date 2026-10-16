"""Pydantic response/request shapes for the question-type routes."""
import uuid

from pydantic import BaseModel


class RankedDocumentTypeOut(BaseModel):
    id: uuid.UUID
    name: str


class QuestionTypeOut(BaseModel):
    name: str
    description: str
    # False only for `unclassified`, which always retrieves unfiltered.
    editable: bool
    # Most authoritative first.
    document_types: list[RankedDocumentTypeOut]


class SetDocumentTypesRequest(BaseModel):
    # The question type's complete ranked list, most authoritative first --
    # it replaces whatever was there. Empty means no doc-type filter.
    document_type_ids: list[uuid.UUID]
