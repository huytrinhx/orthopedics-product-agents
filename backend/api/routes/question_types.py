"""Question Types -> Doc Types hierarchy (issue #17). Admin-only, edited from
the Settings page (frontend/app/admin/doc-type-hierarchy-section.tsx).

Question types themselves (slug + description) are code --
agents/question_types.py -- so this lists them from there and only lets an
admin change which document types each one retrieves from, and in what
order. `unclassified` is listed read-only: it always retrieves unfiltered.
See docs/adr/0006-doc-type-hierarchy-is-data.md.
"""
import psycopg
from fastapi import APIRouter, Depends, HTTPException, status

from agents.question_types import QUESTION_TYPES, UNCLASSIFIED, QuestionType
from auth.dependencies import require_admin
from auth.repository import UserRecord
from doc_type_hierarchy.models import (
    QuestionTypeOut,
    RankedDocumentTypeOut,
    SetDocumentTypesRequest,
)
from doc_type_hierarchy.repository import RankedDocumentType, list_hierarchy, set_document_types

router = APIRouter()


def _question_type_out(
    question_type: QuestionType, document_types: list[RankedDocumentType]
) -> QuestionTypeOut:
    return QuestionTypeOut(
        name=question_type.name,
        description=question_type.description,
        editable=question_type.name != UNCLASSIFIED,
        document_types=[
            RankedDocumentTypeOut(id=document_type.id, name=document_type.name)
            for document_type in document_types
        ],
    )


@router.get("/question-types", response_model=list[QuestionTypeOut])
async def list_question_types(admin: UserRecord = Depends(require_admin)) -> list[QuestionTypeOut]:
    hierarchy = await list_hierarchy()
    return [
        _question_type_out(question_type, hierarchy.get(question_type.name, []))
        for question_type in QUESTION_TYPES
    ]


@router.put("/question-types/{name}/document-types", response_model=QuestionTypeOut)
async def set_question_type_document_types(
    name: str, body: SetDocumentTypesRequest, admin: UserRecord = Depends(require_admin)
) -> QuestionTypeOut:
    """Replaces question type `name`'s ranked document types with
    `body.document_type_ids`, most authoritative first."""
    question_type = next(
        (candidate for candidate in QUESTION_TYPES if candidate.name == name), None
    )
    if question_type is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown question type '{name}'")
    if name == UNCLASSIFIED:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "unclassified always retrieves from every document type and can't be edited",
        )
    if len(set(body.document_type_ids)) != len(body.document_type_ids):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "A document type is listed more than once")
    try:
        await set_document_types(name, body.document_type_ids)
    except psycopg.errors.ForeignKeyViolation:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "One of those document types doesn't exist")
    hierarchy = await list_hierarchy()
    return _question_type_out(question_type, hierarchy.get(name, []))
