"""Pydantic response shape for the admin read of a document's SKU index
(GET /documents/{document_id}/skus, issue #27)."""
from typing import Literal

from pydantic import BaseModel

MatchKind = Literal["exact", "series", "not_in_catalog"]


class DocumentSkuOut(BaseModel):
    sku: str
    mention: str
    match_kind: MatchKind
    # None when the SKU isn't in the catalog.
    in_document_family: bool | None
    # From the catalog, never the document text; None for a flagged token.
    description: str | None
    chunk_index: int
    # `{document_id}#{chunk_index}`, the id an answer cites.
    citation: str
    # Read from the chunk row.
    page_number: int | None
    section_title: str | None
    table_heading: str | None
    row_text: str
