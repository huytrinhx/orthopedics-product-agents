"""create document skus table

Revision ID: d4e8a1f37c25
Revises: b5f05639dedf
Create Date: 2026-10-18 18:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'd4e8a1f37c25'
down_revision: str | Sequence[str] | None = 'b5f05639dedf'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Issue #27: every SKU a document's chunks mention, found by a
    # deterministic scan (backend/document_skus/service.py). A row points at
    # its chunk by (document_id, chunk_index), the same pair a citation id
    # uses, and keeps only what the chunk row doesn't have. Page number,
    # section title and any later heading path are read from `chunks` at
    # query time rather than copied here, and descriptions come from the
    # catalog when read, never from the document text. No FK to `chunks`:
    # the vector leg deletes and rewrites a document's chunks, and both
    # tables are rewritten by the same indexing run anyway.
    op.create_table(
        "document_skus",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        # ON DELETE CASCADE, like chunks: a document delete takes its rows
        # with it.
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("chunk_index", sa.Integer, nullable=False),
        # The catalog SKU, or for a flagged token the token itself.
        sa.Column("sku", sa.Text, nullable=False),
        # The text as written: the SKU itself for an exact match, the
        # series or side-alternative form ("MSFB0xxx", "MPP1101L/R") that a
        # series row was resolved from.
        sa.Column("mention", sa.Text, nullable=False),
        sa.Column("match_kind", sa.Text, nullable=False),
        # NULL exactly when the SKU isn't in the catalog: there's no family
        # to compare.
        sa.Column("in_document_family", sa.Boolean, nullable=True),
        # Best-effort, from the line above the table's column header.
        sa.Column("table_heading", sa.Text, nullable=True),
        # The line the mention appeared on, as provenance only: PDF text
        # drops digits and mis-pairs multi-column rows.
        sa.Column("row_text", sa.Text, nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.CheckConstraint(
            "match_kind IN ('exact', 'series', 'not_in_catalog')",
            name="document_skus_match_kind",
        ),
        sa.CheckConstraint(
            "(match_kind = 'not_in_catalog') = (in_document_family IS NULL)",
            name="document_skus_family_only_for_catalog_skus",
        ),
    )
    op.create_index(
        "document_skus_document_chunk_idx", "document_skus", ["document_id", "chunk_index"]
    )
    op.create_index("document_skus_sku_idx", "document_skus", ["sku"])


def downgrade() -> None:
    op.drop_index("document_skus_sku_idx", table_name="document_skus")
    op.drop_index("document_skus_document_chunk_idx", table_name="document_skus")
    op.drop_table("document_skus")
