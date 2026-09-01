"""create documents table

Revision ID: 11420e734f45
Revises: 40adfc506662
Create Date: 2026-08-25 15:22:35.102058

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '11420e734f45'
down_revision: str | Sequence[str] | None = '40adfc506662'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Ticket 06/07 (ingestion) only ever move a document forward through this
# sequence or park it at "failed" -- nothing skips or reverses a step.
DOCUMENT_STATUSES = ("queued", "processing", "done", "failed")


def upgrade() -> None:
    op.execute(
        "CREATE TYPE document_status AS ENUM ('queued', 'processing', 'done', 'failed')"
    )
    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column("filename", sa.Text, nullable=False),
        # Where the raw upload lives under INGEST_DATA_DIR -- see
        # backend/api/routes/documents.py.
        sa.Column("storage_path", sa.Text, nullable=False),
        sa.Column(
            "status",
            postgresql.ENUM(*DOCUMENT_STATUSES, name="document_status", create_type=False),
            nullable=False,
            server_default="queued",
        ),
        sa.Column("error", sa.Text, nullable=True),
        sa.Column("uploaded_by", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("documents")
    op.execute("DROP TYPE document_status")
