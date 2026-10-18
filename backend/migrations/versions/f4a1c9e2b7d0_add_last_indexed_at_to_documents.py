"""add last_indexed_at to documents

Revision ID: f4a1c9e2b7d0
Revises: b7d3e94a1c58
Create Date: 2026-10-16 18:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'f4a1c9e2b7d0'
down_revision: str | Sequence[str] | None = 'b7d3e94a1c58'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Nullable, no backfill (issue #14): stamped only when an indexing run
    # finishes successfully. Nothing already in the table records that --
    # updated_at also moves on tag edits, status changes, and re-uploads --
    # so existing "done" documents stay NULL (shown as unknown) rather than
    # getting a guessed timestamp.
    op.add_column(
        "documents", sa.Column("last_indexed_at", sa.DateTime(timezone=True), nullable=True)
    )


def downgrade() -> None:
    op.drop_column("documents", "last_indexed_at")
