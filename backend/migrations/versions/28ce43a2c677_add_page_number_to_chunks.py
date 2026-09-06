"""add page number to chunks

Revision ID: 28ce43a2c677
Revises: e2f039991c90
Create Date: 2026-09-05 00:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '28ce43a2c677'
down_revision: str | Sequence[str] | None = 'e2f039991c90'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Ticket 26: the page a chunk's content *starts* on, within its source
    # PDF -- lets the document viewer jump straight to the right page
    # instead of always opening at page 1. Nullable: non-PDF documents
    # (.txt/.md, backend/ingestion/text_extraction.py) have no pages at
    # all, and chunks written before this shipped have no value to backfill
    # from (no migration backfill -- re-indexing, already a one-click admin
    # action, populates it going forward).
    op.add_column("chunks", sa.Column("page_number", sa.Integer, nullable=True))


def downgrade() -> None:
    op.drop_column("chunks", "page_number")
