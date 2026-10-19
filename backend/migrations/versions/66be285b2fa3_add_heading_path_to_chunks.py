"""add heading path to chunks

Revision ID: 66be285b2fa3
Revises: b5f05639dedf
Create Date: 2026-10-18 21:12:53.055234

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '66be285b2fa3'
down_revision: str | Sequence[str] | None = 'b5f05639dedf'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Issue #26: the chunk's headings from outermost to innermost, e.g.
    # [technique title, "Step 3. ..."] (backend/ingestion/headings.py), so
    # a step chunk knows which technique it belongs to. section_title stays
    # and is the path's last element. An empty array means the chunk has
    # no heading; NULL means it was indexed before this shipped -- no
    # backfill, a re-index writes it.
    op.add_column("chunks", sa.Column("heading_path", postgresql.ARRAY(sa.Text), nullable=True))


def downgrade() -> None:
    op.drop_column("chunks", "heading_path")
