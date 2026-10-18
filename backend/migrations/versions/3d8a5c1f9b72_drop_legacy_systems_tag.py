"""drop legacy systems tag

Revision ID: 3d8a5c1f9b72
Revises: f4a1c9e2b7d0
Create Date: 2026-10-17 10:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '3d8a5c1f9b72'
down_revision: str | Sequence[str] | None = 'f4a1c9e2b7d0'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Issue #20, the contract step of systems -> Product Family + Trays
    # (..._tag_documents_with_family_and_trays.py was the expand step).
    # Production's System tags were already carried over to families/trays
    # by the one-off migrate_legacy_system_tags.py script (#12, since
    # deleted), and nothing reads them any more. Dropping each column drops
    # its FK to `systems` with it; no index was ever built on either.
    op.drop_column("chunks", "system_id")
    op.drop_column("documents", "system_id")
    op.drop_index("systems_name_lower_idx", table_name="systems")
    op.drop_table("systems")


def downgrade() -> None:
    # Recreates the empty structures only -- the dropped tag names and
    # assignments aren't recoverable from anything this schema keeps.
    op.create_table(
        "systems",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("systems_name_lower_idx", "systems", [sa.text("lower(name)")], unique=True)
    op.add_column(
        "documents",
        sa.Column("system_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("systems.id"), nullable=True),
    )
    op.add_column(
        "chunks",
        sa.Column("system_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("systems.id"), nullable=True),
    )
