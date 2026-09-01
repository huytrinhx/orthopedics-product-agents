"""create app settings table

Revision ID: 315be7394859
Revises: 7fc445c9b589
Create Date: 2026-09-01 17:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '315be7394859'
down_revision: str | Sequence[str] | None = '7fc445c9b589'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Single-row table for process-wide admin config (ticket 14: the admin-
    # chosen default chat workflow). `id` is pinned to 1 via a check
    # constraint so there's exactly one row ever -- the repository always
    # UPDATEs it rather than choosing among rows, and there's nothing else
    # here yet to justify a general key/value settings table.
    op.create_table(
        "app_settings",
        sa.Column("id", sa.SmallInteger, primary_key=True),
        sa.Column("default_workflow", sa.Text, nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_by", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=True),
        sa.CheckConstraint("id = 1", name="app_settings_singleton"),
    )
    op.execute("INSERT INTO app_settings (id, default_workflow) VALUES (1, 'deterministic')")


def downgrade() -> None:
    op.drop_table("app_settings")
