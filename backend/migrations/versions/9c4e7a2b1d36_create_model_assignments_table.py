"""create model assignments table

Revision ID: 9c4e7a2b1d36
Revises: 3d8a5c1f9b72
Create Date: 2026-10-17 18:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '9c4e7a2b1d36'
down_revision: str | Sequence[str] | None = '3d8a5c1f9b72'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Issue #24: one row per model slot an admin has set. Slots are code
    # (model_assignments/slots.py), like question types, so there's no FK
    # or check constraint on `slot`; the route validates it. A slot with no
    # row runs CHAT_MODEL, so the table starts empty and nothing changes on
    # deploy. reasoning_effort NULL means the model's own default.
    op.create_table(
        "model_assignments",
        sa.Column("slot", sa.Text, primary_key=True),
        sa.Column("model_id", sa.Text, nullable=False),
        sa.Column("reasoning_effort", sa.Text, nullable=True),
        sa.Column(
            "updated_by",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint(
            "reasoning_effort IN ('none', 'low', 'medium', 'high')",
            name="model_assignments_reasoning_effort",
        ),
    )


def downgrade() -> None:
    op.drop_table("model_assignments")
