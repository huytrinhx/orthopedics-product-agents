"""add decision model kind and thresholds to model assignments

Revision ID: b5f05639dedf
Revises: 9c4e7a2b1d36
Create Date: 2026-10-18 14:05:30.029586

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'b5f05639dedf'
down_revision: str | Sequence[str] | None = '9c4e7a2b1d36'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Issue #23: a slot can hold a decision model instead of a chat model,
    # and a decision model's two thresholds are saved with it. Every
    # existing row is a chat model, so the default keeps them exactly as
    # they were. Thresholds are NULL on chat rows; on a decision row NULL
    # means slots.py's default. Which slots accept a decision model is
    # code (slots.py), checked by the route, like `slot` itself.
    op.add_column(
        "model_assignments",
        sa.Column("model_kind", sa.Text, nullable=False, server_default="chat"),
    )
    op.add_column(
        "model_assignments",
        sa.Column("family_confidence_threshold", sa.Float, nullable=True),
    )
    op.add_column(
        "model_assignments",
        sa.Column("question_type_margin", sa.Float, nullable=True),
    )
    op.create_check_constraint(
        "model_assignments_model_kind",
        "model_assignments",
        "model_kind IN ('chat', 'decision')",
    )
    # A decision model has no reasoning effort, and a chat model has no
    # thresholds.
    op.create_check_constraint(
        "model_assignments_decision_fields",
        "model_assignments",
        "(model_kind = 'decision' AND reasoning_effort IS NULL) OR "
        "(model_kind = 'chat' AND family_confidence_threshold IS NULL "
        "AND question_type_margin IS NULL)",
    )
    op.create_check_constraint(
        "model_assignments_threshold_ranges",
        "model_assignments",
        "(family_confidence_threshold IS NULL OR "
        "(family_confidence_threshold > 0 AND family_confidence_threshold <= 1)) AND "
        "(question_type_margin IS NULL OR (question_type_margin >= 0 AND question_type_margin < 1))",
    )


def downgrade() -> None:
    # A decision model can't run as a chat model, so its row goes: the slot
    # is back on the default.
    op.execute("DELETE FROM model_assignments WHERE model_kind = 'decision'")
    op.drop_constraint("model_assignments_threshold_ranges", "model_assignments")
    op.drop_constraint("model_assignments_decision_fields", "model_assignments")
    op.drop_constraint("model_assignments_model_kind", "model_assignments")
    op.drop_column("model_assignments", "question_type_margin")
    op.drop_column("model_assignments", "family_confidence_threshold")
    op.drop_column("model_assignments", "model_kind")
