"""create feedback table

Revision ID: 5d95b6897886
Revises: 315be7394859
Create Date: 2026-09-01 14:19:08.074431

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '5d95b6897886'
down_revision: str | Sequence[str] | None = '315be7394859'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "feedback",
        # message_id is LangGraph's own auto-assigned per-message uuid
        # (langgraph.graph.message.add_messages), not a surrogate id here --
        # mirrors chat_threads' use of thread_id as its own primary key. One
        # row per message: resubmitting (ticket 11's design) overwrites via
        # ON CONFLICT (message_id) DO UPDATE, see feedback/repository.py.
        sa.Column("message_id", sa.Text, primary_key=True),
        sa.Column("thread_id", sa.Text, sa.ForeignKey("chat_threads.thread_id"), nullable=False),
        sa.Column("flagged", sa.Boolean, nullable=False, server_default=sa.false()),
        # Nullable, not just optional -- ticket 12's free-text-only "Give
        # feedback" submits through this same table with every score omitted.
        sa.Column("faithfulness", sa.Float, nullable=True),
        sa.Column("relevance", sa.Float, nullable=True),
        sa.Column("style", sa.Float, nullable=True),
        sa.Column("citation", sa.Float, nullable=True),
        sa.Column("comment", sa.Text, nullable=True),
        sa.Column(
            "submitted_by", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("feedback_thread_id_idx", "feedback", ["thread_id"])
    # Ticket 17 (evals feedback review) lists flagged feedback for the admin
    # promote workflow.
    op.create_index("feedback_flagged_idx", "feedback", ["flagged"], postgresql_where=sa.text("flagged"))


def downgrade() -> None:
    op.drop_table("feedback")
