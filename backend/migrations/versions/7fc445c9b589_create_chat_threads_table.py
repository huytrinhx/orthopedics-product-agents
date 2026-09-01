"""create chat threads table

Revision ID: 7fc445c9b589
Revises: 4b7c2d00de10
Create Date: 2026-09-01 08:39:23.204401

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '7fc445c9b589'
down_revision: str | Sequence[str] | None = '4b7c2d00de10'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "chat_threads",
        # Not a UUID: matches the "{user_id}:{uuid4().hex}" strings
        # backend/api/routes/chat.py's _new_thread_id mints and hands to the
        # LangGraph checkpointer as its own thread_id -- this table only
        # tracks the sidebar-facing metadata (title, ownership, recency) for
        # threads that already exist there, keyed the same way.
        sa.Column("thread_id", sa.Text, primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("title", sa.Text, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("chat_threads_user_id_idx", "chat_threads", ["user_id"])


def downgrade() -> None:
    op.drop_table("chat_threads")
