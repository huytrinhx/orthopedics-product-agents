"""add feedback resolved flag and chat_threads rerun columns

Revision ID: 0bb3fb27c761
Revises: 5d95b6897886
Create Date: 2026-09-05 12:54:50.669107

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '0bb3fb27c761'
down_revision: str | Sequence[str] | None = '5d95b6897886'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Ticket 15 (rescoped 2026-09-05, grilling session): an admin marks a
    # flagged feedback item resolved once a rerun confirms the current code
    # answers it correctly -- separate from `flagged` itself, which stays a
    # permanent record of "a user complained about this."
    op.add_column(
        "feedback", sa.Column("resolved", sa.Boolean, nullable=False, server_default=sa.false())
    )

    # A rerun (ticket 15) is a real chat_threads row like any other -- it
    # goes through the exact same graph/checkpointer/streaming machinery, so
    # every existing read path (GET /chat/threads/{id}, the chat UI itself)
    # works on it completely unchanged. rerun_of_message_id is what marks it
    # as a rerun and links it back to the flagged feedback item it reran
    # (nullable: only set on rerun threads, null for every ordinary thread a
    # rep starts) -- list_threads excludes non-null rows so reruns never
    # clutter an admin's own personal sidebar. workflow_name records which
    # workflow the admin explicitly chose for this rerun (ordinary threads
    # don't need this column at all -- they always use whatever the admin
    # default was at request time), since a later resume call
    # (POST /{workflow_name}/resume) needs to route to the same workflow the
    # thread was created against.
    op.add_column(
        "chat_threads",
        sa.Column("rerun_of_message_id", sa.Text, sa.ForeignKey("feedback.message_id"), nullable=True),
    )
    op.add_column("chat_threads", sa.Column("workflow_name", sa.Text, nullable=True))
    op.create_index(
        "chat_threads_rerun_of_message_id_idx",
        "chat_threads",
        ["rerun_of_message_id"],
        postgresql_where=sa.text("rerun_of_message_id IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("chat_threads_rerun_of_message_id_idx", table_name="chat_threads")
    op.drop_column("chat_threads", "workflow_name")
    op.drop_column("chat_threads", "rerun_of_message_id")
    op.drop_column("feedback", "resolved")
