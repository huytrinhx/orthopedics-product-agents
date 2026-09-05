"""cascade-delete rerun threads when their flagged feedback is deleted

Revision ID: e2f039991c90
Revises: 0bb3fb27c761
Create Date: 2026-09-05 14:47:19.852239

"""
from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'e2f039991c90'
down_revision: str | Sequence[str] | None = '0bb3fb27c761'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Ticket 15 follow-up: an admin can now delete a flagged feedback item
    # outright. A rerun's chat_threads row has no independent value once
    # its flagged item is gone -- it existed purely to check on that one
    # item -- so it cascades away too, the same reasoning chunks.document_id
    # already cascades on documents (migration 4b7c2d00de10), rather than
    # blocking the delete (the default RESTRICT every other FK in this
    # schema uses) or silently SET NULL-ing it back into the admin's own
    # personal sidebar (list_threads only excludes non-null
    # rerun_of_message_id rows).
    op.drop_constraint("chat_threads_rerun_of_message_id_fkey", "chat_threads", type_="foreignkey")
    op.create_foreign_key(
        "chat_threads_rerun_of_message_id_fkey",
        "chat_threads",
        "feedback",
        ["rerun_of_message_id"],
        ["message_id"],
        ondelete="CASCADE",
    )


def downgrade() -> None:
    op.drop_constraint("chat_threads_rerun_of_message_id_fkey", "chat_threads", type_="foreignkey")
    op.create_foreign_key(
        "chat_threads_rerun_of_message_id_fkey",
        "chat_threads",
        "feedback",
        ["rerun_of_message_id"],
        ["message_id"],
    )
