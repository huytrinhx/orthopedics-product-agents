"""add pending document status

Revision ID: a796a69506ed
Revises: eaeb89f1fc32
Create Date: 2026-08-31 15:09:24.254835

"""
from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'a796a69506ed'
down_revision: str | Sequence[str] | None = 'eaeb89f1fc32'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Upload (backend/api/routes/documents.py) no longer auto-triggers
    # ingestion -- a freshly uploaded document rests in "pending" (the new
    # default) until an admin clicks Index/Reindex, which moves it into the
    # queued/processing/done/failed sequence documented in 11420e734f45.
    # ADD VALUE must be committed before it can be referenced (e.g. in the
    # DEFAULT below) -- autocommit_block() runs it outside this migration's
    # normal transaction so the two statements don't share one.
    with op.get_context().autocommit_block():
        op.execute("ALTER TYPE document_status ADD VALUE 'pending' BEFORE 'queued'")
    op.execute("ALTER TABLE documents ALTER COLUMN status SET DEFAULT 'pending'")


def downgrade() -> None:
    # Postgres can't drop a single enum value in place -- recreate the type
    # without it, bumping any "pending" rows to "queued" first so the cast
    # doesn't choke on a value the old type never had.
    op.execute("UPDATE documents SET status = 'queued' WHERE status = 'pending'")
    op.execute("ALTER TABLE documents ALTER COLUMN status DROP DEFAULT")
    op.execute("ALTER TYPE document_status RENAME TO document_status_old")
    op.execute("CREATE TYPE document_status AS ENUM ('queued', 'processing', 'done', 'failed')")
    op.execute(
        "ALTER TABLE documents ALTER COLUMN status TYPE document_status "
        "USING status::text::document_status"
    )
    op.execute("ALTER TABLE documents ALTER COLUMN status SET DEFAULT 'queued'")
    op.execute("DROP TYPE document_status_old")
