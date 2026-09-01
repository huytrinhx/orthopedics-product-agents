"""create systems and document types

Revision ID: eaeb89f1fc32
Revises: 11420e734f45
Create Date: 2026-08-31 14:12:17.579274

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'eaeb89f1fc32'
down_revision: str | Sequence[str] | None = '11420e734f45'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Both lookup tables are the same shape: an admin-managed, open-ended list
# of tag names (no fixed/hardcoded enum -- see ticket 05), reused across
# many documents via the nullable FK columns added to `documents` below.
_TAG_TABLES = ("systems", "document_types")


def upgrade() -> None:
    for table in _TAG_TABLES:
        op.create_table(
            table,
            sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), primary_key=True),
            sa.Column("name", sa.Text, nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        )
        # Case-insensitive uniqueness -- "REFLEX" and "reflex" are the same
        # tag, so a typo in casing reuses the existing row instead of
        # silently forking it.
        op.create_index(f"{table}_name_lower_idx", table, [sa.text("lower(name)")], unique=True)

    op.add_column("documents", sa.Column("system_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("systems.id"), nullable=True))
    op.add_column("documents", sa.Column("document_type_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("document_types.id"), nullable=True))


def downgrade() -> None:
    op.drop_column("documents", "document_type_id")
    op.drop_column("documents", "system_id")
    for table in _TAG_TABLES:
        op.drop_index(f"{table}_name_lower_idx", table_name=table)
        op.drop_table(table)
