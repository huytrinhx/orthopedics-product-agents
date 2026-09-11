"""add last_login_at to users

Revision ID: c35222d8836f
Revises: 48ff0b231339
Create Date: 2026-09-11 10:13:34.980047

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'c35222d8836f'
down_revision: str | Sequence[str] | None = '48ff0b231339'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Nullable, no default: existing rows have never logged in under this
    # column's existence, and NULL is the honest "never logged in" value for
    # them (shown as "Never" in the Users tab) rather than a fabricated
    # timestamp.
    op.add_column("users", sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column("users", "last_login_at")
