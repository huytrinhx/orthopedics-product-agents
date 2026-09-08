"""add is_active to users

Revision ID: 48ff0b231339
Revises: 28ce43a2c677
Create Date: 2026-09-08 00:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '48ff0b231339'
down_revision: str | Sequence[str] | None = '28ce43a2c677'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # New signups start disabled and need an admin to flip them on (see
    # backend/api/routes/users.py) -- but every row that already exists at
    # migration time is grandfathered in as active, so nobody using the app
    # today gets locked out. That's why this is a server_default of true
    # rather than false: the application always passes is_active explicitly
    # on INSERT (create_user), so the default only ever fires for this
    # backfill, never for a fresh signup.
    op.add_column(
        "users", sa.Column("is_active", sa.Boolean, nullable=False, server_default=sa.true())
    )


def downgrade() -> None:
    op.drop_column("users", "is_active")
