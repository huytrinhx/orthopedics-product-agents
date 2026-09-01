"""create users table

Revision ID: 40adfc506662
Revises: 
Create Date: 2026-08-25 15:12:22.818530

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '40adfc506662'
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute('CREATE EXTENSION IF NOT EXISTS "pgcrypto"')  # gen_random_uuid()
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column("email", sa.Text, nullable=False, unique=True),
        # Nullable: an OAuth-only account (see ticket 03, Google sign-in) never
        # sets a password.
        sa.Column("hashed_password", sa.Text, nullable=True),
        sa.Column("is_admin", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("users")
