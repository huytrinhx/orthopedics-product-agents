"""add description override to product families

Revision ID: d9d071ea3bda
Revises: d4e8a1f37c25
Create Date: 2026-10-18 22:08:29.464378

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'd9d071ea3bda'
down_revision: str | Sequence[str] | None = 'd4e8a1f37c25'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # The admin's own wording of a family's description for the intent
    # classifiers, replacing the generated one (catalog/family_descriptions.py,
    # issue #30) when set. NULL means use the generated description. Like
    # display_name, a catalog re-seed never touches it.
    op.add_column("product_families", sa.Column("description_override", sa.Text, nullable=True))


def downgrade() -> None:
    op.drop_column("product_families", "description_override")
