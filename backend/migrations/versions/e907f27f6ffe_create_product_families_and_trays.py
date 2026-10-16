"""create product families and trays

Revision ID: e907f27f6ffe
Revises: c35222d8836f
Create Date: 2026-10-16 08:08:37.136918

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'e907f27f6ffe'
down_revision: str | Sequence[str] | None = 'c35222d8836f'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Postgres mirrors of the graph's ProductFamily and Tray nodes, upserted
    # by backend/ingestion/seed_master_catalog.py so documents can be tagged
    # against them with real foreign keys. `name` is the stable catalog name
    # the seed keys on (unique, case-sensitive -- it's copied verbatim from
    # the master file, never typed by hand); `display_name` defaults to it
    # and is the admin-editable label a re-seed never overwrites.
    op.create_table(
        "product_families",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column("name", sa.Text, nullable=False, unique=True),
        sa.Column("display_name", sa.Text, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_table(
        "trays",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column("name", sa.Text, nullable=False, unique=True),
        sa.Column("display_name", sa.Text, nullable=False),
        sa.Column(
            "product_family_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("product_families.id"),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("trays_product_family_id_idx", "trays", ["product_family_id"])


def downgrade() -> None:
    op.drop_index("trays_product_family_id_idx", table_name="trays")
    op.drop_table("trays")
    op.drop_table("product_families")
