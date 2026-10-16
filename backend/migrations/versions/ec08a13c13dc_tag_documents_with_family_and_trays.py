"""tag documents with a product family and trays

Revision ID: ec08a13c13dc
Revises: e907f27f6ffe
Create Date: 2026-10-16 14:30:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'ec08a13c13dc'
down_revision: str | Sequence[str] | None = 'e907f27f6ffe'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Schema only. Existing documents' legacy `system_id` tags are carried
    # over by the one-off backend/catalog/migrate_legacy_system_tags.py, not
    # a data migration here -- it needs the master catalog seed to have
    # mirrored the families/trays into Postgres first, which a fresh
    # database (CI, local setup) running `alembic upgrade head` hasn't.
    # `systems`/`system_id` stay until the legacy-removal ticket.
    #
    # Nullable: an untagged or not-yet-migrated document simply has no
    # family. No ON DELETE -- families are never deleted by the app (the
    # seed only upserts), so a delete that would orphan documents should
    # fail loudly rather than silently untag them.
    op.add_column(
        "documents",
        sa.Column(
            "product_family_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("product_families.id"),
            nullable=True,
        ),
    )
    # A document's trays must all belong to its family. That's enforced in
    # documents.repository rather than with a composite FK: re-tagging a
    # document to another family replaces its trays in the same
    # transaction anyway, and the repository can say which trays were
    # wrong.
    op.create_table(
        "document_trays",
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "tray_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("trays.id"),
            primary_key=True,
        ),
    )
    op.create_index("document_trays_tray_id_idx", "document_trays", ["tray_id"])
    # Denormalized from `documents`, same as chunks.system_id/
    # document_type_id: the family filter a later ticket adds to retrieval
    # reads it without joining back. Trays deliberately aren't copied onto
    # chunks.
    op.add_column(
        "chunks",
        sa.Column(
            "product_family_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("product_families.id"),
            nullable=True,
        ),
    )
    op.create_index("chunks_product_family_id_idx", "chunks", ["product_family_id"])


def downgrade() -> None:
    op.drop_index("chunks_product_family_id_idx", table_name="chunks")
    op.drop_column("chunks", "product_family_id")
    op.drop_index("document_trays_tray_id_idx", table_name="document_trays")
    op.drop_table("document_trays")
    op.drop_column("documents", "product_family_id")
