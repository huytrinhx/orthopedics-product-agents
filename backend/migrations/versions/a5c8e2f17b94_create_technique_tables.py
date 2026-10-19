"""create technique tables

Revision ID: a5c8e2f17b94
Revises: d9d071ea3bda
Create Date: 2026-10-19 10:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'a5c8e2f17b94'
down_revision: str | Sequence[str] | None = 'd9d071ea3bda'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Issue #28: a document's surgical techniques, their ordered steps, and
    # each step's items with the SKU set they resolve to
    # (backend/techniques/). One technique belongs to one document, and a
    # re-index deletes the document's techniques (cascading to steps and
    # items) and writes the new extraction, the document_skus pattern.
    # Chunks are referenced by chunk_index, the same number a citation id
    # `{document_id}#{chunk_index}` carries; no FK to `chunks`, which the
    # vector leg rewrites in the same indexing run.
    op.create_table(
        "techniques",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        # The technique's position in the document, from 1.
        sa.Column("technique_order", sa.Integer, nullable=False),
        sa.Column("name", sa.Text, nullable=False),
        # The surgeon's choices the step SKU sets depend on, e.g.
        # {screw_diameter, bridge_size}.
        sa.Column(
            "parameters",
            postgresql.ARRAY(sa.Text),
            nullable=False,
            server_default=sa.text("'{}'::text[]"),
        ),
        # Catalog tray names the text says the technique needs, resolved in
        # code from what it wrote ("Procedure requires Core Foot Recon
        # Plating System").
        sa.Column(
            "required_trays",
            postgresql.ARRAY(sa.Text),
            nullable=False,
            server_default=sa.text("'{}'::text[]"),
        ),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint("document_id", "technique_order", name="techniques_document_order"),
    )
    op.create_table(
        "technique_steps",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "technique_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("techniques.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("step_order", sa.Integer, nullable=False),
        # As the document labels it: "Step 6. Proximal Screw Preparation &
        # Insertion".
        sa.Column("step_label", sa.Text, nullable=False),
        # The chunks the step's text is in. Never empty: a step the model
        # tied to no real chunk isn't stored.
        sa.Column("source_chunk_indexes", postgresql.ARRAY(sa.Integer), nullable=False),
        sa.UniqueConstraint("technique_id", "step_order", name="technique_steps_technique_order"),
    )
    op.create_table(
        "technique_step_items",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "step_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("technique_steps.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("item_order", sa.Integer, nullable=False),
        # Verbatim from the step text: "Ø2.6mm drill".
        sa.Column("mention", sa.Text, nullable=False),
        sa.Column("item_type", sa.Text, nullable=False),
        # Only when the text literally names one.
        sa.Column("literal_sku", sa.Text, nullable=True),
        sa.Column(
            "specs", postgresql.JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")
        ),
        sa.Column("condition", sa.Text, nullable=True),
        sa.Column("optional", sa.Boolean, nullable=False, server_default=sa.false()),
        # The parameter values this item's set is for, e.g.
        # {"screw_diameter": "3.7"}; empty when it's used whatever the
        # surgeon picks. A step's items for one parameter's options are its
        # parameterised sets.
        sa.Column(
            "applies_when", postgresql.JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")
        ),
        # Resolved in code from the catalog and the document's SKU index,
        # never by the model.
        sa.Column(
            "resolved_skus",
            postgresql.ARRAY(sa.Text),
            nullable=False,
            server_default=sa.text("'{}'::text[]"),
        ),
        # How the set was found (techniques/resolution.py).
        sa.Column("resolution", sa.Text, nullable=False),
        # The step's chunks plus the chunks the document's SKU index places
        # the resolved SKUs in (the tray tables).
        sa.Column("source_chunk_indexes", postgresql.ARRAY(sa.Integer), nullable=False),
        sa.CheckConstraint(
            "resolution IN ('literal', 'in_document', 'catalog_only', 'unresolved')",
            name="technique_step_items_resolution",
        ),
        sa.UniqueConstraint("step_id", "item_order", name="technique_step_items_step_order"),
    )


def downgrade() -> None:
    op.drop_table("technique_step_items")
    op.drop_table("technique_steps")
    op.drop_table("techniques")
