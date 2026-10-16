"""create question type document types

Revision ID: b7d3e94a1c58
Revises: ec08a13c13dc
Create Date: 2026-10-16 15:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'b7d3e94a1c58'
down_revision: str | Sequence[str] | None = 'ec08a13c13dc'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# The question type -> doc type priority mapping that lived in code
# (agents/workflows/deterministic.py's _DOCTYPE_PRIORITY) until issue #17,
# frozen here as the seed. Matched against existing document type names by
# case-insensitive substring -- exactly how that code matched them -- so
# the seeded rows reproduce the retrieval filter and rerank order it
# produced. features_benefits is new, from the stakeholders' hierarchy CSV
# (backend/evals/doctype-hierarchy.csv): brochure/surgical technique guide,
# then launch presentation. The CSV's "Master Item File" tier isn't a
# document type (it's the graph) and is skipped. documents_lookup and
# unclassified get no rows: no doc-type filter.
SEED_PRIORITY: dict[str, list[str]] = {
    "system_contents": ["Inventory Control", "Tray Layout", "Setup Guide", "Surgical Technique"],
    "product_characteristics": [
        "Inventory Control", "Tray Layout", "Setup Guide", "Surgical Technique",
    ],
    "compatibility_lookup": [
        "Inventory Control", "Tray Layout", "Setup Guide", "Surgical Technique",
    ],
    "technique_procedural": ["Surgical Technique", "Setup Guide", "Launch Presentation"],
    "features_benefits": ["Brochure", "Surgical Technique", "Launch Presentation"],
}


def seed_rows(document_types: list[tuple[str, str]]) -> list[tuple[str, str, int]]:
    """(question_type, document_type_id, rank) rows for the given
    (id, name) document types. A document type ranks at the first seed name
    it contains, matching the old code's first-match bonus; several types
    matching the same name rank in name order.
    """
    rows: list[tuple[str, str, int]] = []
    by_name = sorted(document_types, key=lambda document_type: document_type[1].lower())
    for question_type, seed_names in SEED_PRIORITY.items():
        ranked_ids: list[str] = []
        for seed_name in seed_names:
            for document_type_id, document_type_name in by_name:
                if seed_name.lower() in document_type_name.lower() and document_type_id not in ranked_ids:
                    ranked_ids.append(document_type_id)
        rows.extend(
            (question_type, document_type_id, rank) for rank, document_type_id in enumerate(ranked_ids)
        )
    return rows


def upgrade() -> None:
    # One row per (question type, document type), ranked within its
    # question type -- rank 0 is the most authoritative. Question type
    # slugs are code (agents/question_types.py), not a table, so there's no
    # FK on that side; the route validates them. Deleting a document type
    # (only possible once no document uses it) drops its rows; merging one
    # re-points them (tags.repository.merge_document_type).
    op.create_table(
        "question_type_document_types",
        sa.Column("question_type", sa.Text, primary_key=True),
        sa.Column(
            "document_type_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("document_types.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("rank", sa.Integer, nullable=False),
        # unclassified always retrieves unfiltered -- never give it rows.
        sa.CheckConstraint(
            "question_type <> 'unclassified'", name="question_type_document_types_not_unclassified"
        ),
    )
    op.create_index(
        "question_type_document_types_document_type_idx",
        "question_type_document_types",
        ["document_type_id"],
    )

    connection = op.get_bind()
    document_types = [
        (row[0], row[1])
        for row in connection.execute(sa.text("SELECT id::text, name FROM document_types")).all()
    ]
    for question_type, document_type_id, rank in seed_rows(document_types):
        connection.execute(
            sa.text(
                "INSERT INTO question_type_document_types (question_type, document_type_id, rank) "
                "VALUES (:question_type, CAST(:document_type_id AS uuid), :rank)"
            ),
            {"question_type": question_type, "document_type_id": document_type_id, "rank": rank},
        )


def downgrade() -> None:
    op.drop_index(
        "question_type_document_types_document_type_idx", table_name="question_type_document_types"
    )
    op.drop_table("question_type_document_types")
