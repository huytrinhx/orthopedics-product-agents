"""create chunks table

Revision ID: 4b7c2d00de10
Revises: a796a69506ed
Create Date: 2026-08-31 16:10:59.541203

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '4b7c2d00de10'
down_revision: str | Sequence[str] | None = 'a796a69506ed'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# text-embedding-3-small (backend/config/llm_clients.py) is 1536-dimensional.
EMBEDDING_DIM = 1536


def upgrade() -> None:
    # Production Postgres is Supabase, pgvector-enabled (see README) -- this
    # is a no-op there, not a deployment risk.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        # ON DELETE CASCADE so a document delete can't orphan its chunks --
        # a DB-level cascade can't be forgotten by some future direct-delete
        # code path the way an application-level cleanup call could.
        sa.Column("document_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("chunk_index", sa.Integer, nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        # Detected heading (PDF: font-size heuristic, .md: literal syntax) --
        # null for the plain-text paragraph-fallback path. See
        # backend/ingestion/chunking.py.
        sa.Column("section_title", sa.Text, nullable=True),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=False),
        # Generated from `content` -- always in sync, never written directly.
        sa.Column(
            "tsv",
            postgresql.TSVECTOR,
            sa.Computed("to_tsvector('english', content)", persisted=True),
            nullable=False,
        ),
        # Denormalized from `documents` at ingest time (not joined at query
        # time) so retrieval filtering doesn't need a join back to
        # `documents` -- see backend/retrieval/vector_store.py. A re-tag
        # already fully re-triggers ingestion, so these stay in sync without
        # extra bookkeeping.
        sa.Column("system_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("systems.id"), nullable=True),
        sa.Column("document_type_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("document_types.id"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("chunks_document_id_idx", "chunks", ["document_id"])
    op.create_index("chunks_tsv_idx", "chunks", ["tsv"], postgresql_using="gin")
    # HNSW (pgvector >=0.5) rather than IVFFlat -- no training-set-size
    # tuning needed, and the pgvector/pgvector:pg16 image is current enough
    # to have it.
    op.execute(
        "CREATE INDEX chunks_embedding_hnsw_idx ON chunks "
        "USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    op.drop_table("chunks")
    # Extension is left in place -- other objects/future migrations may
    # depend on it, and CREATE EXTENSION IF NOT EXISTS makes re-upgrading
    # idempotent either way.
