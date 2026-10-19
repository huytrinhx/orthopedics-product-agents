"""unique document filenames

Revision ID: a3f81c6e5d24
Revises: d9d071ea3bda
Create Date: 2026-10-19 10:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'a3f81c6e5d24'
down_revision: str | Sequence[str] | None = 'd9d071ea3bda'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Issue #31: one document per filename, ignoring case and surrounding
    # whitespace. The upload and replace-file routes check first and answer
    # 409 naming the existing document; this index is what stops two
    # concurrent uploads that both passed that check
    # (documents.repository.DuplicateFilename). Its expression must stay the
    # same as the repository's lookup, lower(btrim(filename)).
    #
    # Building it fails while duplicates exist, so name them instead of
    # leaving Postgres's bare "could not create unique index". An admin
    # deletes one copy of each before this can run (the issue's Rollout).
    duplicates = op.get_bind().execute(
        sa.text(
            "SELECT lower(btrim(filename)), array_agg(id::text ORDER BY created_at) "
            "FROM documents GROUP BY lower(btrim(filename)) HAVING count(*) > 1"
        )
    ).fetchall()
    if duplicates:
        raise RuntimeError(
            "Delete one copy of each duplicate document before running this migration: "
            + "; ".join(
                f"{filename!r} ({', '.join(document_ids)})" for filename, document_ids in duplicates
            )
        )
    op.execute(
        "CREATE UNIQUE INDEX documents_filename_unique_idx ON documents (lower(btrim(filename)))"
    )


def downgrade() -> None:
    op.execute("DROP INDEX documents_filename_unique_idx")
