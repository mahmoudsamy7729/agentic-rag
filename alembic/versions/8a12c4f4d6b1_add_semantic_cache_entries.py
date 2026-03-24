"""add semantic cache entries

Revision ID: 8a12c4f4d6b1
Revises: 5e31e0045d8e
Create Date: 2026-03-24 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "8a12c4f4d6b1"
down_revision: Union[str, Sequence[str], None] = "5e31e0045d8e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.add_column(
        "documents",
        sa.Column("last_indexed_at", sa.DateTime(timezone=True), nullable=True),
    )

    op.create_table(
        "semantic_cache_entries",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("owner_user_id", sa.Uuid(), nullable=False),
        sa.Column("doc_id", sa.String(length=255), nullable=False),
        sa.Column("doc_version", sa.DateTime(timezone=True), nullable=False),
        sa.Column("model_name", sa.String(length=255), nullable=False),
        sa.Column("question_normalized", sa.Text(), nullable=False),
        sa.Column("question_embedding", Vector(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=False),
        sa.Column("citations", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["doc_id"], ["documents.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["owner_user_id"], ["user.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index(
        op.f("ix_semantic_cache_entries_owner_user_id"),
        "semantic_cache_entries",
        ["owner_user_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_semantic_cache_entries_doc_id"),
        "semantic_cache_entries",
        ["doc_id"],
        unique=False,
    )
    op.create_index(
        "ix_semantic_cache_entries_owner_doc_version_model",
        "semantic_cache_entries",
        ["owner_user_id", "doc_id", "doc_version", "model_name"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(
        "ix_semantic_cache_entries_owner_doc_version_model",
        table_name="semantic_cache_entries",
    )
    op.drop_index(
        op.f("ix_semantic_cache_entries_doc_id"),
        table_name="semantic_cache_entries",
    )
    op.drop_index(
        op.f("ix_semantic_cache_entries_owner_user_id"),
        table_name="semantic_cache_entries",
    )
    op.drop_table("semantic_cache_entries")
    op.drop_column("documents", "last_indexed_at")
