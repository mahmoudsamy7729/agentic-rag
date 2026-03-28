"""add evaluation run config snapshot columns

Revision ID: c92f3d1a7b4e
Revises: b4e9d2a1f0c7
Create Date: 2026-03-28 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c92f3d1a7b4e"
down_revision: Union[str, Sequence[str], None] = "b4e9d2a1f0c7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "evaluation_runs",
        sa.Column(
            "cfg_rag_top_k",
            sa.Integer(),
            nullable=True,
            server_default=sa.text("12"),
        ),
    )
    op.add_column(
        "evaluation_runs",
        sa.Column(
            "cfg_rag_prefetch_k",
            sa.Integer(),
            nullable=True,
            server_default=sa.text("80"),
        ),
    )
    op.add_column(
        "evaluation_runs",
        sa.Column(
            "cfg_embedding_provider",
            sa.String(length=64),
            nullable=True,
            server_default=sa.text("'huggingface'"),
        ),
    )
    op.add_column(
        "evaluation_runs",
        sa.Column(
            "cfg_embedding_model",
            sa.String(length=255),
            nullable=True,
            server_default=sa.text("'sentence-transformers/all-MiniLM-L6-v2'"),
        ),
    )
    op.add_column(
        "evaluation_runs",
        sa.Column(
            "cfg_reranker_enabled",
            sa.Boolean(),
            nullable=True,
            server_default=sa.text("true"),
        ),
    )
    op.add_column(
        "evaluation_runs",
        sa.Column(
            "cfg_reranker_model",
            sa.String(length=255),
            nullable=True,
            server_default=sa.text("'rerank-v4.0-fast'"),
        ),
    )
    op.add_column(
        "evaluation_runs",
        sa.Column(
            "cfg_answer_model",
            sa.String(length=255),
            nullable=True,
            server_default=sa.text("'gpt-oss:120b-cloud'"),
        ),
    )
    op.add_column(
        "evaluation_runs",
        sa.Column(
            "cfg_chunk_strategy",
            sa.String(length=64),
            nullable=True,
            server_default=sa.text("'fixed_window'"),
        ),
    )
    op.add_column(
        "evaluation_runs",
        sa.Column(
            "cfg_chunk_size",
            sa.Integer(),
            nullable=True,
            server_default=sa.text("800"),
        ),
    )
    op.add_column(
        "evaluation_runs",
        sa.Column(
            "cfg_chunk_overlap",
            sa.Integer(),
            nullable=True,
            server_default=sa.text("120"),
        ),
    )


def downgrade() -> None:
    op.drop_column("evaluation_runs", "cfg_chunk_overlap")
    op.drop_column("evaluation_runs", "cfg_chunk_size")
    op.drop_column("evaluation_runs", "cfg_chunk_strategy")
    op.drop_column("evaluation_runs", "cfg_answer_model")
    op.drop_column("evaluation_runs", "cfg_reranker_model")
    op.drop_column("evaluation_runs", "cfg_reranker_enabled")
    op.drop_column("evaluation_runs", "cfg_embedding_model")
    op.drop_column("evaluation_runs", "cfg_embedding_provider")
    op.drop_column("evaluation_runs", "cfg_rag_prefetch_k")
    op.drop_column("evaluation_runs", "cfg_rag_top_k")
