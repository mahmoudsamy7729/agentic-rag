"""add retrieval evaluation tables

Revision ID: a1b2c3d4e5f6
Revises: e6f7a8b9c0d1
Create Date: 2026-04-09 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, Sequence[str], None] = "e6f7a8b9c0d1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    json_type = sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")

    op.create_table(
        "evaluation_runs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("owner_user_id", sa.Uuid(), nullable=False),
        sa.Column("doc_id", sa.String(length=255), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("evaluation_type", sa.String(length=32), nullable=False),
        sa.Column("dataset_name", sa.String(length=512), nullable=False),
        sa.Column("dataset_path", sa.String(length=1024), nullable=False),
        sa.Column("dataset_sha256", sa.String(length=64), nullable=False),
        sa.Column("total_cases", sa.Integer(), nullable=False),
        sa.Column("processed_cases", sa.Integer(), nullable=False),
        sa.Column("k", sa.Integer(), nullable=False),
        sa.Column("config_snapshot", json_type, nullable=False),
        sa.Column("hit_at_k_avg", sa.Float(), nullable=True),
        sa.Column("recall_at_k_avg", sa.Float(), nullable=True),
        sa.Column("precision_at_k_avg", sa.Float(), nullable=True),
        sa.Column("mrr_avg", sa.Float(), nullable=True),
        sa.Column("keyword_coverage_avg", sa.Float(), nullable=True),
        sa.Column("context_relevance_score_avg", sa.Float(), nullable=True),
        sa.Column("grouped_summary", json_type, nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["doc_id"], ["documents.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["owner_user_id"], ["user.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_evaluation_runs_owner_user_id"),
        "evaluation_runs",
        ["owner_user_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_evaluation_runs_doc_id"),
        "evaluation_runs",
        ["doc_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_evaluation_runs_status"),
        "evaluation_runs",
        ["status"],
        unique=False,
    )

    op.create_table(
        "evaluation_cases",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("case_index", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("reference_answer", sa.Text(), nullable=False),
        sa.Column("must_include_keywords", json_type, nullable=False),
        sa.Column("must_include_phrases", json_type, nullable=False),
        sa.Column("difficulty", sa.String(length=64), nullable=True),
        sa.Column("category", sa.String(length=64), nullable=True),
        sa.Column("retrieved_chunk_ids", json_type, nullable=False),
        sa.Column("retrieved_chunk_texts", json_type, nullable=False),
        sa.Column("matched_phrases", json_type, nullable=False),
        sa.Column("matched_keywords", json_type, nullable=False),
        sa.Column("hit_at_k", sa.Float(), nullable=True),
        sa.Column("recall_at_k", sa.Float(), nullable=True),
        sa.Column("precision_at_k", sa.Float(), nullable=True),
        sa.Column("mrr", sa.Float(), nullable=True),
        sa.Column("keyword_coverage", sa.Float(), nullable=True),
        sa.Column("context_relevance_score", sa.Integer(), nullable=True),
        sa.Column("context_relevance_explanation", sa.Text(), nullable=True),
        sa.Column("first_correct_rank", sa.Integer(), nullable=True),
        sa.Column("useful_chunk_count", sa.Integer(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["run_id"], ["evaluation_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_evaluation_cases_run_id"),
        "evaluation_cases",
        ["run_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_evaluation_cases_status"),
        "evaluation_cases",
        ["status"],
        unique=False,
    )
    op.create_index(
        "ix_evaluation_cases_run_case_index",
        "evaluation_cases",
        ["run_id", "case_index"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("ix_evaluation_cases_run_case_index", table_name="evaluation_cases")
    op.drop_index(op.f("ix_evaluation_cases_status"), table_name="evaluation_cases")
    op.drop_index(op.f("ix_evaluation_cases_run_id"), table_name="evaluation_cases")
    op.drop_table("evaluation_cases")

    op.drop_index(op.f("ix_evaluation_runs_status"), table_name="evaluation_runs")
    op.drop_index(op.f("ix_evaluation_runs_doc_id"), table_name="evaluation_runs")
    op.drop_index(op.f("ix_evaluation_runs_owner_user_id"), table_name="evaluation_runs")
    op.drop_table("evaluation_runs")
