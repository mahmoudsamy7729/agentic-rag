"""add evaluation tables

Revision ID: b4e9d2a1f0c7
Revises: 8a12c4f4d6b1
Create Date: 2026-03-26 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "b4e9d2a1f0c7"
down_revision: Union[str, Sequence[str], None] = "8a12c4f4d6b1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "evaluation_runs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("owner_user_id", sa.Uuid(), nullable=False),
        sa.Column("doc_id", sa.String(length=255), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("dataset_name", sa.String(length=512), nullable=False),
        sa.Column("dataset_sha256", sa.String(length=64), nullable=False),
        sa.Column("total_cases", sa.Integer(), nullable=False),
        sa.Column("processed_cases", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("hit_at_k", sa.Float(), nullable=True),
        sa.Column("recall_at_k", sa.Float(), nullable=True),
        sa.Column("mrr", sa.Float(), nullable=True),
        sa.Column("accuracy_avg", sa.Float(), nullable=True),
        sa.Column("completeness_avg", sa.Float(), nullable=True),
        sa.Column("relevance_avg", sa.Float(), nullable=True),
        sa.Column("groundedness_avg", sa.Float(), nullable=True),
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
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("reference_answer", sa.Text(), nullable=False),
        sa.Column(
            "expected_chunk_ids",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column("difficulty", sa.String(length=64), nullable=True),
        sa.Column("category", sa.String(length=64), nullable=True),
        sa.Column(
            "retrieved_chunk_ids",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column("hit", sa.Boolean(), nullable=True),
        sa.Column("recall", sa.Float(), nullable=True),
        sa.Column("first_relevant_rank", sa.Integer(), nullable=True),
        sa.Column("reciprocal_rank", sa.Float(), nullable=True),
        sa.Column("generated_answer", sa.Text(), nullable=True),
        sa.Column(
            "citations",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column("accuracy", sa.Integer(), nullable=True),
        sa.Column("completeness", sa.Integer(), nullable=True),
        sa.Column("relevance", sa.Integer(), nullable=True),
        sa.Column("groundedness", sa.Integer(), nullable=True),
        sa.Column("judge_feedback", sa.Text(), nullable=True),
        sa.Column(
            "status",
            sa.String(length=32),
            nullable=False,
            server_default=sa.text("'queued'"),
        ),
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
