"""add keyword-based evaluation fields

Revision ID: d3e4f5a6b7c8
Revises: f0a1b2c3d4e5
Create Date: 2026-04-09 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "d3e4f5a6b7c8"
down_revision: Union[str, Sequence[str], None] = "f0a1b2c3d4e5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    json_type = sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")

    op.add_column(
        "evaluation_runs",
        sa.Column("keyword_coverage_avg", sa.Float(), nullable=True),
    )
    op.add_column(
        "evaluation_cases",
        sa.Column("must_include_keywords", json_type, nullable=True),
    )
    op.add_column(
        "evaluation_cases",
        sa.Column("keyword_coverage", sa.Float(), nullable=True),
    )
    op.add_column(
        "evaluation_cases",
        sa.Column("failure_type", sa.String(length=32), nullable=True),
    )
    op.execute("UPDATE evaluation_cases SET must_include_keywords = '[]' WHERE must_include_keywords IS NULL")
    op.alter_column("evaluation_cases", "must_include_keywords", nullable=False)


def downgrade() -> None:
    op.drop_column("evaluation_cases", "failure_type")
    op.drop_column("evaluation_cases", "keyword_coverage")
    op.drop_column("evaluation_cases", "must_include_keywords")
    op.drop_column("evaluation_runs", "keyword_coverage_avg")
