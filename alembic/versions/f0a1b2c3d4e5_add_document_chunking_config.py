"""add document chunking config columns

Revision ID: f0a1b2c3d4e5
Revises: c92f3d1a7b4e
Create Date: 2026-04-01 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f0a1b2c3d4e5"
down_revision: Union[str, Sequence[str], None] = "c92f3d1a7b4e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("chunking_strategy", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "documents",
        sa.Column("chunk_size", sa.Integer(), nullable=True),
    )
    op.add_column(
        "documents",
        sa.Column("chunk_overlap", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("documents", "chunk_overlap")
    op.drop_column("documents", "chunk_size")
    op.drop_column("documents", "chunking_strategy")
