"""remove evaluation tables

Revision ID: e6f7a8b9c0d1
Revises: d3e4f5a6b7c8
Create Date: 2026-04-09 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e6f7a8b9c0d1"
down_revision: Union[str, Sequence[str], None] = "d3e4f5a6b7c8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("DROP TABLE IF EXISTS evaluation_cases CASCADE")
    op.execute("DROP TABLE IF EXISTS evaluation_runs CASCADE")


def downgrade() -> None:
    raise RuntimeError("Downgrade is not supported after evaluation hard reset.")
