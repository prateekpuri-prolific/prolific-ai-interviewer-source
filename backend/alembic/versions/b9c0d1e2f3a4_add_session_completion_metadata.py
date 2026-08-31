"""add session completion metadata

Revision ID: b9c0d1e2f3a4
Revises: a8b9c0d1e2f3
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "b9c0d1e2f3a4"
down_revision: Union[str, None] = "a8b9c0d1e2f3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("sessions", sa.Column("completion_metadata", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("sessions", "completion_metadata")
