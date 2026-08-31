"""add participant profile metadata

Revision ID: f7a8b9c0d1e2
Revises: e6f7a8b9c0d1
"""

from typing import Union

from alembic import op
import sqlalchemy as sa


revision: str = "f7a8b9c0d1e2"
down_revision: Union[str, None] = "e6f7a8b9c0d1"
branch_labels: Union[str, None] = None
depends_on: Union[str, None] = None


def upgrade() -> None:
    op.add_column(
        "participant_identifiers",
        sa.Column("profile", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("participant_identifiers", "profile")
