"""add research protocol and immutable session snapshots

Revision ID: e6f7a8b9c0d1
Revises: d5e6f7a8b9c0
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "e6f7a8b9c0d1"
down_revision: Union[str, None] = "d5e6f7a8b9c0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("studies", sa.Column("research_spec", sa.JSON(), nullable=True))
    op.add_column("sessions", sa.Column("research_protocol_version", sa.String(32), nullable=True))
    op.add_column("sessions", sa.Column("research_stage", sa.String(16), nullable=True))
    op.add_column("sessions", sa.Column("protocol_snapshot", sa.JSON(), nullable=True))
    op.add_column(
        "sessions",
        sa.Column(
            "participant_consent_confirmed", sa.Boolean(), nullable=False,
            server_default=sa.false(),
        ),
    )


def downgrade() -> None:
    op.drop_column("sessions", "participant_consent_confirmed")
    op.drop_column("sessions", "protocol_snapshot")
    op.drop_column("sessions", "research_stage")
    op.drop_column("sessions", "research_protocol_version")
    op.drop_column("studies", "research_spec")
