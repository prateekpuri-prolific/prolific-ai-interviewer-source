"""add post-interview participant feedback

Revision ID: c0d1e2f3a4b5
Revises: b9c0d1e2f3a4
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "c0d1e2f3a4b5"
down_revision: Union[str, None] = "b9c0d1e2f3a4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "sessions",
        sa.Column("post_interview_feedback", sa.JSON(), nullable=True),
    )
    op.add_column(
        "sessions",
        sa.Column(
            "post_interview_feedback_submitted_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("sessions", "post_interview_feedback_submitted_at")
    op.drop_column("sessions", "post_interview_feedback")
