"""add external recruitment identifiers

Revision ID: a8b9c0d1e2f3
Revises: f7a8b9c0d1e2
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "a8b9c0d1e2f3"
down_revision: Union[str, None] = "f7a8b9c0d1e2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # PostgreSQL enums must be extended explicitly; SQLAlchemy's Enum does not
    # add new labels when the model changes.
    op.execute("ALTER TYPE participant_id_mode ADD VALUE IF NOT EXISTS 'external'")
    op.add_column("sessions", sa.Column("recruitment_source", sa.String(32), nullable=True))
    op.add_column("sessions", sa.Column("external_study_id", sa.String(255), nullable=True))
    op.add_column("sessions", sa.Column("external_session_id", sa.String(255), nullable=True))
    op.add_column(
        "sessions",
        sa.Column("participant_consent_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_unique_constraint(
        "uq_sessions_agent_external_session",
        "sessions",
        ["agent_id", "external_session_id"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_sessions_agent_external_session", "sessions", type_="unique"
    )
    op.drop_column("sessions", "external_session_id")
    op.drop_column("sessions", "external_study_id")
    op.drop_column("sessions", "recruitment_source")
    op.drop_column("sessions", "participant_consent_at")
    # PostgreSQL enum values cannot be removed safely in-place. Leaving the
    # unused label is the non-destructive downgrade behavior.
