"""Add opt-in retention policies and deletion receipts."""
from alembic import op
import sqlalchemy as sa

revision = "d1e2f3a4b5c6"
down_revision = "c0d1e2f3a4b5"
branch_labels = None
depends_on = None


def common():
    return [sa.Column("id", sa.Uuid(), primary_key=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False)]


def upgrade():
    op.create_table("privacy_lifecycle_policies", *common(),
        sa.Column("study_id", sa.Uuid(), nullable=False, unique=True),
        sa.Column("owner", sa.String(255), nullable=False),
        sa.Column("project_completed_at", sa.DateTime(timezone=True)),
        sa.Column("enabled", sa.Boolean(), nullable=False))
    op.create_table("privacy_deletion_receipts", *common(),
        sa.Column("study_id", sa.Uuid(), nullable=False, index=True),
        sa.Column("session_id", sa.Uuid(), nullable=False, unique=True),
        sa.Column("reason", sa.String(32), nullable=False),
        sa.Column("owner", sa.String(255), nullable=False),
        sa.Column("status", sa.String(48), nullable=False),
        sa.Column("error_code", sa.String(64)),
        sa.Column("application_deleted_at", sa.DateTime(timezone=True)),
        sa.Column("external_checks", sa.JSON(), nullable=False))


def downgrade():
    op.drop_table("privacy_deletion_receipts")
    op.drop_table("privacy_lifecycle_policies")
