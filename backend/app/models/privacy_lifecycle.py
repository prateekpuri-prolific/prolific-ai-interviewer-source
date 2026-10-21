"""Opt-in study retention policy and minimal deletion receipts.

Receipts deliberately have no FK to deleted source records. Session UUIDs are
restricted tombstones for restore reconciliation, not anonymous statistics.
"""
import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, JSON, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class PrivacyLifecyclePolicy(Base):
    __tablename__ = "privacy_lifecycle_policies"
    study_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), unique=True)
    owner: Mapped[str] = mapped_column(String(255))
    project_completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    enabled: Mapped[bool] = mapped_column(Boolean, default=False)


class PrivacyDeletionReceipt(Base):
    __tablename__ = "privacy_deletion_receipts"
    study_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), index=True)
    session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), unique=True)
    reason: Mapped[str] = mapped_column(String(32))
    owner: Mapped[str] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(48), default="pending")
    error_code: Mapped[str | None] = mapped_column(String(64))
    application_deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    external_checks: Mapped[dict] = mapped_column(JSON, default=dict)
