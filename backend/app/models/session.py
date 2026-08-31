"""
OASIS — Session and TranscriptEntry models.

A Session represents a single interview between a participant and an agent.
TranscriptEntry holds each diarized (user/agent) utterance within that session.
"""

import enum
import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Enum, Float, ForeignKey, Integer, JSON, String, Text, UniqueConstraint, func, select
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, backref, mapped_column, relationship

from app.models.base import Base


class SessionStatus(str, enum.Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    TIMED_OUT = "timed_out"
    ERROR = "error"


class AudioRecordingStatus(str, enum.Enum):
    NONE = "none"
    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILED = "failed"


class SpeakerRole(str, enum.Enum):
    USER = "user"
    AGENT = "agent"
    SYSTEM = "system"


class Session(Base):
    __tablename__ = "sessions"
    __table_args__ = (
        # Prolific's SESSION_ID identifies a single submission. Nulls remain
        # unrestricted for direct-link and legacy sessions, while an external
        # submission cannot accidentally create two transcripts for one agent.
        UniqueConstraint(
            "agent_id",
            "external_session_id",
            name="uq_sessions_agent_external_session",
        ),
    )

    # ── Ownership ──
    agent_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("agents.id", ondelete="CASCADE"),
        nullable=False,
    )

    # ── Status ──
    status: Mapped[SessionStatus] = mapped_column(
        Enum(
            SessionStatus,
            name="session_status",
            values_callable=lambda e: [member.value for member in e],
        ),
        default=SessionStatus.ACTIVE,
        server_default=SessionStatus.ACTIVE.value,
    )

    # ── Metrics ──
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True, default=0)
    ended_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # ── Participant metadata ──
    participant_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    recruitment_source: Mapped[str | None] = mapped_column(String(32), nullable=True)
    external_study_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    external_session_id: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Immutable audit snapshot of the protocol used to start this session.
    research_protocol_version: Mapped[str | None] = mapped_column(
        String(32), nullable=True
    )
    research_stage: Mapped[str | None] = mapped_column(String(16), nullable=True)
    protocol_snapshot: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    completion_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    participant_consent_confirmed: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default="false"
    )
    participant_consent_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # ── Audio recording (voice web interviews) ──
    audio_recording_enabled: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default="false"
    )
    audio_storage_uri: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    audio_recording_status: Mapped[str] = mapped_column(
        String(32),
        default=AudioRecordingStatus.NONE.value,
        server_default=AudioRecordingStatus.NONE.value,
    )

    # ── Adaptive behavior (set at session start when live) ──
    adaptive_active: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default="false"
    )

    # ── Relationships ──
    agent = relationship(
        "Agent",
        backref=backref("sessions", cascade="all, delete-orphan", passive_deletes=True),
    )
    entries = relationship(
        "TranscriptEntry",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="TranscriptEntry.sequence",
    )
    engagement_turns = relationship(
        "EngagementTurn",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="EngagementTurn.transcript_sequence",
    )
    engagement_events = relationship(
        "EngagementEvent",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="EngagementEvent.created_at",
    )
    adaptive_actions = relationship(
        "AdaptiveAction",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="AdaptiveAction.created_at",
    )


class TranscriptEntry(Base):
    __tablename__ = "transcript_entries"

    # ── Ownership ──
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
    )

    # ── Content ──
    role: Mapped[SpeakerRole] = mapped_column(
        Enum(
            SpeakerRole,
            name="speaker_role",
            values_callable=lambda e: [member.value for member in e],
        ),
        nullable=False,
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # ── Ordering ──
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)

    # ── Token metrics (for agent responses) ──
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # ── Timestamp ──
    spoken_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    # ── Relationships ──
    session = relationship("Session", back_populates="entries")


async def aggregate_session_tokens(db: AsyncSession, session_id: uuid.UUID) -> int:
    """Return the sum of prompt + completion tokens across a session's entries.

    Returns ``0`` when no transcript entries have token counts. This is used by
    every interview-finalize path to populate ``Session.total_tokens`` so the
    dashboard cost/usage column is accurate.
    """
    prompt_total = await db.scalar(
        select(func.coalesce(func.sum(TranscriptEntry.prompt_tokens), 0)).where(
            TranscriptEntry.session_id == session_id
        )
    )
    completion_total = await db.scalar(
        select(func.coalesce(func.sum(TranscriptEntry.completion_tokens), 0)).where(
            TranscriptEntry.session_id == session_id
        )
    )
    return int(prompt_total or 0) + int(completion_total or 0)
