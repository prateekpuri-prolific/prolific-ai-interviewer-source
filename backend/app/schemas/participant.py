"""
OASIS — Pydantic schemas for Participant Identifiers.
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ParticipantIdentifierCreate(BaseModel):
    identifier: str = Field(..., min_length=1, max_length=255)
    label: str | None = None
    profile: dict | None = None


class ParticipantIdentifierBulkCreate(BaseModel):
    """Create plain IDs or richer participant rows with role metadata."""
    identifiers: list[str] = Field(default_factory=list)
    participants: list[ParticipantIdentifierCreate] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_rows(self):
        if not self.identifiers and not self.participants:
            raise ValueError("Provide identifiers or participants")
        return self


class ParticipantIdentifierRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    agent_id: UUID
    identifier: str
    label: str | None
    profile: dict | None
    used: bool
    session_id: UUID | None
    created_at: datetime
    updated_at: datetime
