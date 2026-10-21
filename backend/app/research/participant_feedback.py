"""Participant-safe authorization for the optional post-interview survey."""

from __future__ import annotations

import hashlib
import hmac
from uuid import UUID

from app.config import settings


def participant_feedback_token(widget_key: str, session_id: UUID) -> str:
    """Return an unguessable token scoped to one widget and session."""
    message = f"participant-feedback:v1:{widget_key}:{session_id}"
    return hmac.new(
        settings.secret_key.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def valid_participant_feedback_token(
    supplied: str, widget_key: str, session_id: UUID
) -> bool:
    expected = participant_feedback_token(widget_key, session_id)
    return hmac.compare_digest(str(supplied or ""), expected)
