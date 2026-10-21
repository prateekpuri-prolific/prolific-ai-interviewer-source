from datetime import datetime, timezone
from unittest.mock import AsyncMock
import uuid

import pytest
from sqlalchemy import select

from app.models.agent import ParticipantIdentifier
from app.models.session import Session, SessionStatus, TranscriptEntry, SpeakerRole
from app.models.engagement import EngagementTurn, EngagementEvent, AdaptiveAction
from app.models.privacy_lifecycle import PrivacyLifecyclePolicy, PrivacyDeletionReceipt
from app.models.study import StudyStatus
from app.privacy_lifecycle import add_months, delete_source_session, retention_candidates

NOW = datetime(2026, 9, 8, tzinfo=timezone.utc)


async def seed(db, make_study, make_agent, *, active=False):
    study = make_study(status=StudyStatus.COMPLETED)
    db.add(study)
    await db.flush()
    agent = make_agent(study_id=study.id)
    db.add(agent)
    await db.flush()
    session = Session(agent_id=agent.id, participant_id="synthetic-donor", status=SessionStatus.ACTIVE if active else SessionStatus.COMPLETED,
        post_interview_feedback={"comments": "synthetic feedback"}, protocol_snapshot={"synthetic": True})
    db.add(session)
    await db.flush()
    db.add_all([
        PrivacyLifecyclePolicy(study_id=study.id, owner="Test operator", project_completed_at=NOW.replace(month=3), enabled=True),
        TranscriptEntry(session_id=session.id, role=SpeakerRole.USER, sequence=1, content="synthetic canary"),
        EngagementTurn(session_id=session.id, transcript_sequence=1, extras={"text": "synthetic"}),
        EngagementEvent(session_id=session.id, event_type="test", payload={"text": "synthetic"}),
        AdaptiveAction(session_id=session.id, trigger="test", action="test", mode="shadow", detail={"text": "synthetic"}),
        ParticipantIdentifier(agent_id=agent.id, identifier="synthetic-donor", session_id=session.id, profile={"text": "synthetic"}),
    ])
    await db.commit()
    return study, agent, session


def test_calendar_months_leap_year_and_boundary():
    assert add_months(datetime(2023, 8, 31, tzinfo=timezone.utc), 6).isoformat() == "2024-02-29T00:00:00+00:00"
    assert add_months(datetime(2026, 1, 31), 1).day == 28


async def test_withdrawal_deletes_all_application_sources_and_is_idempotent(db_session, make_study, make_agent):
    study, agent, session = await seed(db_session, make_study, make_agent)
    sid = session.id
    redis = AsyncMock()
    r = await delete_source_session(db_session, study_id=study.id, session_id=sid,
        reason="withdrawal", storage=None, redis=redis, now=NOW)
    assert r.status == "application_deleted_external_pending"
    assert all(v == "pending" for v in r.external_checks.values())
    for model in (Session, TranscriptEntry, EngagementTurn, EngagementEvent, AdaptiveAction, ParticipantIdentifier):
        assert not (await db_session.scalars(select(model))).all()
    again = await delete_source_session(db_session, study_id=study.id, session_id=sid,
        reason="withdrawal", storage=None, redis=redis, now=NOW)
    assert again.id == r.id
    redis.delete.assert_awaited_once()


async def test_active_and_failed_external_cleanup_preserve_source_for_retry(db_session, make_study, make_agent):
    study, agent, session = await seed(db_session, make_study, make_agent, active=True)
    redis = AsyncMock()
    kwargs = dict(study_id=study.id, session_id=session.id, reason="withdrawal", storage=None, redis=redis, now=NOW)
    r = await delete_source_session(db_session, **kwargs)
    assert r.status == "blocked_active_session"
    redis.delete.assert_not_awaited()
    session.status = SessionStatus.COMPLETED
    await db_session.commit()
    redis.delete.side_effect = RuntimeError("RAW SECRET MUST NOT ENTER RECEIPT")
    r = await delete_source_session(db_session, **kwargs)
    assert r.status == "failed" and r.error_code == "RuntimeError"
    assert await db_session.scalar(select(TranscriptEntry.id))
    redis.delete.side_effect = None
    r = await delete_source_session(db_session, **kwargs)
    assert r.application_deleted_at


async def test_retention_requires_opt_in_completed_project_and_due_date(db_session, make_study, make_agent):
    study, _, session = await seed(db_session, make_study, make_agent)
    assert await retention_candidates(db_session, NOW) == [(study.id, session.id)]
    assert await retention_candidates(db_session, NOW.replace(day=7)) == []
    study.status = StudyStatus.ACTIVE
    await db_session.commit()
    assert await retention_candidates(db_session, NOW) == []
    with pytest.raises(ValueError, match="not due"):
        await delete_source_session(db_session, study_id=study.id, session_id=session.id,
            reason="retention", storage=None, redis=AsyncMock(), now=NOW)


async def test_wrong_study_and_missing_owner_cannot_delete(db_session, make_study, make_agent):
    study, _, session = await seed(db_session, make_study, make_agent)
    with pytest.raises(ValueError):
        await delete_source_session(db_session, study_id=uuid.uuid4(), session_id=session.id,
            reason="withdrawal", storage=None, redis=AsyncMock())
    policy = await db_session.scalar(select(PrivacyLifecyclePolicy))
    policy.owner = " "
    await db_session.commit()
    with pytest.raises(ValueError, match="owner"):
        await delete_source_session(db_session, study_id=study.id, session_id=session.id,
            reason="withdrawal", storage=None, redis=AsyncMock())


async def test_audio_failure_prevents_false_success(db_session, make_study, make_agent):
    study, _, session = await seed(db_session, make_study, make_agent)
    session.audio_recording_enabled = True
    await db_session.commit()
    storage = AsyncMock()
    storage.delete_prefix.side_effect = RuntimeError("S3 failed")
    r = await delete_source_session(db_session, study_id=study.id, session_id=session.id,
        reason="withdrawal", storage=storage, redis=AsyncMock())
    assert r.status == "failed"
    assert await db_session.scalar(select(TranscriptEntry.id))
