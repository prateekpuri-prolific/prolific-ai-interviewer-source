"""Operator-driven withdrawal and opt-in scheduled retention.

Application deletion is distinct from completion of external/backup checks.
All dates are UTC calendar dates; no 30-day legal-rights cutoff is enforced.
"""
import calendar
from datetime import datetime, timezone

from sqlalchemy import delete, or_, select

from app.models.agent import Agent, ParticipantIdentifier
from app.models.engagement import AdaptiveAction, EngagementEvent, EngagementTurn
from app.models.privacy_lifecycle import PrivacyDeletionReceipt, PrivacyLifecyclePolicy
from app.models.session import Session, SessionStatus, TranscriptEntry
from app.models.study import Study, StudyStatus
from app.audio.storage import build_session_prefix


def utc(value):
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def add_months(value: datetime, months: int) -> datetime:
    value = utc(value)
    year, month = divmod(value.year * 12 + value.month - 1 + months, 12)
    month += 1
    return value.replace(year=year, month=month, day=min(value.day, calendar.monthrange(year, month)[1]))


async def retention_candidates(db, now=None):
    now = utc(now or datetime.now(timezone.utc))
    policies = (await db.scalars(select(PrivacyLifecyclePolicy).where(
        PrivacyLifecyclePolicy.enabled.is_(True),
        PrivacyLifecyclePolicy.project_completed_at.is_not(None),
    ))).all()
    candidates = []
    for policy in policies:
        study = await db.get(Study, policy.study_id)
        if not study or study.status != StudyStatus.COMPLETED:
            continue
        if add_months(policy.project_completed_at, 6) <= now:
            ids = (await db.scalars(select(Session.id).join(Agent).where(Agent.study_id == policy.study_id))).all()
            candidates.extend((policy.study_id, sid) for sid in ids)
    return candidates


async def delete_source_session(db, *, study_id, session_id, reason, storage, redis, now=None):
    if reason not in {"withdrawal", "retention"}:
        raise ValueError("Unsupported deletion reason")
    now = utc(now or datetime.now(timezone.utc))
    # Serialize operators/scheduler per study. Never use a client-supplied
    # storage URI as a delete prefix.
    policy = await db.scalar(select(PrivacyLifecyclePolicy).where(
        PrivacyLifecyclePolicy.study_id == study_id).with_for_update())
    if not policy or not policy.owner.strip():
        raise ValueError("A named policy owner is required")
    receipt = await db.scalar(select(PrivacyDeletionReceipt).where(
        PrivacyDeletionReceipt.session_id == session_id))
    if receipt and receipt.study_id != study_id:
        raise ValueError("Session does not belong to study")
    if receipt and receipt.application_deleted_at:
        return receipt
    session = await db.get(Session, session_id)
    agent = await db.get(Agent, session.agent_id) if session else None
    if not session or not agent or agent.study_id != study_id:
        raise ValueError("Session does not belong to study")
    if reason == "retention":
        study = await db.get(Study, study_id)
        if (not policy.enabled or not policy.project_completed_at
                or study.status != StudyStatus.COMPLETED
                or add_months(policy.project_completed_at, 6) > now):
            raise ValueError("Retention is not due")
    if not receipt:
        receipt = PrivacyDeletionReceipt(study_id=study_id, session_id=session_id,
            reason=reason, owner=policy.owner, status="pending", external_checks={
                "backups_and_restore": "pending", "downloaded_exports": "pending",
                "derived_artifacts": "pending", "provider_retention": "pending",
            })
        db.add(receipt)
    # Never race an active writer. Operator stops the interview and retries.
    if session.status == SessionStatus.ACTIVE:
        receipt.status = "blocked_active_session"
        receipt.error_code = "stop_interview_then_retry"
        await db.commit()
        return receipt
    # Commit the request before external side effects, so a killed process
    # leaves a retryable receipt instead of an unaudited partial deletion.
    await db.commit()
    await db.scalar(select(PrivacyLifecyclePolicy).where(
        PrivacyLifecyclePolicy.study_id == study_id).with_for_update())
    await db.refresh(receipt)
    if receipt.application_deleted_at:
        return receipt
    await db.refresh(session)
    try:
        if session.audio_storage_uri or session.audio_recording_enabled:
            if storage is None:
                raise RuntimeError("audio_storage_unavailable")
            prefix = build_session_prefix(study_id=study_id, agent_id=agent.id,
                participant_id=session.participant_id, session_id=session.id)
            # Refuse silently deleting from a different backend after a config change.
            if session.audio_storage_uri and session.audio_storage_uri.rstrip("/") != storage.uri_for_prefix(prefix).rstrip("/"):
                raise RuntimeError("audio_storage_mismatch")
            await storage.delete_prefix(prefix + "/")
        await redis.delete(f"oasis:session:{session_id}")
        await redis.srem("oasis:active_sessions", str(session_id))
    except Exception as exc:
        receipt.status = "failed"
        # Exception strings can contain content, paths or credentials.
        receipt.error_code = type(exc).__name__
        await db.commit()
        return receipt
    for model in (TranscriptEntry, EngagementTurn, EngagementEvent, AdaptiveAction):
        await db.execute(delete(model).where(model.session_id == session_id))
    # Delete linked recruitment metadata too, rather than leaving labels/profiles.
    # Preserve an identifier if another source session still uses it; deleting
    # all of a donor's sessions will remove it on the final iteration.
    another = await db.scalar(select(Session.id).where(Session.agent_id == agent.id,
        Session.participant_id == session.participant_id, Session.id != session_id).limit(1))
    conditions = [ParticipantIdentifier.session_id == session_id]
    if not another and session.participant_id:
        conditions.append((ParticipantIdentifier.agent_id == agent.id) &
                          (ParticipantIdentifier.identifier == session.participant_id))
    await db.execute(delete(ParticipantIdentifier).where(or_(*conditions)))
    await db.execute(delete(Session).where(Session.id == session_id))
    receipt.status = "application_deleted_external_pending"
    receipt.error_code = None
    receipt.application_deleted_at = now
    await db.commit()
    return receipt


async def run_retention(db, *, storage, redis, now=None):
    outcomes = []
    for study_id, session_id in await retention_candidates(db, now):
        receipt = await delete_source_session(db, study_id=study_id,
            session_id=session_id, reason="retention", storage=storage, redis=redis, now=now)
        outcomes.append({"receipt_id": str(receipt.id), "status": receipt.status})
    return outcomes


async def retention_loop():
    """Hourly sweep; failures emit counts only and are retried next sweep."""
    import asyncio
    from loguru import logger
    from app.database import async_session_factory
    from app.audio.storage import get_audio_storage
    from app.redis import get_redis
    while True:
        try:
            async with async_session_factory() as db:
                results = await run_retention(db, storage=await get_audio_storage(), redis=await get_redis())
                failed = sum(r["status"] != "application_deleted_external_pending" for r in results)
                logger.info("Retention sweep: processed={} unresolved={}", len(results), failed)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.error("Retention sweep failed; operator action required")
        await asyncio.sleep(3600)
