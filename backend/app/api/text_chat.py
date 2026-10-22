"""
OASIS — WebSocket endpoint for text-based chat interviews.

When an agent has modality="text", the frontend connects to:
    ws://host/ws/chat/{widget_key}?pid=<optional>

This endpoint:
1. Looks up the agent by widget_key
2. Resolves participant_id
3. Creates a Session
4. Runs a simple LLM chat loop (no Pipecat, no audio)
5. Logs transcript entries for researcher review
"""

import asyncio
import json
import secrets
import time
import uuid
from collections import Counter
from datetime import datetime, timezone

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from loguru import logger
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.config import settings
from app.database import async_session_factory
from app.interview_timing import build_interview_timing
from app.models.agent import Agent, AgentStatus, ParticipantIdMode, ParticipantIdentifier
from app.models.session import (
    Session,
    SessionStatus,
    TranscriptEntry,
    SpeakerRole,
    aggregate_session_tokens,
)
from app.models.study import Study
from app.realtime import publish_transcript_event
from app.providers.catalog import resolve_llm_api_kind
from app.providers.validate import validate_agent_pipeline_config
from app.privacy import (
    PRIVACY_WARNING,
    redact_session_data,
    redact_text,
    redact_text_with_llm,
)
from app.research.protocol import (
    build_protocol_snapshot,
    build_research_context,
    consent_required,
    participant_information,
)
from app.research.participant_feedback import participant_feedback_token
from app.session_manager import register_session, unregister_session

router = APIRouter()

_MAX_CHAT_DURATION = 7200  # 2 hours absolute ceiling
_KEEPALIVE_INTERVAL = 20  # seconds between keepalive pings
_INACTIVITY_WARNING_SECONDS = 60
_INACTIVITY_CLOSE_GRACE_SECONDS = 60
_EVIDENCE_JUDGE_TIMEOUT_SECONDS = 20
_PARTICIPANT_EXPERIENCE_EVENTS = {
    "connection_opened",
    "first_prompt_received",
    "first_answer_submitted",
    "response_mode_changed",
    "transcription_started",
    "transcription_succeeded",
    "transcription_failed",
    "automatic_endpoint_triggered",
    "finish_requested",
    "inactivity_resumed",
}

# OpenAI EU regional API (mirrors app.pipeline.runner)
_OPENAI_EU_BASE_URL = "https://eu.api.openai.com/v1"


async def _get_key(field: str) -> str:
    """Get effective API key: dashboard override > .env value."""
    try:
        from app.api.settings import get_effective_key
        return await get_effective_key(field)
    except Exception:
        return getattr(settings, field, "")


async def _openai_use_eu() -> bool:
    """Whether OpenAI HTTP calls should use the EU regional endpoint."""
    try:
        from app.api.settings import get_effective_flag

        return await get_effective_flag("openai_use_eu")
    except Exception:
        return bool(getattr(settings, "openai_use_eu", False))


async def _openai_api_base() -> str | None:
    """LiteLLM api_base for OpenAI chat models, or None for the default region."""
    return _OPENAI_EU_BASE_URL if await _openai_use_eu() else None


async def _redact_participant_text(value: str):
    """Run local minimisation and the configured contextual privacy layer."""
    if not settings.privacy_llm_enabled:
        return redact_text(value)
    return await redact_text_with_llm(
        value,
        api_key=await _get_key("openai_api_key"),
        model=settings.privacy_llm_model,
        api_base=await _openai_api_base(),
        timeout_seconds=settings.privacy_llm_timeout_seconds,
    )


_REALTIME_TO_CHAT: dict[str, str] = {
    # OpenAI Realtime → chat equivalents
    "openai-realtime/gpt-4o-realtime-preview": "gpt-4o",
    "openai-realtime/gpt-4o-mini-realtime-preview": "gpt-4o-mini",
    "openai-realtime/gpt-realtime-1.5": "gpt-4.1",
    "openai-realtime/gpt-realtime-2": "gpt-4.1",
    # Gemini Live → chat equivalents
    "google/gemini-2.5-flash-native-audio": "google/gemini-3.5-flash",
    "google/gemini-2.0-flash-live": "google/gemini-2.0-flash",
}


def _resolve_chat_model(model: str) -> str:
    """
    Map voice-to-voice / realtime models to their chat equivalents.

    Text chat agents can only use chat-completion models. If a user
    accidentally configures a realtime model for a text agent, we
    silently remap it so the interview still works.
    """
    # Exact match
    if model in _REALTIME_TO_CHAT:
        return _REALTIME_TO_CHAT[model]

    # Partial match on known realtime keywords
    lower = model.lower()
    if "realtime" in lower or "native-audio" in lower or "live" in lower:
        # Best-effort fallback
        if "gemini" in lower:
            return "google/gemini-3.5-flash"
        if "gpt" in lower:
            return "gpt-4.1"
        return "gpt-4.1"  # safe default

    return model


async def _maybe_inject_rag_context(
    messages: list[dict],
    study_id: uuid.UUID | None,
    user_text: str,
) -> list[dict]:
    """
    If the study has a knowledge base, search it for context relevant to
    the user's latest message and inject it as a system message so the LLM
    can ground its answer in the uploaded documents.

    Returns a (possibly augmented) copy of the messages list.
    """
    if not study_id or not user_text.strip():
        return messages

    try:
        from app.knowledge.embeddings import search_similar_chunks

        async with async_session_factory() as db:
            results = await search_similar_chunks(
                db=db,
                study_id=study_id,
                query=user_text,
                top_k=5,
            )

        if not results:
            return messages

        # Build context block
        context_parts = []
        for i, r in enumerate(results, 1):
            context_parts.append(
                f"[Source: {r['title']}] (relevance: {r['similarity']})\n"
                f"{r['content']}"
            )
        context_text = "\n\n---\n\n".join(context_parts)
        # User-role note: mid-conversation system messages are rejected by
        # some chat APIs (OpenAI gpt-5.x) with a 400 error.
        rag_message = {
            "role": "user",
            "content": (
                "[Knowledge base context — this is background material for "
                "you, not something the participant said. Use it to inform "
                "your response if appropriate, but do not mention that you "
                "are consulting a knowledge base.]\n\n"
                f"{context_text}"
            ),
        }

        # Insert RAG context just before the latest user message
        augmented = messages[:-1] + [rag_message, messages[-1]]
        logger.debug(
            f"RAG injected {len(results)} chunks for text-chat "
            f"(study={study_id})"
        )
        return augmented

    except Exception as exc:
        logger.warning(f"Text-chat RAG lookup failed (non-fatal): {type(exc).__name__}")
        return messages


async def _call_openai_responses(
    messages: list[dict],
    model: str,
    api_key: str,
    api_base: str | None,
) -> dict:
    """Call a model through OpenAI's Responses API."""
    from openai import AsyncOpenAI

    client_kwargs: dict = {"api_key": api_key}
    if api_base:
        client_kwargs["base_url"] = api_base
    client = AsyncOpenAI(**client_kwargs)

    instructions = "\n\n".join(
        str(message.get("content", ""))
        for message in messages
        if message.get("role") in ("system", "developer")
    )
    input_messages = [
        message
        for message in messages
        if message.get("role") not in ("system", "developer")
    ]
    response = await client.responses.create(
        model=model.removeprefix("openai/"),
        instructions=instructions or None,
        input=input_messages,
        store=False,
        max_output_tokens=2048,
    )
    usage = response.usage
    return {
        "content": response.output_text or "",
        "prompt_tokens": usage.input_tokens if usage else None,
        "completion_tokens": usage.output_tokens if usage else None,
    }


async def _call_llm(
    messages: list[dict],
    model: str,
    study_id: uuid.UUID | None = None,
) -> dict:
    """
    Call the LLM via LiteLLM and return the assistant message + token usage.

    If the study has a knowledge base, relevant context is injected into
    the messages before calling the LLM (inline RAG — unlike voice pipelines
    that use Pipecat's function-calling tool approach).
    """
    import litellm

    # Remap realtime/v2v models to their chat equivalents
    original_model = model
    model = _resolve_chat_model(model)
    if model != original_model:
        logger.info(f"Text chat: remapped v2v model '{original_model}' → '{model}'")

    # ── RAG context injection ──────────────────────────────────────
    # Extract latest user message for the similarity search
    last_user_text = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user_text = m.get("content", "")
            break
    messages = await _maybe_inject_rag_context(messages, study_id, last_user_text)

    api_kind = resolve_llm_api_kind(model)
    openai_model_id = model.removeprefix("openai/")
    is_direct_openai = model.startswith("openai/") or "/" not in model
    if is_direct_openai and api_kind == "responses":
        return await _call_openai_responses(
            messages,
            model,
            await _get_key("openai_api_key"),
            await _openai_api_base(),
        )

    # Resolve model to litellm format
    litellm_model = model
    if model.startswith("scaleway/"):
        litellm_model = f"openai/{model.split('/', 1)[1]}"

    # Set up API keys
    api_key = await _get_key("openai_api_key")
    kwargs: dict = {
        "model": litellm_model,
        "messages": messages,
    }
    if openai_model_id.startswith(("gpt-5", "o1", "o3", "o4")):
        kwargs["max_completion_tokens"] = 2048
    else:
        kwargs["max_tokens"] = 2048
        kwargs["temperature"] = 0.7

    if model.startswith("scaleway/"):
        scw_key = await _get_key("scaleway_secret_key")
        kwargs["api_key"] = scw_key
        kwargs["api_base"] = "https://api.scaleway.ai/v1"
    elif model.startswith("azure/"):
        kwargs["api_key"] = await _get_key("azure_openai_api_key")
        kwargs["api_base"] = getattr(settings, "azure_openai_endpoint", "")
        kwargs["api_version"] = getattr(settings, "azure_openai_api_version", "2024-02-01")
    elif model.startswith("gcp/"):
        gcp_model = model.split("/", 1)[1]
        litellm_model = f"vertex_ai/{gcp_model}"
        kwargs["model"] = litellm_model
        kwargs["api_key"] = await _get_key("gcp_api_key")
    elif model.startswith("google/"):
        google_model = model.split("/", 1)[1]
        litellm_model = f"gemini/{google_model}"
        kwargs["model"] = litellm_model
        kwargs["api_key"] = await _get_key("google_api_key")
    elif model.startswith("custom/"):
        custom_model = model.split("/", 1)[1]
        base_url = await _get_key("openai_compatible_llm_url") or settings.openai_compatible_llm_url
        if not base_url:
            raise ValueError(
                "OPENAI_COMPATIBLE_LLM_URL is not set. Provide a base URL "
                "(e.g. http://my-litellm:4000/v1) for your custom OpenAI-"
                "compatible endpoint, or pick a built-in provider."
            )
        kwargs["model"] = f"openai/{custom_model}"
        kwargs["api_base"] = base_url
        kwargs["api_key"] = await _get_key("openai_compatible_llm_api_key") or "not-needed"
    elif model.startswith("anthropic/"):
        anthropic_key = await _get_key("anthropic_api_key")
        if not anthropic_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. Add it to your .env file or dashboard."
            )
        kwargs["api_key"] = anthropic_key
        kwargs["model"] = model
    else:
        # OpenAI chat models (openai/ prefix, bare gpt-* ids, etc.)
        kwargs["api_key"] = api_key
        eu_base = await _openai_api_base()
        if eu_base:
            kwargs["api_base"] = eu_base

    response = await litellm.acompletion(**kwargs)

    content = response.choices[0].message.content or ""
    usage = response.usage
    return {
        "content": content,
        "prompt_tokens": usage.prompt_tokens if usage else None,
        "completion_tokens": usage.completion_tokens if usage else None,
    }


async def _judge_structured_turn(
    controller,
    *,
    participant_text: str,
    participant_sequence: int,
) -> dict[str, object]:
    """Run one independent batched judge and stage its authoritative state.

    A judge failure still stages an empty authoritative result. The moderator's
    own coverage claims are therefore ignored instead of becoming an unsafe
    fallback that could skip an unanswered card.
    """
    from app.pipeline.conversation_policy import ContextualInterviewController, observe_conversation
    if isinstance(controller, ContextualInterviewController):
        try:
            observation, usage = await asyncio.wait_for(observe_conversation(
                api_key=await _get_key("openai_api_key"), api_base=await _openai_api_base(),
                model=settings.evidence_judge_model, payload=controller.observer_context(),
            ), timeout=_EVIDENCE_JUDGE_TIMEOUT_SECONDS)
            controller.apply_observation(observation)
            return {"authoritative": True, "control_note": controller.authoritative_model_note(),
                    **usage, "candidate_count": controller.total_questions,
                    "returned_count": len(observation.updates), "failed": False}
        except Exception as exc:
            logger.warning("Conversation observer unavailable: {}", type(exc).__name__)
            controller.observation_failed()
            return {"authoritative": True, "control_note": controller.authoritative_model_note(),
                    "prompt_tokens": 0, "completion_tokens": 0,
                    "candidate_count": controller.total_questions, "returned_count": 0, "failed": True}
    model = str(settings.evidence_judge_model or "gpt-5.6-terra").strip()
    empty = {
        "resolutions": {},
        "evidence_sources": {},
        "missing_facets": {},
        "confidences": {},
    }
    try:
        api_key = await _get_key("openai_api_key")
        if not api_key:
            raise ValueError("OpenAI API key is unavailable")
        from app.pipeline.evidence_judge import judge_evidence

        result = await asyncio.wait_for(
            judge_evidence(
                api_key=api_key,
                api_base=await _openai_api_base(),
                model=model,
                latest_response=participant_text,
                context=controller.evidence_judge_context(participant_text),
            ),
            timeout=_EVIDENCE_JUDGE_TIMEOUT_SECONDS,
        )
        controller.prepare_external_judgment(
            participant_text,
            participant_sequence,
            resolutions=result.resolutions,
            evidence_sources=result.evidence_sources,
            missing_facets=result.missing_facets,
            confidences=result.confidences,
            judge_model=model,
        )
        return {
            "authoritative": True,
            "control_note": controller.authoritative_model_note(),
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "candidate_count": result.candidate_count,
            "returned_count": result.returned_count,
            "failed": False,
        }
    except Exception as exc:
        logger.warning(
            "Evidence judge failed; accepting no new evidence: {}",
            type(exc).__name__,
        )
        controller.prepare_external_judgment(
            participant_text,
            participant_sequence,
            **empty,
            judge_model=model,
            failed=True,
        )
        return {
            "authoritative": True,
            "control_note": controller.authoritative_model_note(),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "candidate_count": len(
                controller.evidence_judge_context(participant_text).get("candidates") or []
            ),
            "returned_count": 0,
            "failed": True,
        }


async def _contextual_agent_turn(controller, messages, *, model, study_id, participant_text):
    """Generate, semantically check and repair once before delivering a turn."""
    from app.pipeline.conversation_policy import ConversationRepair, check_question
    from app.pipeline.interview_guide import strip_control_markers, control_marker_prefix

    # Replace transient server state rather than accumulating stale decisions.
    context = [m for m in messages if not str(m.get("content", "")).startswith("[CONVERSATION STATE")]
    context = context + [{"role": "system", "content": controller.authoritative_model_note()}]
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    last_issue = "generation_failed"
    review = None
    rejected_anchors = []
    repair_decision = None
    for attempt in range(2):
        raw = ""
        try:
            result = await _call_llm(context, model=model, study_id=study_id)
            for key in usage:
                usage[key] += result.get(key, 0)
            raw = result["content"]
            text, marker, decision = strip_control_markers(raw)
            text = text.strip()
            # Repair may omit its routing tag. Recover only the previously
            # selected move; validate eligibility and semantics again below.
            if decision is None and repair_decision is not None:
                decision = repair_decision
                controller.events.append({"action": "repair_marker_recovered", "intent": decision.intent})
            if decision is not None:
                repair_decision = decision
            move = controller.validate_turn(text, decision, participant_text)
            if decision.action not in {"close", "clarify"}:
                check, check_usage = await asyncio.wait_for(check_question(
                    api_key=await _get_key("openai_api_key"), api_base=await _openai_api_base(),
                    model=settings.evidence_judge_model,
                    payload=controller.question_check_context(text, move),
                ), timeout=_EVIDENCE_JUDGE_TIMEOUT_SECONDS)
                for key in usage:
                    usage[key] += check_usage[key]
                approved_scope = (decision.action == "main" and check.issue == "wrong_purpose"
                                  and " ".join(text.split()) == " ".join(move["wording"].split()))
                if approved_scope:
                    controller.events.append({"action": "critic_scope_override",
                                              "reason": "approved_anchor_can_cover_part_of_topic",
                                              "proposed_text": text})
                elif not check.acceptable or check.issue != "none":
                    controller.record_rejection(text, check.issue, check.correction)
                    if move["action"] == "main" and check.issue in {"already_answered", "unsupported_premise"}:
                        rejected_anchors.append((move, check.issue))
                    if check.purpose_exhausted or check.issue == "participant_limit":
                        controller.retire_rejected_opportunity(move, check.issue)
                    controller.defer_rejected_anchor(move, check.issue)
                    raise ConversationRepair(check.correction or check.issue)
            review = controller.accept_turn(text, decision, move)
            break
        except Exception as exc:
            last_issue = str(exc) if isinstance(exc, ConversationRepair) else type(exc).__name__
            if not isinstance(exc, ConversationRepair):
                logger.warning("Contextual turn unavailable: {}", type(exc).__name__)
            controller.events.append({"action": "repair_attempt", "attempt": attempt + 1,
                                      "reason": last_issue, "proposed_text": raw})
            if raw:
                context.append({"role": "assistant", "content": raw})
            context.append({"role": "system", "content": "The proposed turn was NOT delivered. "
                            "Repair it once using an eligible move. " + last_issue})
            context.append({"role": "system", "content": controller.authoritative_model_note()})
    if review is None:
        for rejected_anchor, issue in rejected_anchors:
            controller.defer_rejected_anchor(rejected_anchor, issue, final=True)
        review = controller.fallback_turn(last_issue)
    return {"content": control_marker_prefix(review.marker, review.decision) + " " + review.text,
            **usage}, review


async def _keepalive_loop(ws: WebSocket, stop: asyncio.Event) -> None:
    """Send periodic pings to prevent idle-timeout disconnections (e.g. iOS Safari)."""
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=_KEEPALIVE_INTERVAL)
        except asyncio.TimeoutError:
            try:
                await ws.send_json({"type": "ping"})
            except Exception:
                break


async def _log_turn(
    session_id: uuid.UUID,
    role: SpeakerRole,
    content: str,
    sequence: int,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
):
    """Log a transcript entry to the database."""
    async with async_session_factory() as db:
        entry = TranscriptEntry(
            id=uuid.uuid4(),
            session_id=session_id,
            role=role,
            content=content,
            sequence=sequence,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            spoken_at=datetime.now(timezone.utc),
        )
        db.add(entry)
        await db.commit()


async def _send_scripted_agent_turn(
    websocket: WebSocket,
    session_id: uuid.UUID,
    text: str,
    sequence: int,
) -> int:
    """Persist and send one deterministic moderator turn."""
    next_sequence = sequence + 1
    await _log_turn(session_id, SpeakerRole.AGENT, text, next_sequence)
    await publish_transcript_event(str(session_id), {
        "type": "transcript",
        "role": "agent",
        "content": text,
        "sequence": next_sequence,
    })
    await websocket.send_json({
        "type": "message",
        "text": text,
        "role": "agent",
    })
    return next_sequence


async def _send_privacy_warning(
    websocket: WebSocket,
) -> None:
    """Give a non-echoing warning without changing research-turn sequencing."""
    await websocket.send_json(
        {"type": "privacy_warning", "text": PRIVACY_WARNING, "role": "agent"}
    )


async def _record_engagement_turn(
    session_id: uuid.UUID,
    scorer,
    detector,
    *,
    sequence: int,
    text: str,
    language: str | None,
    response_latency_ms: int | None,
) -> None:
    """
    Compute and persist engagement metrics for one text-chat user turn.

    Text interviews expose only lexical and timing signals (no audio), so
    voiced duration, speech rate, and energy stay null. Never raises — a
    failure here must not break the interview.
    """
    from app.engagement.features import TurnFeatures
    from app.models.engagement import EngagementEvent, EngagementTurn

    try:
        features = TurnFeatures.from_turn(
            transcript_sequence=sequence,
            text=text or "",
            language=language,
            response_latency_ms=response_latency_ms,
            voiced_ms=None,
            modality="text",
        )
        result = scorer.score(features)
        events = detector.observe(result.label)
        event_types = [e.event_type for e in events]

        async with async_session_factory() as db:
            db.add(
                EngagementTurn(
                    id=uuid.uuid4(),
                    session_id=session_id,
                    transcript_sequence=sequence,
                    response_latency_ms=features.response_latency_ms,
                    voiced_ms=None,
                    word_count=features.word_count,
                    char_count=features.char_count,
                    speech_rate_wpm=None,
                    filler_count=features.filler_count,
                    rms_energy=None,
                    score=result.score,
                    label=result.label,
                    extras={
                        "flags": result.flags,
                        "components": result.components,
                        "modality": "text",
                    },
                )
            )
            for ev in events:
                db.add(
                    EngagementEvent(
                        id=uuid.uuid4(),
                        session_id=session_id,
                        transcript_sequence=sequence,
                        event_type=ev.event_type,
                        score_at_event=result.score,
                        payload=ev.payload,
                    )
                )
            await db.commit()
        return {"triggers": set(event_types) | set(result.flags)}
    except Exception as exc:
        logger.error(f"Text engagement: failed to record turn {sequence}: {type(exc).__name__}")
        return None


async def _apply_text_adaptation(
    session_id: uuid.UUID,
    engine,
    *,
    sequence: int,
    triggers: set[str],
    messages: list[dict],
) -> None:
    """
    Evaluate the adaptive policy for a text turn and, in live mode, inject a
    system instruction before the next LLM call. Speed actions do not apply to
    text. Every action is recorded for disclosure. Never raises.
    """
    import time

    from app.engagement.adaptive import PROMPT, guidance_message
    from app.models.engagement import AdaptiveAction

    try:
        actions = engine.evaluate(triggers, time.monotonic())
        if not actions:
            return
        live = engine.policy.is_live
        async with async_session_factory() as db:
            for act in actions:
                applied = False
                if act.type == PROMPT and act.instruction:
                    if live:
                        # User-role note: mid-conversation system messages are
                        # rejected by some chat APIs (OpenAI gpt-5.x, 400).
                        messages.append(guidance_message(act.instruction))
                        applied = True
                    detail = {"applied": applied, "instruction": act.instruction}
                else:
                    # tts_speed has no effect in text chat.
                    detail = {
                        "applied": False,
                        "params": act.params,
                        "note": "speed_not_applicable_to_text",
                    }
                db.add(
                    AdaptiveAction(
                        id=uuid.uuid4(),
                        session_id=session_id,
                        transcript_sequence=sequence,
                        trigger=act.trigger,
                        action=act.action,
                        mode=engine.policy.mode,
                        detail=detail,
                    )
                )
            await db.commit()
    except Exception as exc:
        logger.error(f"Text adaptation: failed at turn {sequence}: {type(exc).__name__}")


@router.websocket("/ws/chat/{widget_key}")
async def text_chat_ws(
    websocket: WebSocket,
    widget_key: str,
    pid: str | None = Query(None),
    external_study_id: str | None = Query(None),
    external_session_id: str | None = Query(None),
    consent: bool = Query(False),
    response_mode: str | None = Query(None),
    mic_check_status: str | None = Query(None),
):
    """
    Text-based chat interview WebSocket endpoint.

    Protocol (JSON over WebSocket):
      Client → Server: { "type": "message", "text": "..." }
      Server → Client: { "type": "message", "text": "...", "role": "agent" }
      Server → Client: { "type": "welcome", "text": "...", "avatar": "..." }
      Server → Client: { "type": "ended" }
      Server → Client: { "type": "error", "text": "..." }
    """
    await websocket.accept()

    # ── 1. Resolve agent ──────────────────────────────────────────
    async with async_session_factory() as db:
        result = await db.execute(
            select(Agent).where(
                Agent.widget_key == widget_key,
                Agent.status == AgentStatus.ACTIVE.value,
            )
        )
        agent = result.scalar_one_or_none()

        if not agent:
            await websocket.send_json({"type": "error", "text": "Agent not found or inactive"})
            await websocket.close(code=4004, reason="Agent not found or inactive")
            return

        study = await db.get(Study, agent.study_id)
        research_spec = study.research_spec if study else None
        if consent_required(research_spec) and not consent:
            await websocket.send_json({
                "type": "error",
                "text": "Please review and accept the study information before starting.",
            })
            await websocket.close(code=4003, reason="Consent required")
            return

        # The participant experience is text-first even when the researcher
        # configured a modular voice pipeline. In that case we use the same
        # guide and LLM over chat; optional question audio is synthesized by a
        # separate, participant-controlled endpoint. Voice-to-voice model names
        # are already mapped to chat equivalents by ``_resolve_chat_model``.
        validation_errors = await validate_agent_pipeline_config(
            modality="text",
            pipeline_type="modular",
            llm_model=agent.llm_model,
        )
        if validation_errors:
            await websocket.send_json(
                {"type": "error", "text": "; ".join(validation_errors)}
            )
            await websocket.close(code=4006, reason="Invalid agent configuration")
            return

        # Snapshot config
        agent_cfg = {
            "id": agent.id,
            "study_id": agent.study_id,
            "system_prompt": agent.system_prompt,
            "welcome_message": agent.welcome_message,
            "llm_model": agent.llm_model,
            "language": agent.language,
            "max_duration_seconds": agent.max_duration_seconds,
            "participant_id_mode": agent.participant_id_mode,
            "avatar": agent.avatar or "neutral",
            "interview_mode": (
                agent.interview_mode.value
                if hasattr(agent.interview_mode, "value")
                else (agent.interview_mode or "free_form")
            ),
            "interview_guide": agent.interview_guide,
            "track_engagement": bool(agent.track_engagement),
            "engagement_config": agent.engagement_config,
            "adaptive_enabled": bool(agent.adaptive_enabled),
            "adaptive_policy": agent.adaptive_policy,
            "widget_show_progress": bool(agent.widget_show_progress),
            "research_spec": research_spec,
        }

        # ── 2. Resolve participant_id ─────────────────────────────
        participant_id: str | None = None
        participant_context: dict = {}
        fielding = (research_spec or {}).get("fielding") or {}
        recruitment_source = str(fielding.get("recruitment_source") or "") or None

        if agent_cfg["participant_id_mode"] == ParticipantIdMode.RANDOM:
            participant_id = secrets.token_urlsafe(8)
        elif agent_cfg["participant_id_mode"] == ParticipantIdMode.PREDEFINED:
            if not pid:
                await websocket.send_json({"type": "error", "text": "This interview requires a valid participant link."})
                await websocket.close(code=4003, reason="Missing participant ID")
                return
            pid_result = await db.execute(
                select(ParticipantIdentifier).where(
                    ParticipantIdentifier.agent_id == agent_cfg["id"],
                    ParticipantIdentifier.identifier == pid,
                )
            )
            pid_record = pid_result.scalar_one_or_none()
            if not pid_record:
                await websocket.send_json({"type": "error", "text": "Invalid participant identifier."})
                await websocket.close(code=4003, reason="Invalid participant ID")
                return
            if pid_record.used:
                await websocket.send_json({"type": "error", "text": "This participant link has already been used."})
                await websocket.close(code=4003, reason="Participant ID already used")
                return
            participant_id = pid
            participant_context = pid_record.profile or {}
        elif agent_cfg["participant_id_mode"] == ParticipantIdMode.INPUT:
            # Reject empty/blank IDs — INPUT mode requires the participant
            # to actually type something, otherwise sessions become anonymous.
            if not pid or not pid.strip():
                await websocket.send_json({
                    "type": "error",
                    "text": "Please provide a participant ID before starting the chat.",
                })
                await websocket.close(code=4003, reason="Missing participant ID")
                return
            participant_id = pid.strip()
        elif agent_cfg["participant_id_mode"] == ParticipantIdMode.EXTERNAL:
            if not pid or not pid.strip():
                await websocket.send_json({
                    "type": "error",
                    "text": "This interview requires the participant link supplied by the recruitment platform.",
                })
                await websocket.close(code=4003, reason="Missing external participant ID")
                return
            if recruitment_source == "prolific" and not external_session_id:
                await websocket.send_json({
                    "type": "error",
                    "text": "This Prolific link is incomplete. Please reopen the study from Prolific.",
                })
                await websocket.close(code=4003, reason="Missing Prolific session ID")
                return
            participant_id = pid.strip()

        # Adaptation can affect this chat only when engagement is on, the
        # policy has rules, and the mode is live.
        _policy = agent_cfg.get("adaptive_policy") or {}
        adaptive_active = bool(
            agent_cfg["track_engagement"]
            and agent_cfg["adaptive_enabled"]
            and _policy.get("rules")
            and _policy.get("mode") == "live"
        )

        # ── 3. Create session ─────────────────────────────────────
        session = Session(
            id=uuid.uuid4(),
            agent_id=agent_cfg["id"],
            status=SessionStatus.ACTIVE,
            participant_id=participant_id,
            recruitment_source=recruitment_source,
            external_study_id=(external_study_id or "").strip() or None,
            external_session_id=(external_session_id or "").strip() or None,
            adaptive_active=adaptive_active,
            research_protocol_version=(research_spec or {}).get("version"),
            research_stage=(research_spec or {}).get("stage"),
            protocol_snapshot=build_protocol_snapshot(
                agent_cfg, research_spec, participant_context
            ),
            participant_consent_confirmed=bool(consent),
            participant_consent_at=datetime.now(timezone.utc) if consent else None,
        )
        db.add(session)
        try:
            await db.commit()
        except IntegrityError:
            await db.rollback()
            await websocket.send_json({
                "type": "error",
                "text": "This recruitment-platform submission has already started this interview.",
            })
            await websocket.close(code=4009, reason="Duplicate external submission")
            return
        session_id = session.id
        start_time = datetime.now(timezone.utc)

        # Mark predefined ID as used
        if agent_cfg["participant_id_mode"] == ParticipantIdMode.PREDEFINED and pid:
            pid_result2 = await db.execute(
                select(ParticipantIdentifier).where(
                    ParticipantIdentifier.agent_id == agent_cfg["id"],
                    ParticipantIdentifier.identifier == pid,
                )
            )
            pid_record2 = pid_result2.scalar_one_or_none()
            if pid_record2:
                pid_record2.used = True
                pid_record2.session_id = session_id
                await db.commit()

    logger.info(
        f"Text chat started: session={session_id}, agent={agent_cfg['id']}, "
        "participant metadata omitted"
    )

    timing = build_interview_timing(
        agent_cfg["max_duration_seconds"], fallback_seconds=_MAX_CHAT_DURATION
    )

    await register_session(
        session_id=session_id,
        agent_id=agent_cfg["id"],
        max_duration_seconds=timing.hard_stop_seconds,
    )

    # ── 4. Build system prompt ────────────────────────────────────
    system_prompt = build_research_context(
        agent_cfg["system_prompt"],
        agent_cfg["research_spec"],
        participant_context,
    )
    is_structured = bool(
        agent_cfg["interview_mode"] == "structured"
        and agent_cfg["interview_guide"]
    )
    # Progress bar: only meaningful for structured chats with the toggle on.
    show_progress = bool(is_structured and agent_cfg["widget_show_progress"])
    progress_total = (
        len(agent_cfg["interview_guide"].get("questions", []))
        if is_structured
        else 0
    )
    reported_progress = 0
    structured_controller = None
    if is_structured:
        from app.pipeline.interview_guide import (
            build_structured_prompt,
            resolve_role_guide,
            TextStructuredController,
        )
        runtime_guide = resolve_role_guide(
            agent_cfg["interview_guide"], participant_context
        )
        progress_total = len(runtime_guide.get("questions", []))
        # Emit the hidden [[Qn]] marker even when the progress bar is off: the
        # controller uses it to track which question we're on. It's stripped
        # before the participant and transcript see anything.
        if runtime_guide.get("conversation_policy") == "contextual_v1":
            from app.pipeline.conversation_policy import ContextualInterviewController, build_contextual_prompt
            system_prompt = build_contextual_prompt(system_prompt, runtime_guide)
            structured_controller = ContextualInterviewController(runtime_guide, language=agent_cfg["language"] or "en")
        else:
            system_prompt = build_structured_prompt(system_prompt, runtime_guide, emit_progress=True)
            structured_controller = TextStructuredController(runtime_guide, language=agent_cfg["language"] or "en")

    # Add text-first interface instructions. Optional audio playback uses this
    # exact response text, so spoken and written prompts always stay aligned.
    system_prompt += (
        "\n\n---\n"
        "You are communicating through a text-first interview interface. "
        "Your exact response may also be read aloud if the participant turns "
        "audio on. Keep every turn concise, conversational, and easy to "
        "understand both on screen and when heard once."
    )

    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    sequence = 0
    evidence_judge_prompt_tokens = 0
    evidence_judge_completion_tokens = 0
    evidence_judge_candidates = 0
    evidence_judge_returned = 0
    privacy_redaction_counts: Counter[str] = Counter()

    # ── Engagement metrics (text profile; observational, lexical + timing) ──
    engagement_scorer = None
    engagement_detector = None
    adaptive_engine = None
    agent_sent_at: float | None = None
    if agent_cfg["track_engagement"]:
        from app.engagement.events import EventDetector
        from app.engagement.scorer import RuleBasedScorer, ScorerConfig

        _eng_cfg = ScorerConfig.from_dict(
            agent_cfg["engagement_config"], modality="text"
        )
        engagement_scorer = RuleBasedScorer(_eng_cfg)
        engagement_detector = EventDetector(_eng_cfg)
        logger.info(f"Engagement tracking enabled for text session {session_id}")

        if agent_cfg["adaptive_enabled"]:
            from app.engagement.adaptive import AdaptivePolicy, AdaptivePolicyEngine

            _policy_obj = AdaptivePolicy.from_dict(agent_cfg["adaptive_policy"])
            if _policy_obj.rules:
                adaptive_engine = AdaptivePolicyEngine(_policy_obj)
                logger.info(
                    f"Adaptive behavior ({_policy_obj.mode}) enabled for text "
                    f"session {session_id}"
                )

    # ── 5. Send welcome message ───────────────────────────────────
    welcome = agent_cfg["welcome_message"]
    if welcome:
        messages.append({"role": "assistant", "content": welcome})
        await websocket.send_json({
            "type": "welcome",
            "text": welcome,
            "avatar": agent_cfg["avatar"],
        })
        sequence += 1
        await _log_turn(session_id, SpeakerRole.AGENT, welcome, sequence)
        await publish_transcript_event(str(session_id), {
            "type": "transcript",
            "role": "agent",
            "content": welcome,
            "sequence": sequence,
        })
        if engagement_scorer:
            agent_sent_at = time.monotonic()

    # ── 6. Chat loop ──────────────────────────────────────────────
    keepalive_stop = asyncio.Event()
    keepalive_task = asyncio.create_task(_keepalive_loop(websocket, keepalive_stop))
    final_status = SessionStatus.COMPLETED
    final_reason = "interview_complete"
    participant_experience_events: list[dict[str, object]] = []
    awaiting_interruption_confirmation = False
    prioritization_injected = False
    final_question_asked = False
    try:
        while True:
            # The configured duration is a target. Only the grace-period
            # ceiling is a hard technical stop.
            elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
            if elapsed >= timing.hard_stop_seconds:
                await websocket.send_json({
                    "type": "ended",
                    "reason": "hard_time_limit",
                    "text": (
                        "We've reached the end of the interview window. Your "
                        "responses have been saved. Thank you for taking part."
                    ),
                })
                final_status = SessionStatus.COMPLETED
                final_reason = "hard_time_limit"
                break

            # Wait for user message
            remaining_total = max(timing.hard_stop_seconds - elapsed, 1)
            try:
                raw = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=min(remaining_total, _INACTIVITY_WARNING_SECONDS),
                )
            except asyncio.TimeoutError:
                # Silence should not leave a participant microphone and
                # session open until the full interview ceiling. Warn first,
                # pause browser capture, and allow a generous recovery window.
                elapsed = (
                    datetime.now(timezone.utc) - start_time
                ).total_seconds()
                remaining_total = timing.hard_stop_seconds - elapsed
                if remaining_total <= 0:
                    await websocket.send_json({
                        "type": "ended",
                        "reason": "hard_time_limit",
                        "text": (
                            "We've reached the end of the interview window. "
                            "Your responses have been saved. Thank you for "
                            "taking part."
                        ),
                    })
                    final_status = SessionStatus.COMPLETED
                    final_reason = "hard_time_limit"
                    break
                grace_seconds = max(
                    1,
                    min(
                        _INACTIVITY_CLOSE_GRACE_SECONDS,
                        int(remaining_total),
                    ),
                )
                await websocket.send_json({
                    "type": "inactivity_warning",
                    "seconds_remaining": grace_seconds,
                    "text": (
                        "We paused listening because we haven't heard from "
                        "you. Let us know you're still here to continue."
                    ),
                })
                try:
                    raw = await asyncio.wait_for(
                        websocket.receive_text(),
                        timeout=grace_seconds,
                    )
                except asyncio.TimeoutError:
                    await websocket.send_json({
                        "type": "ended",
                        "reason": "participant_inactive",
                        "text": (
                            "The interview was closed after a period of "
                            "inactivity. Your responses so far have been saved."
                        ),
                    })
                    final_status = SessionStatus.TIMED_OUT
                    final_reason = "participant_inactive"
                    break

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            # Browser-local voice capture and typing can take a while before a
            # complete answer is submitted. Throttled activity signals keep a
            # genuinely active participant from being timed out mid-answer.
            if msg.get("type") == "keepalive":
                continue

            if msg.get("type") == "experience_event":
                event_name = str(msg.get("event") or "").strip()
                if (
                    event_name in _PARTICIPANT_EXPERIENCE_EVENTS
                    and len(participant_experience_events) < 50
                ):
                    raw_payload = msg.get("payload")
                    payload = {}
                    if isinstance(raw_payload, dict):
                        for key, value in list(raw_payload.items())[:8]:
                            safe_key = str(key)[:40]
                            if isinstance(value, (bool, int, float)) or value is None:
                                payload[safe_key] = value
                            elif isinstance(value, str):
                                payload[safe_key] = value[:120]
                    participant_experience_events.append({
                        "event": event_name,
                        "elapsed_seconds": round(
                            (datetime.now(timezone.utc) - start_time).total_seconds(),
                            2,
                        ),
                        "payload": payload,
                    })
                continue

            # Ending the interview is a distinct control action, not an empty
            # answer or a WebSocket accident. A final in-progress voice answer
            # may arrive in the same message after browser transcription.
            if msg.get("type") == "finish":
                final_text = str(msg.get("text") or "").strip()
                if final_text:
                    redaction = await _redact_participant_text(final_text)
                    final_text = redaction.text
                    privacy_redaction_counts.update(redaction.counts)
                    sequence += 1
                    await _log_turn(
                        session_id, SpeakerRole.USER, final_text, sequence
                    )
                    await publish_transcript_event(str(session_id), {
                        "type": "transcript",
                        "role": "user",
                        "content": final_text,
                        "sequence": sequence,
                    })
                    if redaction.detected:
                        await _send_privacy_warning(websocket)
                await websocket.send_json({
                    "type": "ended",
                    "reason": "participant_finished",
                    "text": (
                        "Your responses have been saved. Thank you for your "
                        "participation."
                    ),
                })
                final_status = SessionStatus.COMPLETED
                final_reason = "participant_finished"
                break

            if msg.get("type") != "message" or not msg.get("text", "").strip():
                continue

            redaction = await _redact_participant_text(msg["text"].strip())
            user_text = redaction.text
            privacy_redaction_counts.update(redaction.counts)

            # Log user turn
            sequence += 1
            await _log_turn(session_id, SpeakerRole.USER, user_text, sequence)
            await publish_transcript_event(str(session_id), {
                "type": "transcript",
                "role": "user",
                "content": user_text,
                "sequence": sequence,
            })

            if redaction.detected:
                await _send_privacy_warning(websocket)

            if user_text in {
                "[SENSITIVE_CONTENT_REMOVED]",
                "[POTENTIALLY_IDENTIFYING_CONTEXT_REMOVED]",
                "[PRIVACY_CHECK_UNAVAILABLE]",
            }:
                # A withheld answer provides no research evidence. Keep the
                # current question rather than praising it or advancing the guide.
                redirect = (
                    "Let's stay with the same question. Could you answer in general terms, "
                    "without details about a specific person? You can also skip this question."
                )
                messages.append({"role": "user", "content": user_text})
                messages.append({"role": "assistant", "content": redirect})
                sequence = await _send_scripted_agent_turn(
                    websocket, session_id, redirect, sequence
                )
                agent_sent_at = time.monotonic()
                continue

            if engagement_scorer:
                latency_ms = (
                    int((time.monotonic() - agent_sent_at) * 1000)
                    if agent_sent_at is not None
                    else None
                )
                eng = await _record_engagement_turn(
                    session_id,
                    engagement_scorer,
                    engagement_detector,
                    sequence=sequence,
                    text=user_text,
                    language=agent_cfg["language"],
                    response_latency_ms=latency_ms,
                )
                if adaptive_engine and eng and eng.get("triggers"):
                    await _apply_text_adaptation(
                        session_id,
                        adaptive_engine,
                        sequence=sequence,
                        triggers=eng["triggers"],
                        messages=messages,
                    )

            messages.append({"role": "user", "content": user_text})

            from app.pipeline.conversation_policy import ContextualInterviewController
            if isinstance(structured_controller, ContextualInterviewController):
                structured_controller.remaining_seconds = max(0, timing.target_seconds - (
                    datetime.now(timezone.utc) - start_time).total_seconds())

            # A clearly stated desire to end the session is equivalent to the
            # Finish interview control. This matters in voice-default mode,
            # where participants may naturally say it instead of tapping.
            if structured_controller is not None:
                from app.pipeline.interview_guide import (
                    classify_participant_turn,
                )

                if classify_participant_turn(
                    user_text, agent_cfg["language"] or "en", infer_incomplete=False
                ) == "stop":
                    await websocket.send_json({
                        "type": "ended",
                        "reason": "participant_finished",
                        "text": (
                            "Your responses have been saved. Thank you for "
                            "your participation."
                        ),
                    })
                    final_status = SessionStatus.COMPLETED
                    final_reason = "participant_finished"
                    break

            # High-risk interruptions are answered without a model call. This
            # makes privacy/data-use replies faithful to the public protocol
            # and prevents the active question from being lost to a model-
            # authored progress marker.
            if structured_controller is not None:
                from app.pipeline.interview_guide import (
                    classify_participant_turn,
                    looks_like_more_procedural_help,
                    looks_like_procedural_resume,
                    participant_interruption_response,
                    procedural_resume_response,
                )

                # A study/privacy or scope answer creates a real pause.
                # Do not treat a short "yes" as research evidence and do not
                # re-ask the suspended guide question until the participant
                # has explicitly said they are ready.
                if awaiting_interruption_confirmation:
                    if looks_like_procedural_resume(user_text):
                        interruption_reply = procedural_resume_response(
                            structured_controller.current_prompt_text
                        )
                        awaiting_interruption_confirmation = False
                        messages.append({
                            "role": "assistant",
                            "content": interruption_reply,
                        })
                        sequence = await _send_scripted_agent_turn(
                            websocket,
                            session_id,
                            interruption_reply,
                            sequence,
                        )
                        if engagement_scorer:
                            agent_sent_at = time.monotonic()
                        continue
                    if looks_like_more_procedural_help(user_text):
                        interruption_reply = (
                            "Of course. What else would you like to know "
                            "before we continue?"
                        )
                        messages.append({
                            "role": "assistant",
                            "content": interruption_reply,
                        })
                        sequence = await _send_scripted_agent_turn(
                            websocket,
                            session_id,
                            interruption_reply,
                            sequence,
                        )
                        if engagement_scorer:
                            agent_sent_at = time.monotonic()
                        continue
                    if classify_participant_turn(
                        user_text, agent_cfg["language"] or "en", infer_incomplete=False
                    ) == "answer":
                        # Participants sometimes answer the suspended research
                        # question directly instead of saying "continue".
                        # Respect that choice and resume normal orchestration.
                        awaiting_interruption_confirmation = False

                interruption_reply = participant_interruption_response(
                    user_text,
                    participant_information(agent_cfg["research_spec"]),
                    structured_controller.current_prompt_text,
                    agent_cfg["language"] or "en",
                    infer_incomplete=False,
                )
                if interruption_reply:
                    if classify_participant_turn(
                        user_text, agent_cfg["language"] or "en", infer_incomplete=False
                    ) == "procedural":
                        awaiting_interruption_confirmation = True
                    messages.append({
                        "role": "assistant",
                        "content": interruption_reply,
                    })
                    sequence = await _send_scripted_agent_turn(
                        websocket,
                        session_id,
                        interruption_reply,
                        sequence,
                    )
                    if engagement_scorer:
                        agent_sent_at = time.monotonic()
                    continue

            if (isinstance(structured_controller, ContextualInterviewController)
                    and classify_participant_turn(user_text, agent_cfg["language"] or "en", infer_incomplete=False) == "answer"):
                if structured_controller._last_text:
                    structured_controller.note_answer(user_text, sequence)
                else:
                    structured_controller.note_opening_answer(user_text, sequence, welcome)

            # Evidence coverage is judged independently from the moderator.
            # The call is batched across every unresolved/partial criterion.
            # Procedural turns and the fixed, non-adaptive profile do not need
            # semantic grading.
            authoritative_evidence_for_turn = False
            if (
                structured_controller is not None
                and (settings.evidence_judge_enabled or isinstance(structured_controller, ContextualInterviewController))
                and structured_controller.needs_evidence_judge
                and classify_participant_turn(
                    user_text, agent_cfg["language"] or "en", infer_incomplete=False
                )
                == "answer"
            ):
                judge_result = await _judge_structured_turn(
                    structured_controller,
                    participant_text=user_text,
                    participant_sequence=sequence,
                )
                authoritative_evidence_for_turn = bool(
                    judge_result["authoritative"]
                )
                evidence_judge_prompt_tokens += int(
                    judge_result["prompt_tokens"] or 0
                )
                evidence_judge_completion_tokens += int(
                    judge_result["completion_tokens"] or 0
                )
                evidence_judge_candidates += int(
                    judge_result["candidate_count"] or 0
                )
                evidence_judge_returned += int(
                    judge_result["returned_count"] or 0
                )
                # Place server control immediately before the participant
                # answer. The participant remains the latest user turn for
                # RAG lookup and normal conversational recency.
                messages.insert(
                    max(0, len(messages) - 1),
                    {
                        "role": "user",
                        "content": str(judge_result["control_note"]),
                    },
                )

            # Nudge the model to advance/close once a question's budget is up.
            if structured_controller is not None:
                nudge = structured_controller.maybe_advance_message(user_text)
                if nudge is not None:
                    messages.append(nudge)

            elapsed_after_answer = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds()
            participant_kind_for_timing = (
                classify_participant_turn(
                    user_text, agent_cfg["language"] or "en", infer_incomplete=False
                )
                if structured_controller is not None
                else "answer"
            )
            timing_turn_is_answer = participant_kind_for_timing not in {
                "procedural",
                "flow_repair",
                "clarification",
                "continuation",
            }

            # Once the signposted final question has been answered, close
            # without launching another topic. An answer that runs past the
            # target is still saved in full before this close.
            if timing_turn_is_answer and (
                final_question_asked
                or elapsed_after_answer >= timing.target_seconds
            ):
                closing = (
                    structured_controller.close_for_time()
                    if structured_controller is not None
                    else "Thank you. That covers the questions we had for today."
                )
                messages.append({"role": "assistant", "content": closing})
                sequence = await _send_scripted_agent_turn(
                    websocket, session_id, closing, sequence
                )
                await websocket.send_json({
                    "type": "ended",
                    "reason": "target_duration_reached",
                    "text": "Thank you for your participation.",
                })
                final_status = SessionStatus.COMPLETED
                final_reason = "target_duration_reached"
                break

            # With roughly one minute left, ask one clearly labelled final
            # question. Structured interviews use a deterministic guide card;
            # free-form interviews receive an explicit orchestration cue.
            if (
                not final_question_asked
                and elapsed_after_answer >= timing.final_question_at_seconds
            ):
                if structured_controller is not None:
                    selected = structured_controller.begin_time_limited_final_question()
                    if selected is None:
                        closing = structured_controller.close_for_time()
                        sequence = await _send_scripted_agent_turn(
                            websocket, session_id, closing, sequence
                        )
                        await websocket.send_json({
                            "type": "ended",
                            "reason": "guide_complete_near_target",
                            "text": "Thank you for your participation.",
                        })
                        final_reason = "guide_complete_near_target"
                        break
                    final_text, final_number = selected
                    final_text = (
                        "We're nearly at time, so I have one final question. "
                        f"{final_text}"
                    )
                    messages.append({"role": "assistant", "content": final_text})
                    sequence = await _send_scripted_agent_turn(
                        websocket, session_id, final_text, sequence
                    )
                    final_question_asked = True
                    if show_progress and final_number > reported_progress:
                        reported_progress = min(final_number, progress_total)
                        await websocket.send_json({
                            "type": "progress",
                            "current": reported_progress,
                            "total": progress_total,
                        })
                    if engagement_scorer:
                        agent_sent_at = time.monotonic()
                    continue
                messages.append({
                    "role": "user",
                    "content": (
                        "[Interview timing note — not participant speech. "
                        "There is about one minute left. Ask one final, focused "
                        "question that addresses the most important unresolved "
                        "research objective. Say naturally that it is the final "
                        "question. Do not introduce more questions after it.]"
                    ),
                })
                final_question_asked = True

            # Start compressing optional material before the target, without
            # mentioning the clock or interrupting conversational momentum.
            elif (
                elapsed_after_answer >= timing.prioritize_at_seconds
                and not prioritization_injected
            ):
                messages.append({
                    "role": "user",
                    "content": (
                        "[Interview timing note — not participant speech. "
                        "Prioritize required unresolved research objectives now. "
                        "Skip optional probes, avoid repetition, and keep the "
                        "remaining turns concise. Do not mention this note.]"
                    ),
                })
                prioritization_injected = True

            # Call LLM
            try:
                # Send typing indicator
                await websocket.send_json({"type": "typing"})

                contextual_review = None
                if isinstance(structured_controller, ContextualInterviewController):
                    llm_result, contextual_review = await _contextual_agent_turn(
                        structured_controller, messages, model=agent_cfg["llm_model"],
                        study_id=agent_cfg["study_id"], participant_text=user_text,
                    )
                else:
                    llm_result = await _call_llm(
                        messages=messages, model=agent_cfg["llm_model"], study_id=agent_cfg["study_id"],
                    )
                agent_text = llm_result["content"]

                display_text = agent_text
                context_text = agent_text
                turn_closed = False

                if structured_controller is not None:
                    from app.pipeline.interview_guide import (
                        control_marker_prefix,
                        enforce_one_question_per_turn,
                        strip_control_markers,
                    )

                    # Pull hidden control data out before display, then let
                    # session-owned evidence memory reject repeated or
                    # already-resolved questions without another model call.
                    display_text, marker, decision = strip_control_markers(agent_text)
                    review = contextual_review or structured_controller.review_turn(
                        display_text,
                        marker,
                        decision,
                        user_text,
                        participant_sequence=sequence,
                        authoritative_evidence=authoritative_evidence_for_turn,
                    )
                    turn_closed = review.closed
                    display_text = review.text
                    marker = review.marker
                    decision = review.decision

                    # Semantic definition/scope questions use the same real
                    # pause as privacy questions. The shared controller has
                    # preserved the answer and removed any appended guide
                    # question; wait for an explicit yes before repeating the
                    # exact suspended question.
                    if (
                        classify_participant_turn(
                            user_text, agent_cfg["language"] or "en", infer_incomplete=False
                        )
                        == "clarification"
                        and decision is not None
                        and decision.action == "clarify"
                    ):
                        awaiting_interruption_confirmation = True

                    # One question per turn, like the voice filter. We cut the
                    # extra from the context too, otherwise a crammed-in second
                    # question gets skipped: the participant never saw it but
                    # the model thinks it asked it.
                    enforced, dropped = enforce_one_question_per_turn(display_text)
                    if dropped:
                        logger.info(
                            f"[guide:text] {session_id}: cut {dropped} chars "
                            "after the turn's first question"
                        )
                    display_text = enforced.strip()

                    # The controller can already have closed the guide when
                    # its final nudge asks the model for the exact closing
                    # message. If that model reply omits a TURN decision,
                    # review.closed may be false even though the participant
                    # sees the closing. Send the ended event in that case so
                    # the browser can finish instead of timing out.
                    if (
                        not turn_closed
                        and structured_controller.snapshot()["closed"]
                        and display_text
                        == str(runtime_guide.get("closing_message") or "").strip()
                    ):
                        turn_closed = True

                    # Keep reviewed control state in model context, never in
                    # the participant transcript.
                    prefix = control_marker_prefix(marker, decision)
                    context_text = f"{prefix} {display_text}".strip()

                    structured_controller.register_bot_turn(user_text)

                    if show_progress and marker is not None:
                        marker = min(marker, progress_total) if progress_total else marker
                        if marker > reported_progress:
                            reported_progress = marker
                            await websocket.send_json({
                                "type": "progress",
                                "current": reported_progress,
                                "total": progress_total,
                            })

                messages.append({"role": "assistant", "content": context_text})

                # Log agent turn
                sequence += 1
                await _log_turn(
                    session_id,
                    SpeakerRole.AGENT,
                    display_text,
                    sequence,
                    prompt_tokens=llm_result["prompt_tokens"],
                    completion_tokens=llm_result["completion_tokens"],
                )
                await publish_transcript_event(str(session_id), {
                    "type": "transcript",
                    "role": "agent",
                    "content": display_text,
                    "sequence": sequence,
                })

                await websocket.send_json({
                    "type": "message",
                    "text": display_text,
                    "role": "agent",
                })
                if turn_closed:
                    await websocket.send_json({
                        "type": "ended",
                        "reason": "interview_complete",
                        "text": "Thank you for your participation.",
                    })
                    final_status = SessionStatus.COMPLETED
                    final_reason = "interview_complete"
                    break
                if engagement_scorer:
                    agent_sent_at = time.monotonic()

            except Exception as llm_err:
                logger.error(f"Chat {session_id}: LLM error — {type(llm_err).__name__}")
                await websocket.send_json({
                    "type": "error",
                    "text": "Sorry, I encountered an issue. Please try again.",
                })

    except WebSocketDisconnect:
        logger.info(f"Chat {session_id}: participant disconnected")
        final_status = SessionStatus.TIMED_OUT
        final_reason = "disconnected"
    except asyncio.CancelledError:
        logger.info(f"Chat {session_id}: task cancelled")
        final_status = SessionStatus.TIMED_OUT
        final_reason = "cancelled"
    except Exception as exc:
        logger.error(f"Chat {session_id}: error — {type(exc).__name__}")
        final_status = SessionStatus.ERROR
        final_reason = "error"
    finally:
        keepalive_stop.set()
        keepalive_task.cancel()
        protocol_metadata = None
        if structured_controller is not None:
            protocol_metadata = {
                **structured_controller.stats(),
                "evidence_ledger": structured_controller.evidence_ledger(),
                "evidence_judge_model": (
                    settings.evidence_judge_model
                    if settings.evidence_judge_enabled
                    else None
                ),
                "evidence_judge_prompt_tokens": evidence_judge_prompt_tokens,
                "evidence_judge_completion_tokens": (
                    evidence_judge_completion_tokens
                ),
                "evidence_judge_candidates": evidence_judge_candidates,
                "evidence_judge_returned": evidence_judge_returned,
            }
            logger.info(
                f"Chat {session_id}: structured protocol summary "
                f"{structured_controller.stats()}"
            )
        # ── 7. Finalise session ───────────────────────────────────
        try:
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()

            async with async_session_factory() as db:
                sess = await db.get(Session, session_id)
                if sess:
                    sess.status = final_status
                    sess.ended_at = end_time
                    sess.duration_seconds = duration
                    transcript_tokens = await aggregate_session_tokens(db, session_id)
                    sess.total_tokens = (
                        transcript_tokens
                        + evidence_judge_prompt_tokens
                        + evidence_judge_completion_tokens
                    )
                    sess.completion_metadata = {
                        "reason": final_reason,
                        "timing": timing.as_dict(),
                        "participant_experience": {
                            "initial_response_mode": (
                                response_mode
                                if response_mode in {"voice", "text"}
                                else None
                            ),
                            "microphone_check_status": (
                                mic_check_status
                                if mic_check_status
                                in {"idle", "ready", "quiet", "denied", "unavailable"}
                                else None
                            ),
                            "events": participant_experience_events,
                            "milestones": sorted({
                                str(event["event"])
                                for event in participant_experience_events
                            }),
                            "transcription_failures": sum(
                                event.get("event") == "transcription_failed"
                                for event in participant_experience_events
                            ),
                        },
                        "protocol": (
                            protocol_metadata
                        ),
                    }
                    await redact_session_data(
                        db,
                        session_id,
                        session=sess,
                        ingress_counts=privacy_redaction_counts,
                    )
                    await db.commit()

            # The participant UI only redirects back to the recruitment
            # platform after receiving this post-commit acknowledgement.
            try:
                await websocket.send_json({
                    "type": "finalized",
                    "status": final_status.value,
                    "reason": final_reason,
                    "session_id": str(session_id),
                    "feedback_token": participant_feedback_token(
                        agent.widget_key, session_id
                    ),
                })
            except Exception:
                pass

            await publish_transcript_event(str(session_id), {
                "type": "session_ended",
                "status": final_status.value,
                "duration_seconds": round(duration, 1),
            })

            logger.info(
                f"Chat ended: session={session_id}, "
                f"status={final_status.value}, duration={duration:.1f}s"
            )
            await unregister_session(session_id)

        except Exception as cleanup_exc:
            logger.error(f"Chat {session_id}: failed to finalise — {type(cleanup_exc).__name__}")
