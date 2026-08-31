"""
OASIS — FastAPI application entry point.
"""

from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

import httpx
from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from fastapi import Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.router import api_router
from app.api.interviews import router as interviews_router
from app.api.text_chat import router as text_chat_router
from app.api.monitor import router as monitor_router
from app.api.twilio import router as twilio_router
from app.config import settings
from app.database import engine, get_db
from app.redis import close_redis, get_redis


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup / shutdown lifecycle hook."""
    import asyncio
    from app.session_manager import session_cleanup_loop

    # ── Startup ──
    # Verify Redis is reachable
    redis = await get_redis()
    await redis.ping()

    # Start background session cleanup loop
    cleanup_task = asyncio.create_task(session_cleanup_loop(interval_seconds=60))

    yield

    # ── Shutdown ──
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass

    await engine.dispose()
    await close_redis()


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

# ── CORS (permissive for development) ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.debug else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API Routes ───────────────────────────────────────────────
app.include_router(api_router)

# ── WebSocket Routes (outside /api prefix) ───────────────────
app.include_router(interviews_router)
app.include_router(text_chat_router)
app.include_router(monitor_router)

# ── Twilio Routes (voice webhook + media streams WebSocket) ──
app.include_router(twilio_router)


class WidgetSpeechRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)

    @field_validator("text")
    @classmethod
    def _strip_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Text must not be blank")
        return value


# ── Public Widget Config (no auth required) ──────────────────
@app.get("/api/widget/{widget_key}", tags=["widget"])
async def widget_config(widget_key: str, db: AsyncSession = Depends(get_db)):
    """
    Public endpoint for the interview widget to fetch its configuration.
    Returns title, description, colour, participant ID mode, etc.
    """
    from app.models.agent import Agent, AgentStatus as AS
    from app.models.study import Study
    from app.research.protocol import participant_information

    result = await db.execute(
        select(Agent).where(
            Agent.widget_key == widget_key,
            Agent.status == AS.ACTIVE.value,
        )
    )
    agent = result.scalar_one_or_none()
    if not agent:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Agent not found or inactive")

    study = await db.get(Study, agent.study_id)
    research_spec = study.research_spec if study else None

    _interview_mode = (
        agent.interview_mode.value
        if hasattr(agent.interview_mode, "value")
        else (agent.interview_mode or "free_form")
    )
    _pipeline_type = (
        agent.pipeline_type.value
        if hasattr(agent.pipeline_type, "value")
        else (agent.pipeline_type or "modular")
    )

    return {
        "widget_key": agent.widget_key,
        "modality": (
            agent.modality.value
            if hasattr(agent.modality, "value")
            else (agent.modality or "voice")
        ),
        "avatar": agent.avatar or "neutral",
        "widget_title": agent.widget_title,
        "widget_description": agent.widget_description,
        "widget_primary_color": agent.widget_primary_color or "#0D7377",
        "widget_listening_message": agent.widget_listening_message or "Agent is listening…",
        "participant_id_mode": agent.participant_id_mode.value
            if hasattr(agent.participant_id_mode, "value")
            else agent.participant_id_mode,
        "interview_mode": _interview_mode,
        # Effective progress flag: only true when the bar can actually advance
        # (structured, and not a voice-to-voice pipeline that can't strip the
        # hidden marker). Keeps stale V2V configs from rendering a dead bar.
        "widget_show_progress": bool(
            agent.widget_show_progress
            and _interview_mode == "structured"
            and _pipeline_type != "voice_to_voice"
        ),
        # Total main questions, so the participant UI can render the bar
        # without ever seeing the guide contents.
        "question_count": len((agent.interview_guide or {}).get("questions", []))
            if _interview_mode == "structured"
            else 0,
        # Audio is a participant-controlled enhancement to the text-first UI.
        # It is always off on page load and never enables microphone capture.
        "audio_available": agent.tts_provider in {
            "openai", "elevenlabs", "cartesia", "self_hosted"
        },
        "audio_default_on": False,
        # Participant voice answers use the same text-chat orchestration after
        # turn-based transcription. Typing remains available as a fallback.
        "voice_response_available": agent.stt_provider in {
            "openai", "deepgram", "scaleway", "self_hosted"
        },
        "response_mode_default": "voice",
        "welcome_message": agent.welcome_message,
        "language": agent.language,
        "participant_information": participant_information(research_spec),
        "fielding": (research_spec or {}).get("fielding") or {},
        "research_protocol_version": (research_spec or {}).get("version"),
        "research_stage": (research_spec or {}).get("stage"),
        "source_code_url": settings.source_code_url or None,
    }


@app.post("/api/widget/{widget_key}/speech", tags=["widget"])
async def widget_speech(
    widget_key: str,
    payload: WidgetSpeechRequest,
    db: AsyncSession = Depends(get_db),
):
    """Synthesize one visible interviewer turn after explicit audio opt-in."""
    from app.models.agent import Agent, AgentStatus as AS
    from app.providers.tts_http import synthesize_text

    result = await db.execute(
        select(Agent).where(
            Agent.widget_key == widget_key,
            Agent.status == AS.ACTIVE.value,
        )
    )
    agent = result.scalar_one_or_none()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found or inactive")

    try:
        content, media_type = await synthesize_text(
            provider=agent.tts_provider,
            model=agent.tts_model,
            voice=agent.tts_voice,
            text=payload.text,
        )
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail="The interview audio service is temporarily unavailable",
        ) from exc

    return Response(
        content=content,
        media_type=media_type,
        headers={"Cache-Control": "private, max-age=3600"},
    )


@app.post("/api/widget/{widget_key}/transcribe", tags=["widget"])
async def widget_transcribe(
    widget_key: str,
    audio: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """Transcribe one participant voice answer without storing the audio."""
    from app.models.agent import Agent, AgentStatus as AS
    from app.providers.stt_http import transcribe_audio

    result = await db.execute(
        select(Agent).where(
            Agent.widget_key == widget_key,
            Agent.status == AS.ACTIVE.value,
        )
    )
    agent = result.scalar_one_or_none()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found or inactive")

    content_type = (audio.content_type or "").lower()
    if content_type and not (content_type.startswith("audio/") or content_type == "video/webm"):
        raise HTTPException(status_code=415, detail="Please upload an audio recording")

    content = await audio.read(25 * 1024 * 1024 + 1)
    if not content:
        raise HTTPException(status_code=400, detail="The recording was empty")
    if len(content) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="The recording is too large")

    try:
        transcript = await transcribe_audio(
            provider=agent.stt_provider,
            model=agent.stt_model,
            audio=content,
            filename=audio.filename,
            content_type=audio.content_type,
            language=agent.language,
        )
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail="The transcription service is temporarily unavailable",
        ) from exc

    if not transcript:
        raise HTTPException(
            status_code=422,
            detail="We could not hear a response. Please try again or type instead.",
        )
    return {"text": transcript}


# ── Health Check ─────────────────────────────────────────────
@app.get("/api/health", tags=["system"])
async def health_check():
    """
    Verify that the API, database, and Redis are all reachable.
    Returns service-level status for each dependency.
    """
    status: dict[str, str] = {"api": "ok"}

    # Check PostgreSQL
    try:
        async with engine.connect() as conn:
            await conn.execute(
                __import__("sqlalchemy").text("SELECT 1")
            )
        status["database"] = "ok"
    except Exception as exc:
        status["database"] = f"error: {exc}"

    # Check Redis
    try:
        redis = await get_redis()
        await redis.ping()
        status["redis"] = "ok"
    except Exception as exc:
        status["redis"] = f"error: {exc}"

    healthy = all(v == "ok" for v in status.values())
    return {"healthy": healthy, "services": status}
