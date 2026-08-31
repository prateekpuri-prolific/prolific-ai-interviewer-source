"""Turn-based speech-to-text for the text-first participant interview.

The browser records one participant answer at a time.  This adapter sends the
complete clip to the agent's configured STT provider and returns only the
transcript; audio is not persisted by OASIS on this path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import httpx

from app.config import settings


async def _get_key(field: str) -> str:
    try:
        from app.api.settings import get_effective_key

        return await get_effective_key(field)
    except Exception:
        return str(getattr(settings, field, "") or "")


async def _openai_base_url() -> str:
    try:
        from app.api.settings import get_effective_flag

        use_eu = await get_effective_flag("openai_use_eu")
    except Exception:
        use_eu = bool(getattr(settings, "openai_use_eu", False))
    return "https://eu.api.openai.com/v1" if use_eu else "https://api.openai.com/v1"


def _compatible_endpoint(base_url: str, path: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/{path.lstrip('/')}"
    return f"{base}/v1/{path.lstrip('/')}"


def _filename(original: Optional[str], content_type: Optional[str]) -> str:
    suffix = Path(original or "").suffix.lower()
    if suffix in {".webm", ".wav", ".mp3", ".m4a", ".mp4", ".ogg", ".oga"}:
        return f"answer{suffix}"
    by_type = {
        "audio/webm": ".webm",
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/mpeg": ".mp3",
        "audio/mp4": ".m4a",
        "audio/ogg": ".ogg",
    }
    return f"answer{by_type.get((content_type or '').split(';')[0].lower(), '.webm')}"


def _json_transcript(payload: dict) -> str:
    text = payload.get("text") or payload.get("transcript") or ""
    if not text and isinstance(payload.get("results"), dict):
        channels = payload["results"].get("channels") or []
        if channels:
            alternatives = channels[0].get("alternatives") or []
            if alternatives:
                text = alternatives[0].get("transcript") or ""
    return str(text).strip()


async def transcribe_audio(
    *,
    provider: str,
    model: Optional[str],
    audio: bytes,
    filename: Optional[str],
    content_type: Optional[str],
    language: Optional[str],
) -> str:
    """Transcribe one complete participant response clip."""
    provider = (provider or "openai").lower()
    media_type = (content_type or "audio/webm").split(";")[0]
    upload_name = _filename(filename, media_type)
    timeout = httpx.Timeout(90.0, connect=10.0)
    language_data = {"language": language} if language else {}

    if provider == "openai":
        api_key = await _get_key("openai_api_key")
        if not api_key:
            raise ValueError("OpenAI speech-to-text is not configured")
        effective_model = model or "gpt-4o-mini-transcribe"
        if effective_model == "gpt-realtime-whisper":
            effective_model = "gpt-4o-mini-transcribe"
        endpoint = f"{await _openai_base_url()}/audio/transcriptions"
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                endpoint,
                headers={"Authorization": f"Bearer {api_key}"},
                files={"file": (upload_name, audio, media_type)},
                data={"model": effective_model, **language_data},
            )
            response.raise_for_status()
            return _json_transcript(response.json())

    if provider == "deepgram":
        api_key = await _get_key("deepgram_api_key")
        if not api_key:
            raise ValueError("Deepgram speech-to-text is not configured")
        params = {"model": model or "nova-3", "smart_format": "true"}
        if language:
            params["language"] = language
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                "https://api.deepgram.com/v1/listen",
                params=params,
                headers={"Authorization": f"Token {api_key}", "Content-Type": media_type},
                content=audio,
            )
            response.raise_for_status()
            return _json_transcript(response.json())

    if provider == "scaleway":
        api_key = await _get_key("scaleway_secret_key")
        if not api_key:
            raise ValueError("Scaleway speech-to-text is not configured")
        endpoint = f"{settings.scaleway_api_url.rstrip('/')}/audio/transcriptions"
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                endpoint,
                headers={"Authorization": f"Bearer {api_key}"},
                files={"file": (upload_name, audio, media_type)},
                data={"model": model or "whisper-large-v3", **language_data},
            )
            response.raise_for_status()
            return _json_transcript(response.json())

    if provider == "self_hosted":
        base_url = await _get_key("self_hosted_stt_url") or settings.self_hosted_stt_url
        if not base_url:
            raise ValueError("Self-hosted speech-to-text is not configured")
        api_key = await _get_key("self_hosted_stt_api_key") or "not-needed"
        endpoint = _compatible_endpoint(base_url, "audio/transcriptions")
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                endpoint,
                headers={"Authorization": f"Bearer {api_key}"},
                files={"file": (upload_name, audio, media_type)},
                data={
                    "model": model or settings.self_hosted_stt_model or "whisper-1",
                    **language_data,
                },
            )
            response.raise_for_status()
            return _json_transcript(response.json())

    raise ValueError(f"Turn-based transcription is not supported for STT provider '{provider}'")
