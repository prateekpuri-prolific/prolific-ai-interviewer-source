"""On-demand TTS for the text-first participant interview.

Audio is opt-in in the browser. This adapter deliberately returns a complete
audio response rather than opening the realtime voice pipeline, so enabling
question audio never turns on the participant's microphone.
"""

from __future__ import annotations

from typing import Optional

import httpx

from app.config import settings


async def _get_key(field: str) -> str:
    try:
        from app.api.settings import get_effective_key

        return await get_effective_key(field)
    except Exception:
        return getattr(settings, field, "")


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


async def synthesize_text(
    *,
    provider: str,
    model: Optional[str],
    voice: Optional[str],
    text: str,
) -> tuple[bytes, str]:
    """Synthesize ``text`` with the agent's configured TTS provider."""
    provider = (provider or "openai").lower()
    timeout = httpx.Timeout(45.0, connect=10.0)

    if provider == "openai":
        api_key = await _get_key("openai_api_key")
        if not api_key:
            raise ValueError("OpenAI TTS is not configured")
        endpoint = f"{await _openai_base_url()}/audio/speech"
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                endpoint,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model or "gpt-4o-mini-tts",
                    "voice": voice or "alloy",
                    "input": text,
                    "response_format": "mp3",
                },
            )
            response.raise_for_status()
            return response.content, "audio/mpeg"

    if provider == "elevenlabs":
        from app.pipeline.runner import _resolve_elevenlabs_voice

        api_key = await _get_key("elevenlabs_api_key")
        if not api_key:
            raise ValueError("ElevenLabs TTS is not configured")
        voice_id = _resolve_elevenlabs_voice(voice)
        endpoint = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                endpoint,
                params={"output_format": "mp3_44100_128"},
                headers={"xi-api-key": api_key, "Content-Type": "application/json"},
                json={"text": text, "model_id": model or "eleven_multilingual_v2"},
            )
            response.raise_for_status()
            return response.content, "audio/mpeg"

    if provider == "cartesia":
        api_key = await _get_key("cartesia_api_key")
        if not api_key:
            raise ValueError("Cartesia TTS is not configured")
        endpoint = "https://api.cartesia.ai/tts/bytes"
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                endpoint,
                headers={
                    "X-API-Key": api_key,
                    "Cartesia-Version": "2024-06-10",
                    "Content-Type": "application/json",
                },
                json={
                    "model_id": model or "sonic-2",
                    "transcript": text,
                    "voice": {
                        "mode": "id",
                        "id": voice or "a0e99841-438c-4a64-b679-ae501e7d6091",
                    },
                    "output_format": {
                        "container": "wav",
                        "encoding": "pcm_f32le",
                        "sample_rate": 44100,
                    },
                },
            )
            response.raise_for_status()
            return response.content, "audio/wav"

    if provider == "self_hosted":
        base_url = await _get_key("self_hosted_tts_url") or settings.self_hosted_tts_url
        if not base_url:
            raise ValueError("Self-hosted TTS is not configured")
        api_key = await _get_key("self_hosted_tts_api_key") or "not-needed"
        endpoint = _compatible_endpoint(base_url, "audio/speech")
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                endpoint,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model or settings.self_hosted_tts_model or "tts-1",
                    "voice": voice or "alloy",
                    "input": text,
                    "response_format": "mp3",
                },
            )
            response.raise_for_status()
            return response.content, "audio/mpeg"

    raise ValueError(f"On-demand audio is not supported for TTS provider '{provider}'")
