"""End-to-end endpoint checks for spoken privacy warnings."""

import json
from unittest.mock import AsyncMock

import pytest

from app.privacy import PRIVACY_WARNING


class _VoiceSocket:
    def __init__(self, incoming: list[dict] | None = None):
        self.incoming = [json.dumps(item) for item in (incoming or [])]
        self.sent: list[dict] = []
        self.accepted = False
        self.closed = None

    async def accept(self):
        self.accepted = True

    async def receive_text(self):
        return self.incoming.pop(0)

    async def send_json(self, payload):
        self.sent.append(payload)

    async def close(self, code=None, reason=None):
        self.closed = {"code": code, "reason": reason}


class _PipelineTask:
    def __init__(self, notify):
        self.notify = notify
        self.queued_frames = []

    async def queue_frame(self, frame):
        self.queued_frames.append(frame)


class _PrivacyWarningRunner:
    def __init__(self, *args, **kwargs):
        pass

    async def run(self, task):
        await task.notify(
            {
                "type": "privacy_warning",
                "text": PRIVACY_WARNING,
                "categories": {"email": 1},
            }
        )


@pytest.mark.asyncio
async def test_browser_voice_warning_is_visible_and_queued_for_speech(
    monkeypatch, db_session_factory, make_study, make_agent
):
    import app.api.interviews as interviews
    import app.pipeline.runner as pipeline_runner
    import pipecat.pipeline.runner as pipecat_runner

    study = make_study(title="Browser voice privacy delivery")
    agent = make_agent(study_id=study.id, name="Browser voice agent")
    async with db_session_factory() as db:
        db.add_all([study, agent])
        await db.commit()

    holder = {}

    async def build_pipeline(**kwargs):
        holder["task"] = _PipelineTask(kwargs["notify_callback"])
        return holder["task"], None

    monkeypatch.setattr(interviews, "async_session_factory", db_session_factory)
    monkeypatch.setattr(
        interviews, "validate_agent_pipeline_config", AsyncMock(return_value=[])
    )
    monkeypatch.setattr(interviews, "register_session", AsyncMock())
    monkeypatch.setattr(interviews, "unregister_session", AsyncMock())
    monkeypatch.setattr(interviews, "publish_transcript_event", AsyncMock())
    monkeypatch.setattr(pipeline_runner, "build_pipeline", build_pipeline)
    monkeypatch.setattr(pipecat_runner, "PipelineRunner", _PrivacyWarningRunner)

    websocket = _VoiceSocket()
    await interviews.interview_ws(
        websocket,
        agent.widget_key,
        pid=None,
        external_study_id=None,
        external_session_id=None,
        consent=False,
    )

    assert any(
        item.get("type") == "privacy_warning" and item.get("text") == PRIVACY_WARNING
        for item in websocket.sent
    )
    assert len(holder["task"].queued_frames) == 1
    assert holder["task"].queued_frames[0].text == PRIVACY_WARNING
    assert holder["task"].queued_frames[0].append_to_context is False


@pytest.mark.asyncio
async def test_twilio_warning_is_queued_for_caller_audio(
    monkeypatch, db_session_factory, make_study, make_agent
):
    import app.api.twilio as twilio
    import app.pipeline.runner as pipeline_runner
    import pipecat.pipeline.runner as pipecat_runner

    study = make_study(title="Twilio privacy delivery")
    agent = make_agent(study_id=study.id, name="Telephone voice agent")
    async with db_session_factory() as db:
        db.add_all([study, agent])
        await db.commit()

    holder = {}

    async def build_twilio_pipeline(**kwargs):
        holder["task"] = _PipelineTask(kwargs["notify_callback"])
        return holder["task"]

    monkeypatch.setattr(twilio, "async_session_factory", db_session_factory)
    monkeypatch.setattr(twilio, "_agent_config_errors", AsyncMock(return_value=[]))
    monkeypatch.setattr(twilio, "publish_transcript_event", AsyncMock())
    monkeypatch.setattr(pipeline_runner, "build_twilio_pipeline", build_twilio_pipeline)
    monkeypatch.setattr(pipecat_runner, "PipelineRunner", _PrivacyWarningRunner)

    websocket = _VoiceSocket(
        [
            {
                "event": "start",
                "start": {"streamSid": "MZsynthetic", "callSid": "CAsynthetic"},
            }
        ]
    )
    await twilio.twilio_media_stream(websocket, str(agent.id))

    assert websocket.accepted is True
    assert len(holder["task"].queued_frames) == 1
    assert holder["task"].queued_frames[0].text == PRIVACY_WARNING
    assert holder["task"].queued_frames[0].append_to_context is False
