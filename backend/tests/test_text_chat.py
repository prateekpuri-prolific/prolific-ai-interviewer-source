"""Tests for text-chat LLM routing and privacy-critical interview behavior."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import select

from app.models.session import Session, SessionStatus, SpeakerRole, TranscriptEntry
from app.privacy import REDACTOR_VERSION


def _mock_litellm_response(content: str = "Hello"):
    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 5
    choice = MagicMock()
    choice.message.content = content
    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


class TestCallLlm:
    @pytest.mark.asyncio
    @patch("litellm.acompletion", new_callable=AsyncMock)
    @patch("app.api.text_chat._get_key", new_callable=AsyncMock)
    async def test_anthropic_uses_anthropic_api_key(self, mock_get_key, mock_acompletion):
        mock_get_key.side_effect = lambda field: {
            "anthropic_api_key": "sk-ant-test",
            "openai_api_key": "sk-openai",
        }.get(field, "")
        mock_acompletion.return_value = _mock_litellm_response()

        from app.api.text_chat import _call_llm

        await _call_llm(
            [{"role": "user", "content": "Hi"}],
            model="anthropic/claude-sonnet-4-6",
        )

        kwargs = mock_acompletion.call_args.kwargs
        assert kwargs["api_key"] == "sk-ant-test"
        assert kwargs["model"] == "anthropic/claude-sonnet-4-6"
        assert "api_base" not in kwargs

    @pytest.mark.asyncio
    @patch("litellm.acompletion", new_callable=AsyncMock)
    @patch("app.api.text_chat._get_key", new_callable=AsyncMock)
    async def test_anthropic_missing_key_raises(self, mock_get_key, mock_acompletion):
        mock_get_key.return_value = ""

        from app.api.text_chat import _call_llm

        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            await _call_llm(
                [{"role": "user", "content": "Hi"}],
                model="anthropic/claude-sonnet-4-6",
            )

        mock_acompletion.assert_not_called()

    @pytest.mark.asyncio
    @patch("litellm.acompletion", new_callable=AsyncMock)
    @patch("app.api.text_chat._openai_api_base", new_callable=AsyncMock)
    @patch("app.api.text_chat._get_key", new_callable=AsyncMock)
    async def test_openai_routes_to_eu_when_flag_on(
        self, mock_get_key, mock_eu_base, mock_acompletion
    ):
        mock_get_key.return_value = "sk-openai-test"
        mock_eu_base.return_value = "https://eu.api.openai.com/v1"
        mock_acompletion.return_value = _mock_litellm_response()

        from app.api.text_chat import _call_llm

        await _call_llm(
            [{"role": "user", "content": "Hi"}],
            model="openai/gpt-4o-mini",
        )

        kwargs = mock_acompletion.call_args.kwargs
        assert kwargs["api_key"] == "sk-openai-test"
        assert kwargs["api_base"] == "https://eu.api.openai.com/v1"

    @pytest.mark.asyncio
    @patch("litellm.acompletion", new_callable=AsyncMock)
    @patch("app.api.text_chat._openai_api_base", new_callable=AsyncMock)
    @patch("app.api.text_chat._get_key", new_callable=AsyncMock)
    async def test_openai_skips_eu_when_flag_off(
        self, mock_get_key, mock_eu_base, mock_acompletion
    ):
        mock_get_key.return_value = "sk-openai-test"
        mock_eu_base.return_value = None
        mock_acompletion.return_value = _mock_litellm_response()

        from app.api.text_chat import _call_llm

        await _call_llm(
            [{"role": "user", "content": "Hi"}],
            model="openai/gpt-4o-mini",
        )

        kwargs = mock_acompletion.call_args.kwargs
        assert "api_base" not in kwargs

    @pytest.mark.asyncio
    @patch("app.api.text_chat._call_openai_responses", new_callable=AsyncMock)
    @patch("app.api.text_chat._openai_api_base", new_callable=AsyncMock)
    @patch("app.api.text_chat._get_key", new_callable=AsyncMock)
    async def test_gpt_5_6_uses_responses_api(
        self, mock_get_key, mock_eu_base, mock_responses
    ):
        mock_get_key.return_value = "sk-openai-test"
        mock_eu_base.return_value = "https://eu.api.openai.com/v1"
        mock_responses.return_value = {
            "content": "Hello",
            "prompt_tokens": 10,
            "completion_tokens": 5,
        }

        from app.api.text_chat import _call_llm

        await _call_llm(
            [{"role": "user", "content": "Hi"}],
            model="openai/gpt-5.6-luna",
        )

        mock_responses.assert_awaited_once_with(
            [{"role": "user", "content": "Hi"}],
            "openai/gpt-5.6-luna",
            "sk-openai-test",
            "https://eu.api.openai.com/v1",
        )

    @pytest.mark.asyncio
    @patch("litellm.acompletion", new_callable=AsyncMock)
    @patch("app.api.text_chat._openai_api_base", new_callable=AsyncMock)
    @patch("app.api.text_chat._get_key", new_callable=AsyncMock)
    async def test_anthropic_does_not_use_openai_eu_base(
        self, mock_get_key, mock_eu_base, mock_acompletion
    ):
        mock_get_key.side_effect = lambda field: {
            "anthropic_api_key": "sk-ant-test",
            "openai_api_key": "sk-openai",
        }.get(field, "")
        mock_eu_base.return_value = "https://eu.api.openai.com/v1"
        mock_acompletion.return_value = _mock_litellm_response()

        from app.api.text_chat import _call_llm

        await _call_llm(
            [{"role": "user", "content": "Hi"}],
            model="anthropic/claude-sonnet-4-6",
        )

        kwargs = mock_acompletion.call_args.kwargs
        assert "api_base" not in kwargs


class _SyntheticParticipantSocket:
    """Minimal WebSocket double for exercising the complete chat endpoint."""

    def __init__(self, incoming: list[dict]):
        self.incoming = [json.dumps(message) for message in incoming]
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


@pytest.mark.asyncio
async def test_structured_closing_without_turn_marker_sends_ended_event(
    monkeypatch, db_session_factory, make_study, make_agent
):
    """A visible closing must complete the session even without TURN metadata."""
    import app.api.text_chat as text_chat
    from app.models.agent import InterviewMode

    study = make_study(title="Synthetic closing test")
    study.research_spec = {
        "version": "closing-test-v1",
        "participant_information": {"consent_required": True},
    }
    agent = make_agent(study_id=study.id, welcome_message="Shall we begin?")
    agent.interview_mode = InterviewMode.STRUCTURED
    agent.interview_guide = {
        "orchestration_profile": "semi_structured",
        "questions": [
            {
                "id": "warmup",
                "text": "Do you use AI day to day?",
                "phase": "warm_up",
                "always_ask": True,
                "evidence_goal": [],
                "target_ids": [],
                "probes": [],
                "max_follow_ups": 0,
            }
        ],
        "closing_message": "Thank you. This completes the interview.",
    }
    async with db_session_factory() as db:
        db.add_all([study, agent])
        await db.commit()

    websocket = _SyntheticParticipantSocket([
        {"type": "message", "text": "Yes, let's begin."},
        {"type": "message", "text": "No, I do not use AI day to day."},
    ])
    mock_llm = AsyncMock(side_effect=[
        {
            "content": (
                '[[TURN {"action":"main","question":1,'
                '"intent":"warmup:main","resolved":{}}]] '
                "[[Q1]] Do you use AI day to day?"
            ),
            "prompt_tokens": 20,
            "completion_tokens": 12,
        },
        {
            "content": "Thank you. This completes the interview.",
            "prompt_tokens": 20,
            "completion_tokens": 12,
        },
    ])
    monkeypatch.setattr(text_chat, "async_session_factory", db_session_factory)
    monkeypatch.setattr(text_chat, "validate_agent_pipeline_config", AsyncMock(return_value=[]))
    monkeypatch.setattr(text_chat, "register_session", AsyncMock())
    monkeypatch.setattr(text_chat, "unregister_session", AsyncMock())
    monkeypatch.setattr(text_chat, "publish_transcript_event", AsyncMock())
    monkeypatch.setattr(text_chat, "_call_llm", mock_llm)
    monkeypatch.setattr(
        text_chat,
        "_redact_participant_text",
        AsyncMock(side_effect=text_chat.redact_text),
    )

    await text_chat.text_chat_ws(
        websocket,
        agent.widget_key,
        pid=None,
        external_study_id=None,
        external_session_id=None,
        consent=True,
        response_mode="text",
        mic_check_status=None,
    )

    assert mock_llm.await_count == 2
    assert any(
        item.get("type") == "ended" and item.get("reason") == "interview_complete"
        for item in websocket.sent
    )
    assert any(
        item.get("type") == "finalized" and item.get("status") == "completed"
        for item in websocket.sent
    )
    async with db_session_factory() as db:
        session = await db.scalar(select(Session).where(Session.agent_id == agent.id))
        assert session.status == SessionStatus.COMPLETED


@pytest.mark.asyncio
@pytest.mark.parametrize("stop,reply", [
    (False, "I use AI for coding. My email is synthetic.person@example.com."),
    (False, "I don’t have anything more to add about that."),
    (True, "I want to stop the interview."),
])
async def test_contextual_websocket_observes_sanitized_answers_and_finalizes(
    monkeypatch, db_session_factory, make_study, make_agent, stop, reply
):
    import app.api.text_chat as api
    import app.pipeline.conversation_policy as policy
    from app.models.agent import InterviewMode

    study = make_study(title="Synthetic contextual integration")
    agent = make_agent(study_id=study.id, welcome_message="Shall we begin?")
    agent.interview_mode = InterviewMode.STRUCTURED
    agent.interview_guide = {
        "conversation_policy": "contextual_v1", "orchestration_profile": "exploratory",
        "questions": [{"id": "warmup", "text": "Do you use AI day to day?",
                       "phase": "warm_up", "always_ask": True, "max_follow_ups": 0}],
        "closing_message": "Thank you. This completes the interview.",
    }
    async with db_session_factory() as db:
        db.add_all([study, agent])
        await db.commit()
    socket = _SyntheticParticipantSocket([
        {"type": "message", "text": "Yes, let's begin."},
        {"type": "message", "text": reply},
    ])
    llm = AsyncMock(side_effect=[
        {"content": '[[TURN {"action":"main","question":1,"intent":"warmup:main"}]] Do you use AI day to day?', "prompt_tokens": 1, "completion_tokens": 1},
        {"content": '[[TURN {"action":"close","question":null,"intent":null}]] Thank you.', "prompt_tokens": 1, "completion_tokens": 1},
    ])
    observer = AsyncMock(return_value=(policy.Observation(
        updates=[], opportunities=[], ai_use="unknown", ai_use_quote="", boundaries=[]),
        {"prompt_tokens": 1, "completion_tokens": 1}))
    monkeypatch.setattr(api, "async_session_factory", db_session_factory)
    for name in ("register_session", "unregister_session", "publish_transcript_event"):
        monkeypatch.setattr(api, name, AsyncMock())
    monkeypatch.setattr(api, "validate_agent_pipeline_config", AsyncMock(return_value=[]))
    monkeypatch.setattr(api, "_call_llm", llm)
    monkeypatch.setattr(api, "_get_key", AsyncMock(return_value="test"))
    monkeypatch.setattr(api, "_openai_api_base", AsyncMock(return_value=None))
    monkeypatch.setattr(api, "_redact_participant_text", AsyncMock(side_effect=api.redact_text))
    monkeypatch.setattr(policy, "observe_conversation", observer)
    monkeypatch.setattr(policy, "check_question", AsyncMock(return_value=(
        policy.QuestionCheck(acceptable=True, issue="none", correction=""),
        {"prompt_tokens": 1, "completion_tokens": 1})))
    await api.text_chat_ws(socket, agent.widget_key, pid=None, external_study_id=None,
                           external_session_id=None, consent=True, response_mode="text", mic_check_status=None)
    assert any(m.get("type") == "finalized" and m.get("status") == "completed" for m in socket.sent)
    assert any(m.get("type") == "ended" and m.get("reason") ==
               ("participant_finished" if stop else "interview_complete") for m in socket.sent)
    assert observer.await_count == (0 if stop else 1)
    if not stop:
        context = observer.call_args.kwargs["payload"]
        assert context["actual_question"] == "Do you use AI day to day?"
        assert len(context["conversation"]) == 1  # readiness is not research evidence
        assert "synthetic.person@example.com" not in json.dumps(context)
    async with db_session_factory() as db:
        session = await db.scalar(select(Session).where(Session.agent_id == agent.id))
        assert session.status == SessionStatus.COMPLETED
        assert "contextual_v1" in json.dumps(session.completion_metadata)
        assert "synthetic.person@example.com" not in json.dumps(session.completion_metadata)


@pytest.mark.asyncio
async def test_synthetic_pii_disclosure_warns_continues_and_never_persists_or_exports_raw_value(
    monkeypatch,
    db_session_factory,
    client,
    make_study,
    make_agent,
):
    """Exercise participant input -> warning -> model -> DB -> researcher export.

    The external model is replaced with a deterministic response, but the
    production WebSocket handler, privacy redactor, persistence layer,
    completion sweep, and export endpoint are all used unchanged.
    """
    import app.api.text_chat as text_chat

    secrets = [
        "person@example.com",
        "+1 (212) 555-0199",
        "My daughter was diagnosed with depression and is receiving treatment.",
        "I'm the only pediatric neurosurgeon in a very small island community.",
    ]
    study = make_study(title="Synthetic privacy disclosure study")
    study.research_spec = {
        "version": "privacy-e2e-v1",
        "participant_information": {
            "consent_required": True,
            "consent_label": "I agree to take part.",
        },
    }
    agent = make_agent(
        study_id=study.id,
        name="Synthetic privacy interviewer",
        welcome_message="Tell me about your experience without identifying anyone.",
    )
    async with db_session_factory() as db:
        db.add_all([study, agent])
        await db.commit()

    websocket = _SyntheticParticipantSocket(
        [
            {
                "type": "message",
                "text": f"You can follow up with me at {secrets[0]}. The experience was frustrating.",
            },
            {
                "type": "message",
                "text": f"My phone number is {secrets[1]}. I still want to continue.",
            },
            {"type": "message", "text": secrets[2]},
            {"type": "message", "text": secrets[3]},
            {"type": "finish"},
        ]
    )
    model_reply = (
        "Thanks. Without sharing identifying details, what made the experience "
        "frustrating?"
    )
    mock_llm = AsyncMock(
        return_value={
            "content": model_reply,
            "prompt_tokens": 20,
            "completion_tokens": 12,
        }
    )

    monkeypatch.setattr(text_chat, "async_session_factory", db_session_factory)
    monkeypatch.setattr(
        text_chat, "validate_agent_pipeline_config", AsyncMock(return_value=[])
    )
    monkeypatch.setattr(text_chat, "register_session", AsyncMock())
    monkeypatch.setattr(text_chat, "unregister_session", AsyncMock())
    monkeypatch.setattr(text_chat, "publish_transcript_event", AsyncMock())
    monkeypatch.setattr(text_chat, "_call_llm", mock_llm)
    monkeypatch.setattr(
        text_chat,
        "_redact_participant_text",
        AsyncMock(side_effect=text_chat.redact_text),
    )

    await text_chat.text_chat_ws(
        websocket,
        agent.widget_key,
        pid=None,
        external_study_id=None,
        external_session_id=None,
        consent=True,
        response_mode="text",
        mic_check_status=None,
    )

    assert websocket.accepted is True
    warnings = [
        item
        for item in websocket.sent
        if item.get("role") == "agent"
        and "I removed identifying details" in item.get("text", "")
    ]
    assert len(warnings) == 4
    assert all(item["type"] == "privacy_warning" for item in warnings)
    assert all(secret not in str(warnings) for secret in secrets)
    assert any(item.get("text") == model_reply for item in websocket.sent)
    assert any(item.get("type") == "finalized" for item in websocket.sent)

    assert mock_llm.await_count == 2
    redirects = [item for item in websocket.sent if item.get("text", "").startswith("Let's stay with the same question.")]
    assert len(redirects) == 2
    all_model_messages = str(
        [call.kwargs["messages"] for call in mock_llm.await_args_list]
    )
    assert all(secret not in all_model_messages for secret in secrets)
    assert "[EMAIL]" in all_model_messages
    assert "[PHONE_OR_ID]" in all_model_messages
    assert "[SENSITIVE_CONTENT_REMOVED]" in all_model_messages
    assert "[POTENTIALLY_IDENTIFYING_CONTEXT_REMOVED]" in all_model_messages

    async with db_session_factory() as db:
        session = await db.scalar(select(Session).where(Session.agent_id == agent.id))
        entries = list(
            (
                await db.scalars(
                    select(TranscriptEntry)
                    .where(TranscriptEntry.session_id == session.id)
                    .order_by(TranscriptEntry.sequence)
                )
            ).all()
        )
        assert session.status == SessionStatus.COMPLETED
        assert session.participant_consent_confirmed is True
        assert session.participant_consent_at is not None
        stored_content = str([entry.content for entry in entries])
        assert all(secret not in stored_content for secret in secrets)
        assert "[EMAIL]" in stored_content
        assert "[PHONE_OR_ID]" in stored_content
        assert "[SENSITIVE_CONTENT_REMOVED]" in stored_content
        assert "[POTENTIALLY_IDENTIFYING_CONTEXT_REMOVED]" in stored_content
        assert model_reply in [entry.content for entry in entries]
        assert session.completion_metadata["privacy"] == {
            "redactor_version": REDACTOR_VERSION,
            "ingress_redaction_count": 4,
            "ingress_categories": {
                "email": 1,
                "indirect_identifier": 1,
                "phone_or_id": 1,
                "special_category": 1,
            },
            "completion_redaction_count": 0,
            "completion_categories": {},
            "completion_check_passed": True,
        }

    export = await client.get(
        f"/api/studies/{study.id}/agents/{agent.id}/sessions/export/json"
    )
    assert export.status_code == 200
    assert all(secret not in export.text for secret in secrets)
    assert "[EMAIL]" in export.text
    assert "[PHONE_OR_ID]" in export.text
    assert "[SENSITIVE_CONTENT_REMOVED]" in export.text
    assert "[POTENTIALLY_IDENTIFYING_CONTEXT_REMOVED]" in export.text
    assert model_reply in export.text
