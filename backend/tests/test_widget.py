"""
Tests for the public widget configuration endpoint.

This is a public-facing endpoint (no auth) so it's security-relevant.
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.session import Session, SessionStatus
from app.research.participant_feedback import participant_feedback_token


class TestWidgetEndpoint:
    async def test_widget_config_valid_key(self, client: AsyncClient):
        """Create a study + active agent, then fetch its widget config."""
        # Create study
        resp = await client.post("/api/studies", json={"title": "Widget Test"})
        study_id = resp.json()["id"]

        # Create an active agent
        agent_resp = await client.post(
            f"/api/studies/{study_id}/agents",
            json={
                "name": "Widget Agent",
                "status": "active",
                "welcome_message": "Welcome!",
                "widget_title": "Survey",
                "widget_description": "Please speak.",
                "widget_primary_color": "#000000",
                "widget_listening_message": "Listening...",
            },
        )
        agent_data = agent_resp.json()
        widget_key = agent_data.get("widget_key")

        assert widget_key is not None

        resp = await client.get(f"/api/widget/{widget_key}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["widget_key"] == widget_key
        assert data["widget_title"] == "Survey"
        assert data["widget_description"] == "Please speak."
        assert data["widget_listening_message"] == "Listening..."
        assert data["welcome_message"] == "Welcome!"
        assert data["audio_available"] is True
        assert data["audio_default_on"] is False
        assert data["voice_response_available"] is True
        assert data["response_mode_default"] == "voice"

    async def test_widget_config_invalid_key(self, client: AsyncClient):
        resp = await client.get("/api/widget/nonexistent-key-xyz")
        assert resp.status_code == 404

    async def test_widget_config_draft_agent_returns_404(self, client: AsyncClient):
        """A draft agent's widget should return 404 (only active agents)."""
        resp = await client.post("/api/studies", json={"title": "Inactive Widget"})
        study_id = resp.json()["id"]

        agent_resp = await client.post(
            f"/api/studies/{study_id}/agents",
            json={"name": "Draft Agent", "status": "draft"},
        )
        widget_key = agent_resp.json().get("widget_key")

        if widget_key:
            resp = await client.get(f"/api/widget/{widget_key}")
            assert resp.status_code == 404

    async def test_widget_config_no_auth_required(self, client: AsyncClient):
        """Widget endpoint should work without auth headers."""
        resp = await client.get("/api/widget/any-key")
        # Should return 404 (not 401 or 403)
        assert resp.status_code == 404

    async def test_completed_session_accepts_separate_optional_feedback(
        self, client: AsyncClient, db_session: AsyncSession
    ):
        study = (await client.post(
            "/api/studies", json={"title": "Feedback study"}
        )).json()
        agent = (await client.post(
            f"/api/studies/{study['id']}/agents",
            json={"name": "Feedback agent", "status": "active"},
        )).json()
        session = Session(
            id=uuid.uuid4(),
            agent_id=uuid.UUID(agent["id"]),
            status=SessionStatus.COMPLETED,
            ended_at=datetime.now(timezone.utc),
        )
        db_session.add(session)
        await db_session.commit()
        token = participant_feedback_token(agent["widget_key"], session.id)
        body = {
            "token": token,
            "naturalness_rating": 4,
            "repeated_questions": False,
            "technical_issues": True,
            "comments": "The microphone stopped once.",
        }

        response = await client.post(
            f"/api/widget/{agent['widget_key']}/sessions/{session.id}/feedback",
            json=body,
        )
        assert response.status_code == 200
        assert response.json() == {"submitted": True, "already_submitted": False}

        await db_session.refresh(session)
        assert session.post_interview_feedback == {
            "survey_version": "participant_experience_v1",
            "naturalness_rating": 4,
            "repeated_questions": False,
            "technical_issues": True,
            "comments": "The microphone stopped once.",
        }
        assert session.post_interview_feedback_submitted_at is not None

        retry = await client.post(
            f"/api/widget/{agent['widget_key']}/sessions/{session.id}/feedback",
            json=body,
        )
        assert retry.status_code == 200
        assert retry.json()["already_submitted"] is True

    async def test_feedback_comments_are_redacted_before_storage(
        self, client: AsyncClient, db_session: AsyncSession
    ):
        study = (await client.post(
            "/api/studies", json={"title": "Private feedback study"}
        )).json()
        agent = (await client.post(
            f"/api/studies/{study['id']}/agents",
            json={"name": "Private feedback agent", "status": "active"},
        )).json()
        session = Session(
            id=uuid.uuid4(),
            agent_id=uuid.UUID(agent["id"]),
            status=SessionStatus.COMPLETED,
            ended_at=datetime.now(timezone.utc),
        )
        db_session.add(session)
        await db_session.commit()
        token = participant_feedback_token(agent["widget_key"], session.id)

        response = await client.post(
            f"/api/widget/{agent['widget_key']}/sessions/{session.id}/feedback",
            json={
                "token": token,
                "naturalness_rating": 4,
                "repeated_questions": False,
                "technical_issues": False,
                "comments": "Contact me at private@example.com",
            },
        )

        assert response.status_code == 200
        await db_session.refresh(session)
        assert session.post_interview_feedback["comments"] == "Contact me at [EMAIL]"

    async def test_feedback_rejects_invalid_token_and_active_session(
        self, client: AsyncClient, db_session: AsyncSession
    ):
        study = (await client.post(
            "/api/studies", json={"title": "Feedback security"}
        )).json()
        agent = (await client.post(
            f"/api/studies/{study['id']}/agents",
            json={"name": "Feedback security agent", "status": "active"},
        )).json()
        session = Session(
            id=uuid.uuid4(),
            agent_id=uuid.UUID(agent["id"]),
            status=SessionStatus.ACTIVE,
        )
        db_session.add(session)
        await db_session.commit()
        url = f"/api/widget/{agent['widget_key']}/sessions/{session.id}/feedback"
        body = {
            "token": "x" * 64,
            "naturalness_rating": 3,
            "repeated_questions": False,
            "technical_issues": False,
        }
        assert (await client.post(url, json=body)).status_code == 403

        body["token"] = participant_feedback_token(agent["widget_key"], session.id)
        assert (await client.post(url, json=body)).status_code == 409

    async def test_widget_config_returns_required_fields(self, client: AsyncClient):
        """Verify all fields needed by the frontend widget are present."""
        resp = await client.post("/api/studies", json={"title": "Field Test"})
        study_id = resp.json()["id"]

        agent_resp = await client.post(
            f"/api/studies/{study_id}/agents",
            json={"name": "Field Agent", "status": "active"},
        )
        widget_key = agent_resp.json().get("widget_key")

        if widget_key:
            resp = await client.get(f"/api/widget/{widget_key}")
            assert resp.status_code == 200
            data = resp.json()

            required_fields = [
                "widget_key",
                "widget_title",
                "widget_description",
                "widget_primary_color",
                "widget_listening_message",
                "participant_id_mode",
                "welcome_message",
                "language",
            ]
            for field in required_fields:
                assert field in data, f"Missing required field: {field}"

    async def test_widget_config_default_colors(self, client: AsyncClient):
        """Widget should have default colors even if not explicitly set."""
        resp = await client.post("/api/studies", json={"title": "Color Test"})
        study_id = resp.json()["id"]

        agent_resp = await client.post(
            f"/api/studies/{study_id}/agents",
            json={"name": "Color Agent", "status": "active"},
        )
        widget_key = agent_resp.json().get("widget_key")

        if widget_key:
            resp = await client.get(f"/api/widget/{widget_key}")
            assert resp.status_code == 200
            data = resp.json()
            assert data["widget_primary_color"]  # Should not be empty

    async def test_widget_speech_uses_agent_tts_without_exposing_credentials(
        self, client: AsyncClient
    ):
        study = (await client.post(
            "/api/studies", json={"title": "Audio prompt test"}
        )).json()
        agent = (await client.post(
            f"/api/studies/{study['id']}/agents",
            json={
                "name": "Text-first audio agent",
                "status": "active",
                "tts_provider": "openai",
                "tts_model": "gpt-4o-mini-tts",
                "tts_voice": "nova",
            },
        )).json()

        with patch(
            "app.providers.tts_http.synthesize_text",
            new_callable=AsyncMock,
            return_value=(b"test-mp3", "audio/mpeg"),
        ) as synthesize:
            response = await client.post(
                f"/api/widget/{agent['widget_key']}/speech",
                json={"text": "What happened next?"},
            )

        assert response.status_code == 200
        assert response.content == b"test-mp3"
        assert response.headers["content-type"].startswith("audio/mpeg")
        synthesize.assert_awaited_once_with(
            provider="openai",
            model="gpt-4o-mini-tts",
            voice="nova",
            text="What happened next?",
        )

    async def test_widget_speech_rejects_oversized_text(self, client: AsyncClient):
        response = await client.post(
            "/api/widget/unused/speech",
            json={"text": "x" * 2001},
        )
        assert response.status_code == 422

    async def test_widget_transcribes_turn_without_storing_audio(
        self, client: AsyncClient
    ):
        study = (await client.post(
            "/api/studies", json={"title": "Turn-based voice test"}
        )).json()
        agent = (await client.post(
            f"/api/studies/{study['id']}/agents",
            json={
                "name": "Hybrid interview agent",
                "status": "active",
                "stt_provider": "openai",
                "stt_model": "gpt-4o-mini-transcribe",
                "language": "en",
            },
        )).json()

        with patch(
            "app.providers.stt_http.transcribe_audio",
            new_callable=AsyncMock,
            return_value="This is my answer.",
        ) as transcribe:
            response = await client.post(
                f"/api/widget/{agent['widget_key']}/transcribe",
                files={"audio": ("answer.webm", b"webm-audio", "audio/webm")},
            )

        assert response.status_code == 200
        assert response.json() == {"text": "This is my answer."}
        transcribe.assert_awaited_once()
        kwargs = transcribe.await_args.kwargs
        assert kwargs["provider"] == "openai"
        assert kwargs["model"] == "gpt-4o-mini-transcribe"
        assert kwargs["audio"] == b"webm-audio"
        assert kwargs["language"] == "en"

    async def test_widget_transcribe_rejects_non_audio(self, client: AsyncClient):
        study = (await client.post(
            "/api/studies", json={"title": "Bad upload test"}
        )).json()
        agent = (await client.post(
            f"/api/studies/{study['id']}/agents",
            json={"name": "Hybrid interview agent", "status": "active"},
        )).json()

        response = await client.post(
            f"/api/widget/{agent['widget_key']}/transcribe",
            files={"audio": ("answer.txt", b"not audio", "text/plain")},
        )
        assert response.status_code == 415

    async def test_widget_exposes_only_participant_facing_protocol(self, client: AsyncClient):
        resp = await client.post(
            "/api/studies",
            json={
                "title": "Consent Test",
                "research_spec": {
                    "version": "0.4",
                    "stage": "pilot",
                    "participant_information": {
                        "purpose_summary": "Understand experiences with AI.",
                        "consent_required": True,
                        "consent_label": "I agree.",
                    },
                    "analysis_notes": "A private hypothesis.",
                },
            },
        )
        study_id = resp.json()["id"]
        agent_resp = await client.post(
            f"/api/studies/{study_id}/agents",
            json={"name": "Consent Agent", "status": "active"},
        )

        config = (
            await client.get(f"/api/widget/{agent_resp.json()['widget_key']}")
        ).json()
        assert config["research_protocol_version"] == "0.4"
        assert config["research_stage"] == "pilot"
        assert config["participant_information"]["consent_label"] == "I agree."
        assert "analysis_notes" not in config
        assert "targets" not in config
