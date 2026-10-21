"""
Tests for the TranscriptLogger / TranscriptUserCapture FrameProcessors.

Mocks the database session to avoid needing a real DB.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pipecat.frames.frames import LLMFullResponseEndFrame, TextFrame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection

from app.pipeline.transcript_logger import (
    TranscriptLoggerState,
    TranscriptUserCapture,
    TranscriptLogger,
)


class TestTranscriptLogger:
    def _make_state(self, notify_callback=None):
        """Create a TranscriptLoggerState with a mocked DB session factory."""
        mock_session = AsyncMock()
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()

        mock_factory = MagicMock()
        mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_factory.return_value.__aexit__ = AsyncMock(return_value=False)

        state = TranscriptLoggerState(
            session_id=uuid.uuid4(),
            db_session_factory=mock_factory,
            notify_callback=notify_callback,
        )
        return state, mock_session

    def _make_logger(self, notify_callback=None):
        """Create a TranscriptLogger with shared state and a mocked DB."""
        state, mock_session = self._make_state(notify_callback)
        logger = TranscriptLogger(state=state)
        return logger, state, mock_session

    async def test_persist_entry_user(self):
        state, mock_session = self._make_state()
        await state.persist_entry("user", "Hello, world!")
        await state.drain()  # DB write happens in the background
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()

    async def test_persist_entry_agent(self):
        state, mock_session = self._make_state()
        await state.persist_entry("agent", "Agent response")
        await state.drain()
        mock_session.add.assert_called_once()

    async def test_persist_entry_empty_skipped(self):
        state, mock_session = self._make_state()
        await state.persist_entry("user", "   ")
        mock_session.add.assert_not_called()

    async def test_sequence_increments(self):
        state, _ = self._make_state()
        await state.persist_entry("user", "First")
        assert state.sequence == 1
        await state.persist_entry("agent", "Second")
        assert state.sequence == 2

    async def test_agent_buffer_accumulates(self):
        state, _ = self._make_state()
        state.agent_buffer.append("Hello ")
        state.agent_buffer.append("world!")
        assert "".join(state.agent_buffer) == "Hello world!"

    async def test_flush_agent_buffer(self):
        state, mock_session = self._make_state()
        state.agent_buffer = ["Hello ", "world!"]
        await state.flush_agent_buffer()
        await state.drain()
        mock_session.add.assert_called_once()
        assert state.agent_buffer == []

    async def test_flush_empty_buffer(self):
        state, mock_session = self._make_state()
        await state.flush_agent_buffer()
        mock_session.add.assert_not_called()

    async def test_notify_callback_called(self):
        callback = AsyncMock()
        state, _ = self._make_state(notify_callback=callback)
        await state.persist_entry("user", "Test message")
        await state.drain()
        callback.assert_called_once()
        payload = callback.call_args[0][0]
        assert payload["type"] == "transcript"
        assert payload["role"] == "user"
        assert payload["content"] == "Test message"

    async def test_notify_callback_error_handled(self):
        """Even if the callback raises, it shouldn't crash the logger."""
        callback = AsyncMock(side_effect=Exception("Network error"))
        state, _ = self._make_state(notify_callback=callback)
        # Should not raise
        await state.persist_entry("user", "Test")
        await state.drain()

    async def test_user_pii_is_redacted_before_storage_notification_and_forwarding(self):
        callback = AsyncMock()
        state, mock_session = self._make_state(notify_callback=callback)
        capture = TranscriptUserCapture(state=state)
        capture.push_frame = AsyncMock()
        secret = "person@example.com"

        await capture.process_frame(
            TranscriptionFrame(
                text=f"Please email {secret}", user_id="u", timestamp="t",
                result={"transcript": secret},
            ),
            FrameDirection.DOWNSTREAM,
        )
        await state.drain()

        stored = mock_session.add.call_args[0][0]
        forwarded = capture.push_frame.call_args[0][0]
        assert secret not in stored.content
        assert secret not in forwarded.text
        assert forwarded.result is None
        assert any(
            call.args[0].get("type") == "privacy_warning"
            for call in callback.await_args_list
        )
        assert all(secret not in str(call.args[0]) for call in callback.await_args_list)

    async def test_cleanup_flushes_buffer(self):
        logger, state, mock_session = self._make_logger()
        state.agent_buffer = ["Pending text"]
        await logger.cleanup()
        mock_session.add.assert_called_once()

    async def test_llm_response_end_flushes_for_participant_display(self):
        logger, state, mock_session = self._make_logger()
        logger.push_frame = AsyncMock()

        await logger.process_frame(TextFrame(text="What happened "), FrameDirection.DOWNSTREAM)
        await logger.process_frame(TextFrame(text="next?"), FrameDirection.DOWNSTREAM)
        await logger.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)
        await state.drain()

        assert state.agent_buffer == []
        entry = mock_session.add.call_args[0][0]
        assert entry.content == "What happened next?"
