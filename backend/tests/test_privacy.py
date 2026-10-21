"""Privacy redaction unit and persistence-boundary tests."""

import uuid
from collections import Counter
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import select

from app.models.session import Session, SessionStatus, SpeakerRole, TranscriptEntry
from app.privacy import (
    PRIVACY_WARNING,
    REDACTOR_VERSION,
    queue_privacy_warning_audio,
    redact_json,
    redact_session_data,
    redact_text,
    redact_text_with_llm,
)


@pytest.mark.parametrize(
    ("label", "value", "secret", "placeholder"),
    [
        ("email", "Email alex [at] example dot com", "alex [at] example dot com", "[EMAIL]"),
        ("split email first turn", "It starts with alex@", "alex@", "[EMAIL_FRAGMENT]"),
        ("split email second turn", "example.com is the rest", "example.com", "[LINK]"),
        ("phone", "Call +1 (212) 555-0199", "+1 (212) 555-0199", "[PHONE_OR_ID]"),
        ("spoken phone", "Call two one two five five five zero one nine nine", "two one two five five five zero one nine nine", "[PHONE_OR_ID]"),
        ("address", "I live at 123 maple street", "123 maple street", "[ADDRESS]"),
        ("ssn", "My SSN is 123-45-6789", "123-45-6789", "[GOVERNMENT_ID]"),
        ("username", "My username is @private_person", "@private_person", "[USERNAME]"),
        ("name", "I spoke with Alice Johnson yesterday", "Alice Johnson", "[PERSON]"),
        ("unicode name", "I spoke with José Álvarez yesterday", "José Álvarez", "[PERSON]"),
        ("third party", "My therapist named Alice called", "Alice", "[PERSON]"),
        ("employer", "I work at Acme Research Labs", "Acme Research Labs", "[ORGANIZATION]"),
        ("uuid", "Device id 550e8400-e29b-41d4-a716-446655440000", "550e8400-e29b-41d4-a716-446655440000", "[ID]"),
        ("date", "My birthday is 14 April 1990", "14 April 1990", "[DATE]"),
        ("credential", "Key sk-proj-" + "abcdefghijklmnopqrstuvwxyz123456", "sk-proj-" + "abcdefghijklmnopqrstuvwxyz123456", "[CREDENTIAL]"),
        ("coordinates", "Meet at 40.7128, -74.0060", "40.7128, -74.0060", "[COORDINATES]"),
        ("UK postcode", "My postcode is SW1A 1AA", "SW1A 1AA", "[POSTAL_CODE]"),
    ],
)
def test_redacts_direct_and_contextual_identifiers(label, value, secret, placeholder):
    result = redact_text(value)
    assert secret not in result.text, label
    assert placeholder in result.text, label
    assert result.detected


def test_does_not_redact_common_product_names_or_approximate_time():
    value = "I compared ChatGPT with Microsoft Copilot about three months ago."
    assert redact_text(value).text == value


@pytest.mark.parametrize("text", [
    "The patient approach is to wait and listen.",
    "Good record keeping matters.",
    "The device operates normally.",
    "An employee handbook explains the process.",
    "This case illustrates the problem.",
])
def test_identifier_rule_does_not_treat_ordinary_following_words_as_ids(text):
    assert redact_text(text).text == text


@pytest.mark.parametrize("text,secret", [
    ("My patient ID is ABCDE12345", "ABCDE12345"),
    ("Patient ID ABCDE", "ABCDE"),
    ("Account number 12345678", "12345678"),
    ("Member no. QWERTY", "QWERTY"),
    ("Device: ABCDE", "ABCDE"),
    ("Record AB123CD", "AB123CD"),
])
def test_identifier_rule_still_removes_explicit_and_digit_bearing_ids(text, secret):
    assert secret not in redact_text(text).text


async def test_missing_contextual_key_withholds_instead_of_silently_downgrading():
    result = await redact_text_with_llm("A nuanced synthetic disclosure", api_key="")
    assert result.text == "[PRIVACY_CHECK_UNAVAILABLE]"
    assert result.counts == {"llm_check_unavailable": 1}


@pytest.mark.parametrize(
    ("value", "category", "placeholder"),
    [
        (
            "My daughter was diagnosed with depression and is receiving treatment.",
            "special_category",
            "[SENSITIVE_CONTENT_REMOVED]",
        ),
        (
            "A patient told me they were self-harming after an assault.",
            "special_category",
            "[SENSITIVE_CONTENT_REMOVED]",
        ),
        (
            "I was diagnosed with anxiety last year.",
            "special_category",
            "[SENSITIVE_CONTENT_REMOVED]",
        ),
        (
            "I'm the only pediatric neurosurgeon in a very small island community.",
            "indirect_identifier",
            "[POTENTIALLY_IDENTIFYING_CONTEXT_REMOVED]",
        ),
        (
            "I was featured in the local news for the incident.",
            "indirect_identifier",
            "[POTENTIALLY_IDENTIFYING_CONTEXT_REMOVED]",
        ),
    ],
)
def test_removes_special_category_and_indirectly_identifying_utterances(
    value, category, placeholder
):
    result = redact_text(value)
    assert result.text == placeholder
    assert result.counts[category] == 1


def test_allows_general_non_identifying_health_patterns():
    value = "Clinicians are seeing more questions about anxiety in young people."
    assert redact_text(value).text == value


def _privacy_response(
    decision: str,
    reason_code: str,
    confidence: float = 0.95,
):
    response = MagicMock()
    response.output_text = (
        '{"decision":"%s","confidence":%s,"reason_code":"%s"}'
        % (decision, confidence, reason_code)
    )
    return response


async def test_contextual_llm_flags_nuanced_combined_identifiers():
    client = MagicMock()
    client.responses.create = AsyncMock(
        return_value=_privacy_response(
            "indirect_identifier", "combined_quasi_identifiers"
        )
    )
    with patch("openai.AsyncOpenAI", return_value=client):
        result = await redact_text_with_llm(
            "She is 17, attends the tiny island's only academy, and won its robotics final.",
            api_key="sk-test",
        )

    assert result.text == "[POTENTIALLY_IDENTIFYING_CONTEXT_REMOVED]"
    assert result.counts["llm_indirect_identifier"] == 1


async def test_contextual_llm_sees_locally_minimised_text_only():
    client = MagicMock()
    client.responses.create = AsyncMock(
        return_value=_privacy_response("allow", "none")
    )
    secret = "person@example.com"
    with patch("openai.AsyncOpenAI", return_value=client):
        result = await redact_text_with_llm(
            f"Contact me at {secret}; otherwise the experience was ordinary.",
            api_key="sk-test",
        )

    assert secret not in client.responses.create.await_args.kwargs["input"]
    assert "[EMAIL]" in client.responses.create.await_args.kwargs["input"]
    assert secret not in result.text
    assert result.counts["email"] == 1
    assert client.responses.create.await_args.kwargs["store"] is False


async def test_contextual_llm_allows_general_pattern_discussion():
    client = MagicMock()
    client.responses.create = AsyncMock(
        return_value=_privacy_response("allow", "none")
    )
    value = "Teachers are hearing more general questions about anxiety this year."
    with patch("openai.AsyncOpenAI", return_value=client):
        result = await redact_text_with_llm(value, api_key="sk-test")

    assert result.text == value
    assert result.count == 0


async def test_contextual_llm_failure_withholds_utterance():
    client = MagicMock()
    client.responses.create = AsyncMock(side_effect=TimeoutError("timed out"))
    with patch("openai.AsyncOpenAI", return_value=client):
        result = await redact_text_with_llm(
            "This otherwise ordinary answer could not be checked.",
            api_key="sk-test",
        )

    assert result.text == "[PRIVACY_CHECK_UNAVAILABLE]"
    assert result.counts["llm_check_unavailable"] == 1


async def test_voice_privacy_warning_is_queued_as_standalone_audio():
    task = type("Task", (), {"queue_frame": AsyncMock()})()

    queued = await queue_privacy_warning_audio(
        task,
        {"type": "privacy_warning", "text": PRIVACY_WARNING},
    )

    assert queued is True
    frame = task.queue_frame.await_args.args[0]
    assert frame.text == PRIVACY_WARNING
    assert frame.append_to_context is False


def test_redaction_is_idempotent_and_does_not_reveal_matches():
    first = redact_text("Email me at person@example.com or call 212-555-0199")
    second = redact_text(first.text)
    assert second.text == first.text
    assert second.count == 0
    assert "person@example.com" not in repr(first)


def test_recursively_redacts_derived_json_text():
    value = {"evidence": ["Alice said to email person@example.com"], "score": 4}
    safe, counts = redact_json(value)
    assert "person@example.com" not in safe["evidence"][0]
    assert safe["score"] == 4
    assert counts["email"] == 1


async def test_completion_sweep_redacts_transcript_metadata_and_feedback(
    db_session, make_study, make_agent
):
    study = make_study()
    agent = make_agent(study_id=study.id)
    session = Session(
        id=uuid.uuid4(),
        agent_id=agent.id,
        status=SessionStatus.COMPLETED,
        completion_metadata={"protocol": {"evidence": "Call 212-555-0199"}},
        post_interview_feedback={"comments": "Email me at person@example.com"},
    )
    entry = TranscriptEntry(
        id=uuid.uuid4(),
        session_id=session.id,
        role=SpeakerRole.USER,
        content="I spoke with Alice Johnson.",
        sequence=1,
        spoken_at=datetime.now(timezone.utc),
    )
    db_session.add_all([study, agent, session, entry])
    await db_session.flush()

    result = await redact_session_data(
        db_session,
        session.id,
        session=session,
        ingress_counts=Counter({"email": 2}),
    )
    await db_session.flush()

    stored = await db_session.scalar(
        select(TranscriptEntry).where(TranscriptEntry.id == entry.id)
    )
    assert "Alice Johnson" not in stored.content
    assert "212-555-0199" not in session.completion_metadata["protocol"]["evidence"]
    assert "person@example.com" not in session.post_interview_feedback["comments"]
    privacy = session.completion_metadata["privacy"]
    assert privacy["redactor_version"] == REDACTOR_VERSION
    assert privacy["ingress_redaction_count"] == 2
    assert privacy["completion_check_passed"] is True
    assert result["count"] == 3
