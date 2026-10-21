"""Tests for the independent structured-interview evidence judge."""

from __future__ import annotations

import pytest

from app.api import text_chat
from app.pipeline.evidence_judge import (
    EvidenceJudgeResult,
    EvidenceJudgmentBatch,
    EvidenceJudgmentItem,
    normalize_judgment_batch,
)
from app.pipeline.interview_guide import TextStructuredController, TurnDecision


GUIDE = {
    "orchestration_profile": "semi_structured",
    "questions": [
        {
            "id": "episode",
            "text": "Tell me about one recent task where AI fell short.",
            "evidence_goal": ["Concrete task", "Observable shortfall"],
            "probes": ["What specifically fell short?"],
        },
        {
            "id": "inputs",
            "text": "What did you give the AI to work with?",
            "evidence_goal": [
                "AI product or tool",
                "Instructions provided",
                "Important constraints provided",
            ],
            "probes": ["What instructions did you give it?"],
        },
    ],
}


def _item(
    evidence_id: str,
    status: str,
    quote: str,
    confidence: float,
    missing: list[str] | None = None,
) -> EvidenceJudgmentItem:
    return EvidenceJudgmentItem(
        evidence_id=evidence_id,
        status=status,
        supporting_quote=quote,
        missing_facets=missing or [],
        confidence=confidence,
    )


def test_normalizer_applies_confidence_gates_and_rejects_unknown_ids():
    batch = EvidenceJudgmentBatch(
        judgments=[
            _item("episode:e1", "covered", "a website", 0.96),
            _item(
                "episode:e2",
                "covered",
                "looked generic",
                0.70,
                ["specific visual problem"],
            ),
            _item("inputs:e1", "partial", "Claude", 0.40),
            _item("invented:e1", "covered", "anything", 0.99),
            _item("inputs:e2", "not_supported", "", 0.99),
        ]
    )
    result = normalize_judgment_batch(
        batch, {"episode:e1", "episode:e2", "inputs:e1", "inputs:e2"}
    )

    assert result.resolutions == {
        "episode:e1": "covered",
        "episode:e2": "partial",
    }
    assert result.evidence_sources["episode:e1"] == "a website"
    assert result.missing_facets["episode:e2"] == ["specific visual problem"]
    assert "inputs:e1" not in result.resolutions


def test_external_judge_is_authoritative_over_moderator_coverage_claims():
    controller = TextStructuredController(GUIDE)
    controller.review_turn(
        GUIDE["questions"][0]["text"],
        1,
        TurnDecision(action="main", question=1, intent="episode:main"),
    )
    participant = (
        "I used Claude Design to build a website, but the visual hierarchy "
        "looked generic. I asked for a concise homepage."
    )
    controller.prepare_external_judgment(
        participant,
        3,
        resolutions={
            "episode:e1": "covered",
            "episode:e2": "covered",
            "inputs:e1": "covered",
            "inputs:e2": "covered",
        },
        evidence_sources={
            "episode:e1": "build a website",
            "episode:e2": "the visual hierarchy looked generic",
            "inputs:e1": "Claude Design",
            "inputs:e2": "I asked for a concise homepage",
        },
        missing_facets={},
        confidences={
            "episode:e1": 0.98,
            "episode:e2": 0.97,
            "inputs:e1": 0.99,
            "inputs:e2": 0.96,
        },
        judge_model="gpt-5.6-terra",
    )

    # The moderator incorrectly claims the still-unanswered instructions
    # criterion is covered. The controller must ignore that claim.
    review = controller.review_turn(
        "What instructions did you give it?",
        None,
        TurnDecision(
            action="probe",
            question=2,
            intent="inputs:p1",
            resolved={"inputs:e3": "covered"},
            evidence_sources={"inputs:e3": "a fabricated constraint"},
        ),
        participant,
        participant_sequence=3,
        authoritative_evidence=True,
    )

    assert review.decision is not None
    assert review.decision.question == 2
    assert "constraint" in review.text.casefold()
    assert "what instructions" not in review.text.casefold()
    ledger = controller.evidence_ledger()["questions"]
    assert ledger[0]["coverage_status"] == "complete"
    assert ledger[1]["coverage_status"] == "partial"
    constraints = next(
        item for item in ledger[1]["evidence"] if item["evidence_id"] == "inputs:e3"
    )
    assert constraints["status"] == "missing"
    assert all(
        source.get("evidence_excerpt") != "a fabricated constraint"
        for source in constraints["sources"]
    )


def test_external_judge_rejects_quote_not_in_latest_response():
    controller = TextStructuredController(GUIDE)
    controller.prepare_external_judgment(
        "I tried making a website.",
        3,
        resolutions={"episode:e1": "covered"},
        evidence_sources={"episode:e1": "I analyzed a clinical trial"},
        missing_facets={},
        confidences={"episode:e1": 0.99},
        judge_model="gpt-5.6-terra",
    )

    first = controller.evidence_ledger()["questions"][0]["evidence"][0]
    assert first["status"] == "missing"
    assert controller.stats()["evidence_judgments_rejected"] == 1
    assert controller.stats()["invalid_evidence_source_excerpts"] == 1


def test_partial_judgment_records_missing_facets_and_prior_support():
    controller = TextStructuredController(GUIDE)
    controller.prepare_external_judgment(
        "I used Claude Design.",
        3,
        resolutions={"inputs:e1": "partial"},
        evidence_sources={"inputs:e1": "Claude Design"},
        missing_facets={"inputs:e1": ["model family or version"]},
        confidences={"inputs:e1": 0.91},
        judge_model="gpt-5.6-terra",
    )

    context = controller.evidence_judge_context()
    candidate = next(
        item for item in context["candidates"] if item["evidence_id"] == "inputs:e1"
    )
    assert candidate["prior_status"] == "partial"
    assert candidate["prior_support"] == ["Claude Design"]
    ledger_item = controller.evidence_ledger()["questions"][1]["evidence"][0]
    assert ledger_item["missing_facets"] == ["model family or version"]
    assert ledger_item["judge_confidence"] == 0.91
    assert ledger_item["judged_by"] == "gpt-5.6-terra"


@pytest.mark.asyncio
async def test_judge_failure_stages_empty_authoritative_state(monkeypatch):
    controller = TextStructuredController(GUIDE)

    async def fake_get_key(_field: str) -> str:
        return "present-but-not-used"

    async def fail_judge(**_kwargs):
        raise RuntimeError("synthetic outage")

    monkeypatch.setattr(text_chat, "_get_key", fake_get_key)
    monkeypatch.setattr(
        "app.pipeline.evidence_judge.judge_evidence", fail_judge
    )
    result = await text_chat._judge_structured_turn(
        controller,
        participant_text="I used Claude for it.",
        participant_sequence=3,
    )

    assert result["authoritative"] is True
    assert result["failed"] is True
    assert "AUTHORITATIVE EVIDENCE STATE" in result["control_note"]
    assert controller.stats()["evidence_judge_failures"] == 1
    assert controller.stats()["tracked_evidence_count"] == 0


@pytest.mark.asyncio
async def test_judge_wrapper_stages_successful_batched_result(monkeypatch):
    controller = TextStructuredController(GUIDE)

    async def fake_get_key(_field: str) -> str:
        return "present-but-not-used"

    async def fake_judge(**_kwargs):
        return EvidenceJudgeResult(
            resolutions={"inputs:e1": "covered"},
            evidence_sources={"inputs:e1": "Claude Design"},
            missing_facets={},
            confidences={"inputs:e1": 0.98},
            prompt_tokens=120,
            completion_tokens=30,
            candidate_count=4,
            returned_count=4,
        )

    monkeypatch.setattr(text_chat, "_get_key", fake_get_key)
    monkeypatch.setattr(
        "app.pipeline.evidence_judge.judge_evidence", fake_judge
    )
    result = await text_chat._judge_structured_turn(
        controller,
        participant_text="I used Claude Design.",
        participant_sequence=3,
    )

    assert result["failed"] is False
    assert result["prompt_tokens"] == 120
    assert result["completion_tokens"] == 30
    evidence = controller.evidence_ledger()["questions"][1]["evidence"][0]
    assert evidence["status"] == "covered"
    assert evidence["sources"][0]["evidence_excerpt"] == "Claude Design"
