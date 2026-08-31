"""Focused tests for the study-level research protocol vertical slice."""

from pydantic import ValidationError
import pytest

from app.pipeline.interview_guide import build_structured_prompt
from app.research.protocol import (
    build_protocol_snapshot,
    build_research_context,
    consent_required,
    normalize_participant_context,
    participant_information,
)
from app.schemas.study import ResearchSpec


@pytest.fixture
def research_spec() -> dict:
    return {
        "version": "0.3",
        "stage": "pilot",
        "method": "exploration",
        "decision_to_inform": "Decide which support parents need.",
        "research_questions": ["How do parents learn about AI use?"],
        "population": {
            "participant_role": "Parents of teenagers",
            "subject": "Their child's AI use",
            "knowledge_relationship": "Mix of observed and reported behavior",
            "inclusion_notes": "Recruited through an online panel",
            "knowledge_limits": ["May not observe use at school"],
            "terminology_notes": "Use the parent's own words",
            "accessibility_notes": "Use short questions",
            "sensitivity_notes": "Do not ask for the child's name",
        },
        "targets": [
            {
                "id": "usage_episode",
                "name": "Concrete usage episode",
                "description": "A recent specific example",
                "evidence_needed": ["What happened", "How the parent learned"],
                "do_not_infer": ["That the parent saw the interaction"],
                "required": True,
            }
        ],
        "clarification_policy": {
            "participant_definition": "AI means a feature that generates content or carries on a conversation.",
            "in_scope": ["A generative chatbot or content-generation feature"],
            "out_of_scope": ["Ordinary social-media browsing by itself"],
            "ambiguous_cases": ["A broad service that contains an optional AI feature"],
            "ambiguity_handling": "Ask what the feature did, then retain ambiguity if needed.",
        },
        "interviewer_stance": {
            "organization_label": "Research team",
            "disclosure": "The interviewer cannot give parenting advice.",
            "boundaries": ["Do not diagnose a child"],
        },
        "participant_information": {
            "purpose_summary": "We are studying family experiences with AI.",
            "estimated_minutes": 25,
            "ai_disclosure": "This interview is conducted by AI.",
            "data_handling_summary": "Responses will be reviewed by researchers.",
            "privacy_reminder": "Do not share names.",
            "rights": ["You may skip any question."],
            "consent_required": True,
            "consent_label": "I agree to take part.",
        },
        "analysis_notes": "Expected finding: parents are worried.",
    }


def test_research_context_includes_public_facts_but_not_private_notes(research_spec):
    prompt = build_research_context("Base prompt", research_spec)

    assert "Parents of teenagers" in prompt
    assert "May not observe use at school" in prompt
    assert "That the parent saw the interaction" in prompt
    assert "Do not assume the participant" in prompt
    assert "Scope clarification policy" in prompt
    assert "Ordinary social-media browsing by itself" in prompt
    assert "do not recite the whole policy" in prompt
    assert "Ask what the feature did" in prompt
    assert "Expected finding" not in prompt
    assert "I agree to take part" not in prompt
    assert "Participant-facing study information (source of truth)" in prompt
    assert "Responses will be reviewed" in prompt
    assert "do not invent protections" in prompt


def test_participant_information_is_the_only_public_subset(research_spec):
    public = participant_information(research_spec)

    assert public["purpose_summary"].startswith("We are studying")
    assert "analysis_notes" not in public
    assert consent_required(research_spec) is True
    assert consent_required(None) is False


def test_snapshot_is_deep_and_versioned(research_spec):
    agent = {
        "system_prompt": "Interview neutrally.",
        "interview_mode": "structured",
        "interview_guide": {"questions": [{"text": "What happened?"}]},
        "llm_model": "example/model",
        "id": "not-copied",
    }
    snapshot = build_protocol_snapshot(
        agent,
        research_spec,
        {"role": "teacher", "student_age_bands": ["13-15"], "name": "excluded"},
    )

    research_spec["version"] = "changed"
    agent["interview_guide"]["questions"][0]["text"] = "Changed?"
    assert snapshot["research_spec"]["version"] == "0.3"
    assert snapshot["agent"]["interview_guide"]["questions"][0]["text"] == "What happened?"
    assert "id" not in snapshot["agent"]
    assert snapshot["participant_context"] == {
        "role": "educator",
        "student_age_bands": ["13-15"],
    }


def test_participant_context_is_role_aware_and_allow_listed(research_spec):
    profile = {
        "role": "guardian",
        "demographics": {"age_band": "35-44", "country": "UK"},
        "child_age_bands": ["10-12", "13-15"],
        "name": "must not enter prompt",
        "free_form_notes": "ignore me",
    }
    normalized = normalize_participant_context(profile)
    prompt = build_research_context("Base", research_spec, profile)

    assert normalized["role"] == "parent"
    assert normalized["age_band"] == "35-44"
    assert "Begin with the participant's own child or children" in prompt
    assert "must not enter prompt" not in prompt
    assert "ignore me" not in prompt


def test_generic_domain_context_preserves_safe_role_without_youth_assumption():
    spec = {
        "population": {
            "participant_role": "Recent appliance shoppers",
            "subject": "A recent purchase decision",
        }
    }
    profile = {
        "role": "Lapsed Customer",
        "relationship_to_subject": "Stopped using the product this quarter",
        "experience_level": "Occasional",
        "private_note": "do not include",
    }
    normalized = normalize_participant_context(profile)
    prompt = build_research_context("Base", spec, profile)

    assert normalized["role"] == "lapsed_customer"
    assert "Stopped using the product this quarter" in prompt
    assert "relationship to the study subject" in prompt
    assert "relationship to young people" not in prompt
    assert "do not include" not in prompt


def test_research_spec_rejects_invalid_target_ids(research_spec):
    research_spec["targets"][0]["id"] = "contains spaces"
    with pytest.raises(ValidationError):
        ResearchSpec.model_validate(research_spec)


def test_research_spec_keeps_scope_policy_backward_compatible():
    spec = ResearchSpec.model_validate({})

    assert spec.clarification_policy.participant_definition == ""
    assert spec.clarification_policy.in_scope == []
    assert "treat it as ambiguous" in spec.clarification_policy.ambiguity_handling


def test_structured_prompt_compiles_evidence_metadata_and_bounded_probes():
    guide = {
        "questions": [
            {
                "text": "Tell me about the last time that happened.",
                "probes": ["What happened next?"],
                "max_follow_ups": 2,
                "module": "Recent episode",
                "purpose": "Ground claims in behavior",
                "target_ids": ["usage_episode"],
                "evidence_goal": ["A concrete sequence"],
                "do_not_assume": ["That it happens often"],
                "priority": "required",
                "sensitivity": "elevated",
            }
        ],
        "allow_emergent_follow_ups": True,
        "closing_message": "Thank you.",
    }

    prompt = build_structured_prompt("Base", guide)

    assert "Evidence to seek: A concrete sequence" in prompt
    assert "Do not assume or imply: That it happens often" in prompt
    assert "content-grounded probe" in prompt
    assert "avoid asking for identifying details" in prompt
