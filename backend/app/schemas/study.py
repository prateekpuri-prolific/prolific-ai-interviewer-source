"""
OASIS — Pydantic schemas for Study CRUD.
"""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.study import StudyStatus


class ResearchPopulation(BaseModel):
    participant_role: str = Field("", max_length=500)
    subject: str = Field("", max_length=500)
    knowledge_relationship: str = Field("", max_length=1000)
    inclusion_notes: str = Field("", max_length=3000)
    knowledge_limits: list[str] = Field(default_factory=list, max_length=20)
    terminology_notes: str = Field("", max_length=3000)
    accessibility_notes: str = Field("", max_length=3000)
    sensitivity_notes: str = Field("", max_length=3000)


class ResearchTarget(BaseModel):
    id: str = Field(..., min_length=1, max_length=80, pattern=r"^[a-zA-Z0-9_-]+$")
    name: str = Field(..., min_length=1, max_length=250)
    description: str = Field("", max_length=2000)
    evidence_needed: list[str] = Field(default_factory=list, max_length=20)
    do_not_infer: list[str] = Field(default_factory=list, max_length=20)
    required: bool = True


class InterviewerStance(BaseModel):
    organization_label: str = Field("", max_length=300)
    disclosure: str = Field("", max_length=2000)
    boundaries: list[str] = Field(default_factory=list, max_length=20)


class ScopeClarificationPolicy(BaseModel):
    """Participant-facing rules for answering scope and definition questions."""

    participant_definition: str = Field("", max_length=2000)
    in_scope: list[str] = Field(default_factory=list, max_length=30)
    out_of_scope: list[str] = Field(default_factory=list, max_length=30)
    ambiguous_cases: list[str] = Field(default_factory=list, max_length=30)
    ambiguity_handling: str = Field(
        "If it is unclear whether an experience fits, invite the participant to "
        "describe the relevant feature or interaction briefly and treat it as "
        "ambiguous rather than forcing a classification.",
        max_length=2000,
    )


class ParticipantInformation(BaseModel):
    purpose_summary: str = Field("", max_length=3000)
    participation_summary: str = Field("", max_length=3000)
    estimated_minutes: int | None = Field(None, ge=1, le=240)
    ai_disclosure: str = Field(
        "You will be speaking with an AI interviewer.", max_length=2000
    )
    data_handling_summary: str = Field("", max_length=3000)
    data_controller: str = Field("", max_length=1000)
    data_recipients_summary: str = Field("", max_length=3000)
    risks_summary: str = Field("", max_length=3000)
    benefits_summary: str = Field("", max_length=3000)
    compensation_summary: str = Field("", max_length=2000)
    privacy_reminder: str = Field(
        "Please avoid sharing names or other details that identify you or anyone else.",
        max_length=2000,
    )
    rights: list[str] = Field(
        default_factory=lambda: [
            "You may skip any question.",
            "You may stop the interview at any time.",
        ],
        max_length=20,
    )
    consent_required: bool = True
    consent_label: str = Field(
        "I have read this information and agree to take part.", max_length=1000
    )
    researcher_identity: str = Field("", max_length=1000)
    researcher_contact: str = Field("", max_length=1000)
    complaints_contact: str = Field("", max_length=1000)
    ethics_review_reference: str = Field("", max_length=1000)
    data_retention_period: str = Field("", max_length=1000)
    withdrawal_instructions: str = Field("", max_length=2000)
    privacy_notice_url: str = Field("", max_length=2048)


class ResearchFielding(BaseModel):
    """Small, provider-neutral launch plan persisted with the research spec."""

    recruitment_source: Literal["prolific", "direct_link", "mixed"] = "prolific"
    response_limit: int = Field(50, ge=1, le=100000)
    soft_launch_limit: int | None = Field(10, ge=1, le=100000)
    participant_id_parameter: str = Field(
        "PROLIFIC_PID", min_length=1, max_length=80, pattern=r"^[a-zA-Z0-9_-]+$"
    )
    require_profile: bool = False
    completion_url: str = Field("", max_length=2048)
    no_consent_url: str = Field("", max_length=2048)
    inactivity_url: str = Field("", max_length=2048)

    @field_validator("completion_url", "no_consent_url", "inactivity_url")
    @classmethod
    def validate_handoff_url(cls, value: str) -> str:
        cleaned = value.strip()
        if cleaned and not cleaned.startswith("https://"):
            raise ValueError("Recruitment-platform return URLs must use HTTPS")
        return cleaned


class ChatbotScreeningConfig(BaseModel):
    cohort: Literal["ai_nonusers", "older_nondegree", "young_nonusers", "health_nonusers", "health_nonclaude_users", "claude_control"] | None = None
    require_other_chatbot: bool = True
    external_study_id: str = Field(..., pattern=r"^[a-f0-9]{24}$")
    eligible_completion_url: str = Field(..., pattern=r"^https://app\.prolific\.com/submissions/complete\?cc=[A-Za-z0-9]+$")
    other_completion_url: str = Field(..., pattern=r"^https://app\.prolific\.com/submissions/complete\?cc=[A-Za-z0-9]+$")


class ResearchSpec(BaseModel):
    """Versioned study design, independent of any one interview agent."""

    version: str = Field("0.1", min_length=1, max_length=32)
    stage: Literal["pilot", "field"] = "pilot"
    method: Literal[
        "exploration",
        "journey",
        "needs",
        "concept_test",
        "evaluation",
        "standardized_comparison",
        "other",
    ] = "exploration"
    decision_to_inform: str = Field("", max_length=3000)
    research_questions: list[str] = Field(default_factory=list, max_length=20)
    population: ResearchPopulation = Field(default_factory=ResearchPopulation)
    targets: list[ResearchTarget] = Field(default_factory=list, max_length=30)
    clarification_policy: ScopeClarificationPolicy = Field(
        default_factory=ScopeClarificationPolicy
    )
    interviewer_stance: InterviewerStance = Field(default_factory=InterviewerStance)
    participant_information: ParticipantInformation = Field(
        default_factory=ParticipantInformation
    )
    fielding: ResearchFielding = Field(default_factory=ResearchFielding)
    # Persisted for analysis, but deliberately excluded from the live prompt.
    analysis_notes: str = Field("", max_length=10000)
    chatbot_screening: ChatbotScreeningConfig | None = None


# ── Request Schemas ──────────────────────────────────────────


class StudyCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)
    description: str | None = None
    status: StudyStatus = StudyStatus.DRAFT
    research_spec: ResearchSpec | None = None


class StudyUpdate(BaseModel):
    title: str | None = Field(None, min_length=1, max_length=255)
    description: str | None = None
    status: StudyStatus | None = None
    research_spec: ResearchSpec | None = None


# ── Response Schemas ─────────────────────────────────────────


class StudyRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    title: str
    description: str | None
    status: StudyStatus
    research_spec: ResearchSpec | None = None
    created_at: datetime
    updated_at: datetime


class StudyList(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    title: str
    status: StudyStatus
    created_at: datetime
