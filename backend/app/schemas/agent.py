"""
OASIS — Pydantic schemas for Agent CRUD.
"""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.models.agent import AgentModality, AgentStatus, InterviewMode, ParticipantIdMode, PipelineType


# ── Structured Interview Guide Sub-Schemas ───────────────────


class InterviewQuestion(BaseModel):
    """A single question in the structured interview guide."""
    text: str = Field(..., min_length=1, description="The main question text")
    probes: list[str] = Field(
        default_factory=list,
        description="Example follow-up probes for deeper exploration",
    )
    max_follow_ups: int = Field(
        3,
        ge=0,
        le=10,
        description="Maximum follow-up exchanges before moving on",
    )
    transition: str | None = Field(
        None,
        description="Optional transition text when moving to the next question",
    )
    id: str | None = Field(None, max_length=80, pattern=r"^$|^[a-zA-Z0-9_-]+$")
    module: str | None = Field(None, max_length=250)
    purpose: str | None = Field(None, max_length=1500)
    target_ids: list[str] = Field(default_factory=list, max_length=20)
    evidence_goal: list[str] = Field(default_factory=list, max_length=20)
    do_not_assume: list[str] = Field(default_factory=list, max_length=20)
    priority: Literal["required", "optional"] = "required"
    sensitivity: Literal["standard", "elevated"] = "standard"
    phase: Literal["warm_up", "core", "closing"] = Field(
        "core",
        description="Interview phase used to pace rapport before substantive topics.",
    )
    visit_scope: Literal["all", "first_time", "returning"] = Field(
        "all",
        description="Whether this question is used for all, first-time, or returning participants.",
    )
    skip_remaining_module_if_not_applicable: bool = Field(
        False,
        description=(
            "When true, an explicit none/not-applicable response skips the "
            "remaining consecutive questions in this question's module."
        ),
    )
    always_ask: bool = Field(
        False,
        description=(
            "Protect a task or module anchor from evidence-based skipping, "
            "even when its evidence was supplied earlier."
        ),
    )
    depends_on: list[str] = Field(
        default_factory=list,
        max_length=20,
        description=(
            "Stable question IDs that must be reached before this card can "
            "be asked. Use this for modules that depend on an earlier anchor."
        ),
    )
    role_variants: dict[str, str | dict] = Field(
        default_factory=dict,
        description=(
            "Optional role-specific wording keyed by participant role. Values may "
            "be a question string or an object containing text and probes."
        ),
    )
    returning_role_variants: dict[str, str | dict] = Field(
        default_factory=dict,
        description="Optional role-specific wording for returning participants.",
    )


class InterviewGuide(BaseModel):
    """Structured interview protocol — a list of questions with probes."""
    questions: list[InterviewQuestion] = Field(
        ...,
        min_length=1,
        description="Ordered list of interview questions",
    )
    orchestration_profile: Literal[
        "fixed", "repair_only", "semi_structured", "exploratory"
    ] | None = Field(
        None,
        description=(
            "Interview-control policy. Null preserves legacy guides and is "
            "resolved deterministically at runtime."
        ),
    )
    closing_message: str | None = Field(
        "Thank you for your time. This concludes our interview.",
        description="Message spoken after the last question is complete",
    )
    allow_follow_ups: bool = Field(
        True,
        description=(
            "Whether the interviewer may ask follow-up questions. When false, "
            "the interviewer asks only the predefined main questions in order; "
            "per-question probes and follow-up caps are ignored."
        ),
    )
    skip_answered_questions: bool | None = Field(
        None,
        description=(
            "Whether questions whose full evidence goal was supplied earlier "
            "may be skipped. Null preserves the legacy behavior: enabled for "
            "semi-structured guides and disabled for fixed surveys."
        ),
    )
    probe_selection: Literal["ordered", "relevance"] = Field(
        "ordered",
        description=(
            "How follow-up probes are chosen. 'ordered' asks the probes in "
            "list order (standardized/reproducible across participants). "
            "'relevance' lets the agent pick the most relevant probe from the "
            "list for each answer (more adaptive). The per-question "
            "max_follow_ups cap applies either way."
        ),
    )
    allow_emergent_follow_ups: bool = Field(
        False,
        description=(
            "Allow one content-grounded, neutral probe when listed probes do "
            "not address an evidence gap. The max_follow_ups cap still applies."
        ),
    )

    @model_validator(mode="after")
    def _validate_protocol_structure(self) -> "InterviewGuide":
        ids = [question.id for question in self.questions if question.id]
        duplicates = sorted({question_id for question_id in ids if ids.count(question_id) > 1})
        # Legacy persisted guides may contain duplicate IDs. They remain
        # readable so the composer can migrate them; every explicit-profile
        # guide is held to the new stable-ID contract.
        if duplicates and self.orchestration_profile:
            raise ValueError(
                "Question IDs must be unique; duplicates: " + ", ".join(duplicates)
            )

        positions = {question.id: index for index, question in enumerate(self.questions) if question.id}
        for index, question in enumerate(self.questions):
            if question.depends_on and not question.id:
                raise ValueError("A question with dependencies must have a stable ID.")
            for dependency in question.depends_on:
                if dependency not in positions:
                    raise ValueError(
                        f"Question {question.id!r} depends on unknown question ID {dependency!r}."
                    )
                if positions[dependency] >= index:
                    raise ValueError(
                        f"Question {question.id!r} must depend only on an earlier question."
                    )

        profile_rules = {
            "fixed": (False, False, False),
            "repair_only": (False, True, False),
            "semi_structured": (True, True, False),
            "exploratory": (True, True, True),
        }
        if self.orchestration_profile:
            expected = profile_rules[self.orchestration_profile]
            configured = (
                self.allow_follow_ups,
                self.skip_answered_questions,
                self.allow_emergent_follow_ups,
            )
            field_names = (
                "allow_follow_ups",
                "skip_answered_questions",
                "allow_emergent_follow_ups",
            )
            for field_name, actual, required in zip(field_names, configured, expected):
                if field_name in self.model_fields_set and actual is not None and actual != required:
                    raise ValueError(
                        f"orchestration_profile={self.orchestration_profile!r} "
                        f"requires {field_name}={required}."
                    )
            self.allow_follow_ups = expected[0]
            self.skip_answered_questions = expected[1]
            self.allow_emergent_follow_ups = expected[2]

            if self.orchestration_profile in {"semi_structured", "exploratory"}:
                has_capacity = any(
                    question.max_follow_ups > 0
                    and (question.probes or self.orchestration_profile == "exploratory")
                    for question in self.questions
                )
                if not has_capacity:
                    raise ValueError(
                        f"orchestration_profile={self.orchestration_profile!r} "
                        "requires at least one question with follow-up capacity."
                    )
        return self


# ── Engagement config sub-schemas ────────────────────────────


class EngagementWeights(BaseModel):
    length: float = Field(0.35, ge=0, le=1)
    latency: float = Field(0.25, ge=0, le=1)
    rate: float = Field(0.15, ge=0, le=1)
    fillers: float = Field(0.15, ge=0, le=1)
    energy: float = Field(0.10, ge=0, le=1)


class EngagementConfig(BaseModel):
    """Per-agent tuning for the engagement scorer and event detector."""
    window_size: int = Field(3, ge=2, le=10, description="Turns for streak/event detection")
    low_threshold: float = Field(0.34, ge=0, le=1)
    high_threshold: float = Field(0.67, ge=0, le=1)
    long_latency_ms: int = Field(4000, ge=500, le=30000)
    short_answer_words: int = Field(3, ge=1, le=20)
    weights: EngagementWeights = Field(default_factory=EngagementWeights)


# ── Adaptive behavior sub-schemas ────────────────────────────

ADAPTIVE_TRIGGERS = (
    "sustained_disengagement",
    "positive_engagement_streak",
    "recovery_after_dip",
    "long_latency",
    "very_short_answer",
    "high_filler",
)

ADAPTIVE_ACTIONS = (
    "offer_break",
    "soften_next_probe",
    "encourage_elaboration",
    "acknowledge_effort",
    "privacy_check",
    "slow_down",
    "reset_pace",
)


class AdaptiveRule(BaseModel):
    on: str = Field(..., description="Trigger event or per-turn flag")
    action: str = Field(..., description="Curated action id")
    custom_instruction: str | None = Field(None, max_length=2000)
    cooldown_seconds: int = Field(0, ge=0, le=3600)
    params: dict = Field(default_factory=dict)

    @field_validator("on")
    @classmethod
    def _valid_trigger(cls, v: str) -> str:
        if v not in ADAPTIVE_TRIGGERS:
            raise ValueError(f"unknown trigger: {v}")
        return v

    @field_validator("action")
    @classmethod
    def _valid_action(cls, v: str) -> str:
        if v not in ADAPTIVE_ACTIONS:
            raise ValueError(f"unknown action: {v}")
        return v


class AdaptivePolicy(BaseModel):
    """Per-agent adaptive behavior policy. Defaults to safe shadow mode."""
    mode: str = Field("shadow", pattern="^(shadow|live)$")
    rules: list[AdaptiveRule] = Field(default_factory=list)


# ── Request Schemas ──────────────────────────────────────────


class AgentCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    modality: AgentModality = AgentModality.VOICE
    avatar: str | None = Field("neutral", max_length=50)
    system_prompt: str = ""
    welcome_message: str | None = "Hello, thank you for participating in this study."

    pipeline_type: PipelineType = PipelineType.MODULAR

    llm_model: str = Field("openai/gpt-5.6-luna", max_length=255)

    stt_provider: str = Field("openai", max_length=100)
    stt_model: str | None = "gpt-realtime-whisper"

    tts_provider: str = Field("openai", max_length=100)
    tts_model: str | None = "gpt-4o-mini-tts"
    tts_voice: str | None = "alloy"

    turn_detection: str = Field(
        "local",
        pattern="^(local|remote)$",
        description=(
            "Turn-detection model for modular voice. 'local' runs the bundled "
            "on-device smart-turn model; 'remote' uses the configured HTTP "
            "smart-turn endpoint (falls back to local if none is set)."
        ),
    )

    language: str = Field("en", max_length=10)
    max_duration_seconds: int | None = Field(None, ge=60, le=7200)

    status: AgentStatus = AgentStatus.ACTIVE

    # Participant ID
    participant_id_mode: ParticipantIdMode = ParticipantIdMode.RANDOM

    # Widget customisation
    widget_title: str | None = None
    widget_description: str | None = None
    widget_primary_color: str | None = "#111827"
    widget_listening_message: str | None = "Agent is listening…"
    widget_show_progress: bool = False

    # Interview mode
    interview_mode: InterviewMode = InterviewMode.FREE_FORM
    interview_guide: InterviewGuide | None = None

    # Silence handling
    silence_timeout_seconds: int | None = None
    silence_prompt: str | None = None

    # Telephony
    twilio_phone_number: str | None = None

    # Audio recording (voice web interviews; per-agent opt-in)
    store_audio: bool = False

    # Engagement metrics (voice web interviews; per-agent opt-in, observational)
    track_engagement: bool = False
    engagement_config: EngagementConfig | None = None

    # Adaptive behavior (requires engagement; defaults to shadow mode)
    adaptive_enabled: bool = False
    adaptive_policy: AdaptivePolicy | None = None

    @model_validator(mode="after")
    def _text_requires_modular(self) -> "AgentCreate":
        if self.modality == AgentModality.TEXT and self.pipeline_type != PipelineType.MODULAR:
            raise ValueError("Text agents must use the modular pipeline.")
        return self


class AgentUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=255)
    modality: AgentModality | None = None
    avatar: str | None = None
    system_prompt: str | None = None
    welcome_message: str | None = None

    pipeline_type: PipelineType | None = None

    llm_model: str | None = Field(None, max_length=255)

    stt_provider: str | None = Field(None, max_length=100)
    stt_model: str | None = None

    tts_provider: str | None = Field(None, max_length=100)
    tts_model: str | None = None
    tts_voice: str | None = None

    turn_detection: str | None = Field(None, pattern="^(local|remote)$")

    language: str | None = Field(None, max_length=10)
    max_duration_seconds: int | None = Field(None, ge=60, le=7200)

    status: AgentStatus | None = None

    participant_id_mode: ParticipantIdMode | None = None

    widget_title: str | None = None
    widget_description: str | None = None
    widget_primary_color: str | None = None
    widget_listening_message: str | None = None
    widget_show_progress: bool | None = None

    # Interview mode
    interview_mode: InterviewMode | None = None
    interview_guide: InterviewGuide | None = None

    silence_timeout_seconds: int | None = None
    silence_prompt: str | None = None

    twilio_phone_number: str | None = None

    store_audio: bool | None = None

    track_engagement: bool | None = None
    engagement_config: EngagementConfig | None = None
    adaptive_enabled: bool | None = None
    adaptive_policy: AdaptivePolicy | None = None


# ── Response Schemas ─────────────────────────────────────────


class AgentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    study_id: UUID
    name: str
    modality: AgentModality
    avatar: str | None
    status: AgentStatus
    system_prompt: str
    welcome_message: str | None

    pipeline_type: PipelineType

    llm_model: str

    stt_provider: str
    stt_model: str | None

    tts_provider: str
    tts_model: str | None
    tts_voice: str | None

    turn_detection: str

    language: str
    max_duration_seconds: int | None

    participant_id_mode: ParticipantIdMode

    widget_key: str
    widget_title: str | None
    widget_description: str | None
    widget_primary_color: str | None
    widget_listening_message: str | None
    widget_show_progress: bool

    interview_mode: InterviewMode
    interview_guide: dict | None

    silence_timeout_seconds: int | None
    silence_prompt: str | None

    twilio_phone_number: str | None

    store_audio: bool

    track_engagement: bool
    engagement_config: dict | None
    adaptive_enabled: bool
    adaptive_policy: dict | None

    created_at: datetime
    updated_at: datetime


class AgentList(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    study_id: UUID
    name: str
    modality: AgentModality
    avatar: str | None
    status: AgentStatus
    pipeline_type: PipelineType
    llm_model: str
    stt_provider: str
    stt_model: str | None
    tts_provider: str
    tts_model: str | None
    tts_voice: str | None
    language: str
    widget_key: str
    participant_id_mode: ParticipantIdMode
    interview_mode: InterviewMode
    created_at: datetime
