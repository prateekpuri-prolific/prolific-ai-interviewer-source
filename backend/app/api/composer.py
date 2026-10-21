"""AI-assisted, researcher-controlled interview guide composition."""

from __future__ import annotations

import json
import re
from typing import Any, Literal, TypeVar
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.text_chat import _get_key, _openai_api_base
from app.database import get_db
from app.models.agent import Agent
from app.models.study import Study
from app.pipeline.interview_guide import interview_guide_quality_warnings
from app.schemas.agent import InterviewGuide, InterviewQuestion
from app.schemas.study import ResearchSpec

router = APIRouter(prefix="/studies/{study_id}/composer", tags=["composer"])

# Guide authoring is a low-volume, high-judgment workload. Keep it on the
# frontier model independently of the lower-cost model chosen for live
# participant interviews.
_COMPOSER_MODEL = "gpt-5.6-sol"
_ComposerModel = TypeVar("_ComposerModel", bound=BaseModel)


class ComposerTurn(BaseModel):
    role: Literal["researcher", "composer"]
    text: str = Field(..., min_length=1, max_length=2000)


class ComposerEditChange(BaseModel):
    path: str = Field(..., min_length=1, max_length=500)
    before: Any = None
    after: Any = None


class ComposerEditEvent(BaseModel):
    id: str = Field(..., min_length=1, max_length=100)
    timestamp: str = Field(..., min_length=1, max_length=100)
    source: Literal[
        "manual", "study_composer", "question_composer", "undo", "redo"
    ]
    summary: str | None = Field(None, max_length=1000)
    changes: list[ComposerEditChange] = Field(default_factory=list, max_length=12)


class ComposeRequest(BaseModel):
    instruction: str = Field(..., min_length=2, max_length=12000)
    agent_id: UUID | None = None
    current_research_spec: ResearchSpec | None = None
    current_interview_guide: InterviewGuide | None = None
    current_system_prompt: str | None = Field(None, max_length=50000)
    current_welcome_message: str | None = Field(None, max_length=5000)
    conversation_history: list[ComposerTurn] = Field(default_factory=list, max_length=12)
    selected_question_index: int | None = Field(None, ge=0, le=100)
    selected_question_id: str | None = Field(None, max_length=255)
    edited_question_ids: list[str] = Field(default_factory=list, max_length=100)
    edit_history: list[ComposerEditEvent] = Field(default_factory=list, max_length=50)


class ComposerDraft(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)
    summary: str = Field(..., min_length=1, max_length=2000)
    estimated_minutes: int = Field(..., ge=3, le=120)
    research_spec: ResearchSpec
    system_prompt: str = Field(..., min_length=1, max_length=50000)
    welcome_message: str = Field(..., min_length=1, max_length=5000)
    interview_guide: InterviewGuide
    warnings: list[str] = Field(default_factory=list, max_length=20)


class ComposerSourceNote(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    url: str = Field(..., min_length=1, max_length=2000)
    relevant_finding: str = Field(..., min_length=1, max_length=3000)
    design_caution: str = Field(..., min_length=1, max_length=3000)


class ComposerEvidenceTargetPlan(BaseModel):
    id: str = Field(..., min_length=1, max_length=80, pattern=r"^[a-zA-Z0-9_-]+$")
    name: str = Field(..., min_length=1, max_length=250)
    why_it_matters: str = Field(..., min_length=1, max_length=2000)
    evidence_needed: list[str] = Field(default_factory=list, max_length=20)
    participant_can_know: list[str] = Field(default_factory=list, max_length=20)
    do_not_infer: list[str] = Field(default_factory=list, max_length=20)


class ComposerStudyFoundation(BaseModel):
    decision_to_inform: str = Field(..., min_length=1, max_length=3000)
    primary_research_question: str = Field(..., min_length=1, max_length=2000)
    secondary_research_questions: list[str] = Field(default_factory=list, max_length=20)
    participant_population: str = Field(..., min_length=1, max_length=2000)
    inclusion_signal: str = Field(..., min_length=1, max_length=2000)
    knowledge_relationship: str = Field(..., min_length=1, max_length=3000)
    participant_scope_definition: str = Field(..., min_length=1, max_length=2000)
    in_scope_categories: list[str] = Field(default_factory=list, max_length=30)
    out_of_scope_categories: list[str] = Field(default_factory=list, max_length=30)
    ambiguous_scope_cases: list[str] = Field(default_factory=list, max_length=30)
    clarification_principles: list[str] = Field(default_factory=list, max_length=20)
    evidence_targets: list[ComposerEvidenceTargetPlan] = Field(
        default_factory=list, max_length=30
    )
    privacy_and_safety_constraints: list[str] = Field(
        default_factory=list, max_length=30
    )
    wording_and_sequence_principles: list[str] = Field(
        default_factory=list, max_length=30
    )
    material_ambiguities: list[str] = Field(default_factory=list, max_length=20)
    source_notes: list[ComposerSourceNote] = Field(default_factory=list, max_length=12)


class ComposerAuditDimension(BaseModel):
    score: float = Field(..., ge=1, le=5)
    finding: str = Field(..., min_length=1, max_length=3000)


class ComposerAuditReport(BaseModel):
    goal_fit: ComposerAuditDimension
    answerability: ComposerAuditDimension
    realistic_evidence_capture: ComposerAuditDimension
    evidence_quality: ComposerAuditDimension
    neutrality: ComposerAuditDimension
    cohesion: ComposerAuditDimension
    privacy_and_safety: ComposerAuditDimension
    scope_clarity: ComposerAuditDimension
    timing_and_efficiency: ComposerAuditDimension
    changes_made: list[str] = Field(default_factory=list, max_length=30)
    remaining_warnings: list[str] = Field(default_factory=list, max_length=20)


class AuditedComposerDraft(BaseModel):
    audit: ComposerAuditReport
    draft: ComposerDraft


class QuestionComposeRequest(BaseModel):
    instruction: str = Field(..., min_length=2, max_length=6000)
    agent_id: UUID
    question_index: int = Field(..., ge=0, le=100)
    current_research_spec: ResearchSpec | None = None
    current_interview_guide: InterviewGuide
    conversation_history: list[ComposerTurn] = Field(default_factory=list, max_length=12)
    edit_history: list[ComposerEditEvent] = Field(default_factory=list, max_length=50)


class QuestionComposerDraft(BaseModel):
    summary: str = Field(..., min_length=1, max_length=2000)
    question: InterviewQuestion
    warnings: list[str] = Field(default_factory=list, max_length=10)


_COMPOSER_INSTRUCTIONS = """
You are a senior qualitative research director helping a researcher create or
revise an AI-moderated voice interview. Return only the requested structured
draft. Apply the researcher's latest instruction to the supplied current draft;
do not silently discard useful existing content.

Conversation and revision contract:
- Treat the latest instruction as the next turn in an ongoing editing
  conversation. Use the composer context to resolve references such as "it",
  "this question", "the first question", or "the opening".
- A critique of existing wording is an edit request. Respond to the critique;
  do not fall back to restating the study brief or generating a fresh guide.
- When the instruction is localized, make the smallest coherent change.
  Preserve the title, research specification, system prompt, welcome message,
  question order, and unrelated question wording verbatim unless the edit
  genuinely requires one of them to change.
- The selected question is the researcher's current editor focus. Use it to
  resolve "this question" or similar language, but prefer an explicit question
  number in the latest instruction when one is supplied.
- The composer context identifies question text edited manually since the last
  save. When the researcher asks to update "the questions I changed" or their
  "subdetails", preserve those question texts verbatim and revise only their
  supporting purpose, evidence_goal, probes, do_not_assume, target_ids, and
  genuinely relevant role variants. Do not rewrite untouched questions.
- Use edit_history to understand how the researcher arrived at the current
  draft, including manual edits, earlier AI revisions, undo, and redo. It is
  historical provenance, not a competing draft: the supplied current study,
  agent, and guide are always the source of truth. Never resurrect text that a
  later edit replaced or undid.
- The summary is the composer's direct conversational reply to the latest
  instruction. In 1-3 sentences, say what changed and why. When question
  wording changed, identify the question and include the new wording. Never
  use a generic study description as the summary and never output a menu of
  suggested actions.

Design requirements:
- Separate the decision, research questions, target population, evidence goals,
  participant information, fielding plan, and interview questions.
- Create a structured clarification_policy for every new study. It must contain
  a short participant-facing operational definition, concise in-scope and
  out-of-scope categories, known borderline cases, and a neutral rule for
  handling cases the policy cannot classify. The policy should answer likely
  questions such as "what does this term mean?" or "does this example count?"
  consistently without turning the definition into a leading explanation.
- Scope guidance is available on demand, not an opening script. Keep the
  participant definition minimal, avoid long product/example catalogues, and
  never assume that a named service is wholly in or out of scope when only a
  particular feature or interaction qualifies. For an ambiguous case, invite
  a brief description of what the feature did and retain the ambiguity rather
  than forcing a classification.
- Prefer 5-8 neutral, open, single-focus core questions for a 10-20 minute
  study, plus the short warm-up described below.
- For every first-time interview, include a short warm-up phase before deeper
  probing. Use one or two first_time questions that are concrete, bounded, and
  answerable on first hearing. A warm-up may enter the study topic directly;
  do not default to biography prompts, broad day-to-day role summaries, or
  questions such as "where does technology show up?" Budget about 1-3 minutes.
- In a longitudinal study, add one returning-only reconnection question about
  what has changed since the last conversation; do not repeat the full
  first-time warm-up for returning participants.
- Mark questions with phase=warm_up/core/closing and
  visit_scope=first_time/returning/all so the runtime can select them.
- Give every question a concise, semantic, unique stable ID. Never create
  copy-style IDs such as `question_copy`; retain an existing ID when revising
  the same card. Use `depends_on` only when a later card is unintelligible or
  unsafe before a named earlier anchor, and mark truly indispensable anchors
  `always_ask=true`.
- Choose one explicit orchestration_profile for the whole guide. Use
  `repair_only` for a standardized guide that permits one narrow answer-quality
  repair but no exploratory follow-ups; `semi_structured` for authored probes
  with bounded conversational flexibility; `exploratory` only when the
  researcher explicitly wants emergent probing; and `fixed` for an exact
  question sequence without evidence skipping. Do not combine a profile with
  contradictory legacy flags or give a follow-up profile zero follow-up
  capacity on every card.
- Warm-up is not demographic intake. Never request names, exact locations,
  employers, schools, child identifiers, or profile facts already supplied by
  recruitment. Do not fabricate interviewer self-disclosure or shared lived
  experience. Build trust through transparency, participant control, easy
  bounded questions, and content-grounded listening. The participant should
  not have to decide which parts of their life or role to summarize.
- Start the core with a concrete experience question. Move from broad experience to
  specific incidents, meaning, unmet needs, and a short closing reflection.
- When the rest of a guide truly requires a qualifying problem episode, mark
  that anchor answer_requirement=problem_episode. Use
  requirement_miss_action=end_interview only when downstream questions would
  otherwise inherit a false premise and no useful alternate module remains.
  The runtime will ask one neutral fit repair; do not duplicate that repair as
  a separate guide card.
- Each main question needs a purpose, evidence goal, things not to assume, and
  1-3 short probes. Probes must be optional routes, not a mandatory checklist.
- Make evidence goals atomic enough to drive a real follow-up decision. For a
  concrete episode, cover the observable sequence, consequence or outcome,
  and source of knowledge; when change over time matters, explicitly seek a
  reference period plus recurrence, persistence, or a before-and-after point.
- Audit adjacent cards and probes for semantic duplication. A probe should fill
  one named evidence gap, not invite the participant to repeat the episode or
  example they just gave. Do not use a generic "give me an example" probe on a
  card that is already anchored in a concrete incident.
- Never ask a compound question, imply that a problem exists, praise an answer,
  diagnose a child/student, or ask the participant to speculate as fact.
- Ask one question at a time. Use critical-incident probes such as what happened,
  what they noticed directly, what they did next, and what changed.
- Preserve provenance: distinguish direct observation, what the child/student
  said, and stories heard from other people.
- Use role_variants where wording genuinely differs for parents, educators, or
  another named participant role. Do not force every role into one sentence.
- Include privacy/skip/stop rights and avoid collecting names or identifying
  information about minors or third parties.
- The system prompt should sound like an attentive human qualitative moderator:
  concise acknowledgements, no canned praise, no repeated paraphrase, no answer
  summaries unless checking ambiguity, and a natural transition after probing.
- Require one brief, transparent signpost between warm-up and core topics, then
  maintain rapport through accurate grounding rather than generic social filler.
- Keep researcher-only analysis_notes out of the system prompt and questions.
- This is a voice interview with a text-first participant UI: wording must be
  easy to understand when heard once.
- Follow the output schema's enums exactly. Question priority is only
  "required" or "optional"; sensitivity is only "standard" or "elevated";
  probe_selection is only "ordered" or "relevance". Do not substitute labels
  such as high, medium, low, or moderate.
- Never encode an unknown fielding value as zero. Preserve supplied fielding
  values. Otherwise use the schema defaults (response_limit=50,
  soft_launch_limit=10, participant_id_parameter="PROLIFIC_PID") and add a
  warning that the researcher must confirm them before launch.
- The researcher owns the guide. Explain material concerns in warnings instead
  of inventing unsupported facts.
""".strip()


_QUESTION_COMPOSER_INSTRUCTIONS = """
You are a senior qualitative research director editing exactly one question
card in an AI-moderated interview guide. Return only the requested structured
question draft. You cannot edit the study, another question, or the guide order.

Editing contract:
- Treat the supplied target question as the source of truth and apply the
  researcher's latest instruction as the next turn in an editing conversation.
- Use relevant edit_history to understand how the target card arrived at its
  current state. The target question is always authoritative; never restore
  superseded or undone wording from history. Ignore history unrelated to this
  card unless it is needed to resolve a direct reference in the instruction.
- Preserve the main question text verbatim when the request concerns only its
  supporting details. Change the text only when the researcher asks to revise
  the wording or when the requested goal cannot otherwise be met.
- Keep the card internally coherent: module, phase, visit scope, purpose,
  evidence goals, optional probes, assumptions, targets, follow-up cap, and
  role variants should all reflect what the question can actually establish.
- Use adjacent questions only as read-only context to avoid duplication and
  keep the interview flow natural.
- Ask one neutral, open, single-focus question that is readily answerable on
  first hearing. Avoid broad biography, role-summary, or "where does X show
  up" wording that makes the participant choose the frame. Do not imply that
  harm, concern, benefit, use, or parental knowledge exists unless the wording
  intentionally allows the participant to say it does not.
- Probes are optional routes, not a checklist. Keep them short and grounded in
  evidence gaps. Separate direct observation, reports, and inference where
  relevant.
- A probe must add evidence the main answer may genuinely leave missing. Avoid
  generic requests for another example when the participant will already have
  supplied a concrete episode, and avoid duplicating adjacent cards.
- Never request names, schools, locations, usernames, private messages, or
  other identifying information about children or third parties.
- Follow the output schema's enums exactly. Question priority is only
  "required" or "optional" and sensitivity is only "standard" or "elevated";
  do not substitute labels such as high, medium, low, or moderate.
- The summary is a direct 1-3 sentence reply explaining what changed on this
  card. Do not restate the study brief or suggest changes to other cards.
""".strip()


_FOUNDATION_INSTRUCTIONS = """
You are the research-design stage of a bounded qualitative study composer.
Build a methodological foundation before anyone writes or revises interview
questions. Return only the requested structured foundation.

Use web search to retrieve a small amount of relevant prior knowledge about
the domain, target population, current terminology, and material research or
safety considerations. Prefer primary research, official sources, and
authoritative methodological guidance. Do not copy private posts, scrape
sensitive stories, or present source anecdotes as participant evidence.

Define the study construct operationally for participants. Identify broad
in-scope and out-of-scope categories and the borderline cases participants are
likely to ask about. Distinguish a service from a qualifying feature within
that service. Design clarification principles that give a direct, minimal
answer without front-loading examples or teaching participants what to say.

The interview's job is discovery. External knowledge may identify dimensions
the guide should be capable of capturing, but it must never become a premise
that tells participants which experiences, attitudes, benefits, harms, or
failures to report. Define what participants can reasonably know, what the
researcher must not infer, and what provenance distinctions matter.

Treat the latest researcher instruction and the supplied current draft as an
ongoing design conversation. The current draft is the source of truth. Preserve
the researcher's established decisions and manual edits unless the latest
instruction changes them. Resolve references using the supplied conversation
and edit history. Record consequential unresolved assumptions in
material_ambiguities instead of silently deciding them.

Treat a requested interview duration as a real design constraint. Carry it
into the sequence principles so the drafting and audit stages can balance
coverage, depth, question count, and optional probing realistically.
""".strip()


_STAGED_DRAFT_INSTRUCTIONS = f"""
{_COMPOSER_INSTRUCTIONS}

This is the drafting stage of a bounded workflow. A methodological foundation
is supplied with the researcher's instruction and current draft. Use it to
build explicit traceability from the decision to evidence targets and then to
questions. Do not turn source findings into leading question premises. Prefer
a coherent conversational funnel over a catalogue of themes.

Turn the foundation's construct boundaries into the research_spec's structured
clarification_policy. Write it so the live interviewer can answer a narrow
definition or "does this count?" question consistently. Do not paste source
notes, citations, or an exhaustive example list into participant-facing text.

Respect the requested duration. Design a plausible spoken interview within
that constraint by prioritizing main questions, capping follow-ups, and making
lower-priority routes optional. Do not shorten or lengthen the requested
duration unless the researcher explicitly asks for that change.
""".strip()


_AUDIT_INSTRUCTIONS = f"""
You are the independent audit-and-revision stage of a qualitative interview
composer. Evaluate the candidate against the original researcher instruction,
the current draft, and the methodological foundation, then return a revised,
launchable draft in the supplied schema.

{_COMPOSER_INSTRUCTIONS}

Be exacting. A polished list of open questions is not enough. The guide must
produce evidence that answers the research decision, remain answerable when
heard once, preserve evidence provenance and uncertainty, avoid hidden
premises, and progress like a natural interview rather than a checklist.

Audit the necessity of every question, the spoken progression, optional probe
quality, redundancy, participant burden, privacy, sensitivity boundaries,
role and visit fit, and question-to-evidence traceability. Treat the requested
duration as a real constraint: assess the welcome, main questions, plausible
answer depth, follow-up caps, and closing together. Preserve useful coverage
when it fits; make secondary routes optional or remove them when it does not.

Audit the clarification policy independently: can it answer likely definition
and boundary questions consistently, does it separate a service from an
in-scope feature, does it preserve genuinely ambiguous cases, and can it do so
without priming participants? Score that judgment in scope_clarity.

For localized revisions, do not use the audit as permission to rewrite
unrelated content. Preserve the current draft and the drafting stage's strong
material, making the smallest set of changes that materially improves the
requested result. Score every audit dimension on the schema's 1-to-5 scale
only. Put every unresolved launch concern in both audit.remaining_warnings and
the final draft's warnings. Return the audit and revised draft only.
""".strip()


def _normalize_question_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    """Map common model-authored labels onto the persisted guide vocabulary."""
    nested_draft = payload.get("draft")
    if isinstance(nested_draft, dict):
        _normalize_question_metadata(nested_draft)

    guide = payload.get("interview_guide")
    questions = guide.get("questions", []) if isinstance(guide, dict) else []
    single_question = payload.get("question")
    if isinstance(single_question, dict):
        questions = [*questions, single_question]

    for question in questions:
        if not isinstance(question, dict):
            continue

        priority = str(question.get("priority") or "").strip().lower()
        if priority not in {"required", "optional"}:
            question["priority"] = (
                "optional"
                if priority in {"low", "nice_to_have", "nice-to-have", "supplementary"}
                else "required"
            )

        sensitivity = str(question.get("sensitivity") or "").strip().lower()
        if sensitivity not in {"standard", "elevated"}:
            question["sensitivity"] = (
                "standard"
                if sensitivity in {"low", "minimal", "normal", "routine"}
                else "elevated"
            )

    if isinstance(guide, dict):
        # Preserve existing semantic IDs when possible, but make model-authored
        # omissions and copy-suffix collisions deterministic before Pydantic's
        # launch contract validates the draft.
        used_ids: set[str] = set()
        first_id_mapping: dict[str, str] = {}
        guide_questions = [q for q in guide.get("questions", []) if isinstance(q, dict)]
        for index, question in enumerate(guide_questions):
            old_id = str(question.get("id") or "").strip()
            seed = old_id or str(question.get("module") or "") or str(question.get("text") or "")
            base = re.sub(r"[^a-z0-9]+", "_", seed.lower()).strip("_")[:64]
            base = re.sub(r"(?:_copy(?:_\d+)?)$", "", base).strip("_")
            base = base or f"question_{index + 1}"
            candidate = base
            suffix = 2
            while candidate in used_ids:
                candidate = f"{base}_{suffix}"
                suffix += 1
            question["id"] = candidate
            used_ids.add(candidate)
            if old_id:
                first_id_mapping.setdefault(old_id, candidate)

        positions = {
            str(question["id"]): index for index, question in enumerate(guide_questions)
        }
        for index, question in enumerate(guide_questions):
            dependencies: list[str] = []
            for raw_dependency in question.get("depends_on") or []:
                dependency = first_id_mapping.get(
                    str(raw_dependency), str(raw_dependency)
                )
                if (
                    dependency in positions
                    and positions[dependency] < index
                    and dependency not in dependencies
                ):
                    dependencies.append(dependency)
            question["depends_on"] = dependencies

        profile = str(guide.get("orchestration_profile") or "").strip().lower()
        if profile not in {
            "fixed", "repair_only", "semi_structured", "exploratory"
        }:
            allow_follow_ups = guide.get("allow_follow_ups", True) is not False
            skip_answered = guide.get("skip_answered_questions")
            skip_answered = allow_follow_ups if skip_answered is None else bool(skip_answered)
            allow_emergent = bool(guide.get("allow_emergent_follow_ups", False))
            has_capacity = any(
                int(question.get("max_follow_ups", 3) or 0) > 0
                and (
                    bool(question.get("probes"))
                    or allow_emergent
                    or "max_follow_ups" not in question
                )
                for question in guide_questions
            )
            if not allow_follow_ups or not has_capacity:
                profile = "repair_only" if skip_answered else "fixed"
            elif allow_emergent:
                profile = "exploratory"
            else:
                profile = "semi_structured"
            guide["orchestration_profile"] = profile

        profile_rules = {
            "fixed": (False, False, False),
            "repair_only": (False, True, False),
            "semi_structured": (True, True, False),
            "exploratory": (True, True, True),
        }
        if profile in profile_rules:
            (
                guide["allow_follow_ups"],
                guide["skip_answered_questions"],
                guide["allow_emergent_follow_ups"],
            ) = profile_rules[profile]

        selection = str(guide.get("probe_selection") or "").strip().lower()
        if selection not in {"ordered", "relevance"}:
            guide["probe_selection"] = "relevance"

        warnings = payload.setdefault("warnings", [])
        if isinstance(warnings, list):
            for warning in interview_guide_quality_warnings(guide):
                if warning not in warnings and len(warnings) < 20:
                    warnings.append(warning)

    audit = payload.get("audit")
    if isinstance(audit, dict):
        for value in audit.values():
            if not isinstance(value, dict) or "score" not in value:
                continue
            try:
                score = float(value["score"])
            except (TypeError, ValueError):
                continue
            # Audit metadata is internal. Do not reject an otherwise valid
            # revised guide merely because the model used a 10-point scale.
            if 5 < score <= 10:
                value["score"] = score / 2

    research_spec = payload.get("research_spec")
    fielding = research_spec.get("fielding") if isinstance(research_spec, dict) else None
    if isinstance(fielding, dict):
        provisional = False
        try:
            response_limit = int(fielding.get("response_limit", 50))
        except (TypeError, ValueError):
            response_limit = 50
        if response_limit < 1:
            response_limit = 50
            provisional = True
        fielding["response_limit"] = min(response_limit, 100_000)

        soft_launch = fielding.get("soft_launch_limit", 10)
        if soft_launch is not None:
            try:
                soft_launch = int(soft_launch)
            except (TypeError, ValueError):
                soft_launch = 10
            if soft_launch < 1:
                soft_launch = 10
                provisional = True
            fielding["soft_launch_limit"] = min(
                soft_launch, fielding["response_limit"]
            )

        parameter = str(fielding.get("participant_id_parameter") or "").strip()
        if not parameter or not parameter.replace("_", "").replace("-", "").isalnum():
            fielding["participant_id_parameter"] = "PROLIFIC_PID"
            provisional = True

        if provisional:
            warnings = payload.setdefault("warnings", [])
            warning = (
                "Fielding counts or participant-ID settings were not specified; "
                "provisional defaults were applied and must be confirmed before launch."
            )
            if isinstance(warnings, list) and warning not in warnings and len(warnings) < 20:
                warnings.append(warning)

    return payload


def _validate_composer_json(
    raw: str, model: type[_ComposerModel]
) -> _ComposerModel:
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("The composer response must be a JSON object.")
    return model.model_validate(_normalize_question_metadata(payload))


def _model_name(value: str | None) -> str:
    model = value or _COMPOSER_MODEL
    return model.removeprefix("openai/")


def _question_texts(guide: InterviewGuide | dict[str, Any] | None) -> list[str]:
    if guide is None:
        return []
    raw = guide.model_dump(mode="json") if isinstance(guide, InterviewGuide) else guide
    return [str(question.get("text") or "") for question in raw.get("questions", [])]


def _edited_question_context(
    *,
    saved: dict[str, Any] | None,
    current: dict[str, Any] | None,
    explicit_ids: list[str],
) -> list[dict[str, Any]]:
    """Describe manually changed question text, preferring UI-tracked ids."""
    if not current:
        return []
    current_questions = current.get("questions", [])
    saved_questions = (saved or {}).get("questions", [])
    saved_by_id = {
        str(question.get("id")): question
        for question in saved_questions
        if question.get("id")
    }
    explicit = set(explicit_ids)
    changed: list[dict[str, Any]] = []
    for index, question in enumerate(current_questions):
        question_id = str(question.get("id") or "")
        saved_question = saved_by_id.get(question_id)
        if saved_question is None and index < len(saved_questions):
            saved_question = saved_questions[index]
        text_changed = bool(
            saved_question
            and str(saved_question.get("text") or "").strip()
            != str(question.get("text") or "").strip()
        )
        if question_id in explicit or (not explicit and text_changed):
            changed.append({
                "index": index,
                "number": index + 1,
                "id": question_id or None,
                "saved_text": (
                    str(saved_question.get("text") or "")
                    if saved_question
                    else None
                ),
                "current_text": str(question.get("text") or ""),
            })
    return changed


def _concrete_revision_summary(
    *,
    before: InterviewGuide | dict[str, Any] | None,
    result: ComposerDraft,
) -> str:
    """Replace vague model summaries for a single-question wording edit."""
    old_texts = _question_texts(before)
    new_texts = _question_texts(result.interview_guide)
    if not old_texts or len(old_texts) != len(new_texts):
        return result.summary

    changed = [
        index
        for index, (old, new) in enumerate(zip(old_texts, new_texts, strict=True))
        if old.strip() != new.strip()
    ]
    if len(changed) != 1:
        return result.summary

    index = changed[0]
    wording = new_texts[index].strip()
    if len(wording) > 500:
        wording = f"{wording[:497].rstrip()}…"
    return (
        f"I revised Question {index + 1} to: “{wording}” "
        "The other question wording is unchanged."
    )


async def _compose_with_openai(
    *, model: str, instruction: str, context: dict[str, Any]
) -> ComposerDraft:
    api_key = await _get_key("openai_api_key")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="Add an OpenAI API key in Settings before using the study composer.",
        )

    client = AsyncOpenAI(api_key=api_key, base_url=await _openai_api_base())
    model_name = _model_name(model)
    serialized_context = json.dumps(context, ensure_ascii=False, default=str)

    try:
        foundation_response = await client.responses.create(
            model=model_name,
            reasoning={"effort": "medium"},
            instructions=_FOUNDATION_INSTRUCTIONS,
            input=(
                "RESEARCHER INSTRUCTION\n"
                f"{instruction}\n\n"
                "CURRENT STUDY DRAFT AND EDIT CONTEXT (JSON)\n"
                f"{serialized_context}"
            ),
            tools=[{"type": "web_search"}],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "qualitative_study_foundation",
                    "schema": ComposerStudyFoundation.model_json_schema(),
                    "strict": False,
                }
            },
            max_output_tokens=7000,
            store=False,
        )
        if not foundation_response.output_text:
            raise ValueError("The foundation stage returned no structured output.")
        foundation = _validate_composer_json(
            foundation_response.output_text, ComposerStudyFoundation
        )
    except Exception as exc:
        raise ValueError("while researching the study foundation") from exc

    try:
        draft_response = await client.responses.create(
            model=model_name,
            reasoning={"effort": "medium"},
            instructions=_STAGED_DRAFT_INSTRUCTIONS,
            input=(
                "RESEARCHER INSTRUCTION\n"
                f"{instruction}\n\n"
                "METHODOLOGICAL FOUNDATION (JSON)\n"
                f"{foundation.model_dump_json()}\n\n"
                "CURRENT STUDY DRAFT AND EDIT CONTEXT (JSON)\n"
                f"{serialized_context}"
            ),
            text={
                "format": {
                    "type": "json_schema",
                    "name": "grounded_interview_study_draft",
                    "schema": ComposerDraft.model_json_schema(),
                    "strict": False,
                }
            },
            max_output_tokens=12000,
            store=False,
        )
        if not draft_response.output_text:
            raise ValueError("The drafting stage returned no structured output.")
        draft = _validate_composer_json(draft_response.output_text, ComposerDraft)
    except Exception as exc:
        raise ValueError("while drafting the interview guide") from exc

    try:
        audit_response = await client.responses.create(
            model=model_name,
            reasoning={"effort": "medium"},
            instructions=_AUDIT_INSTRUCTIONS,
            input=(
                "RESEARCHER INSTRUCTION\n"
                f"{instruction}\n\n"
                "METHODOLOGICAL FOUNDATION (JSON)\n"
                f"{foundation.model_dump_json()}\n\n"
                "CURRENT STUDY DRAFT AND EDIT CONTEXT (JSON)\n"
                f"{serialized_context}\n\n"
                "CANDIDATE DRAFT TO AUDIT (JSON)\n"
                f"{draft.model_dump_json()}"
            ),
            text={
                "format": {
                    "type": "json_schema",
                    "name": "audited_interview_study_draft",
                    "schema": AuditedComposerDraft.model_json_schema(),
                    "strict": False,
                }
            },
            max_output_tokens=14000,
            store=False,
        )
        if not audit_response.output_text:
            raise ValueError("The audit stage returned no structured output.")
        audited = _validate_composer_json(
            audit_response.output_text, AuditedComposerDraft
        )
    except Exception as exc:
        raise ValueError("while auditing the interview guide") from exc

    final = audited.draft
    for warning in audited.audit.remaining_warnings:
        if warning not in final.warnings and len(final.warnings) < 20:
            final.warnings.append(warning)
    scope_policy = final.research_spec.clarification_policy
    if not (
        scope_policy.participant_definition.strip()
        or scope_policy.in_scope
        or scope_policy.out_of_scope
    ):
        warning = (
            "The study's scope clarification policy is empty; define what is in "
            "scope, out of scope, and ambiguous before launch."
        )
        if warning not in final.warnings and len(final.warnings) < 20:
            final.warnings.append(warning)
    return final


async def _compose_question_with_openai(
    *, model: str, instruction: str, context: dict[str, Any]
) -> QuestionComposerDraft:
    api_key = await _get_key("openai_api_key")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="Add an OpenAI API key in Settings before using the question editor.",
        )

    client = AsyncOpenAI(api_key=api_key, base_url=await _openai_api_base())
    schema = QuestionComposerDraft.model_json_schema()
    response = await client.responses.create(
        model=_model_name(model),
        instructions=_QUESTION_COMPOSER_INSTRUCTIONS,
        input=(
            "RESEARCHER INSTRUCTION\n"
            f"{instruction}\n\n"
            "QUESTION EDITOR CONTEXT (JSON)\n"
            f"{json.dumps(context, ensure_ascii=False, default=str)}"
        ),
        text={
            "format": {
                "type": "json_schema",
                "name": "interview_question_card_draft",
                "schema": schema,
                "strict": False,
            }
        },
        max_output_tokens=5000,
        store=False,
    )
    raw = response.output_text
    if not raw:
        raise ValueError("The question editor returned an empty draft.")
    return _validate_composer_json(raw, QuestionComposerDraft)


@router.post("/question", response_model=QuestionComposerDraft)
async def compose_question(
    study_id: UUID,
    payload: QuestionComposeRequest,
    db: AsyncSession = Depends(get_db),
):
    """Revise one question card; the response cannot mutate any other card."""
    study = await db.get(Study, study_id)
    if not study:
        raise HTTPException(status_code=404, detail="Study not found")

    agent = await db.get(Agent, payload.agent_id)
    if not agent or agent.study_id != study_id:
        raise HTTPException(status_code=404, detail="Agent not found")

    questions = payload.current_interview_guide.questions
    if payload.question_index >= len(questions):
        raise HTTPException(status_code=422, detail="Question index is out of range")

    index = payload.question_index
    context = {
        "study": {
            "title": study.title,
            "research_spec": (
                payload.current_research_spec.model_dump(mode="json")
                if payload.current_research_spec
                else study.research_spec
            ),
        },
        "conversation_history": [
            turn.model_dump(mode="json") for turn in payload.conversation_history
        ],
        "edit_history": [
            event.model_dump(mode="json") for event in payload.edit_history
        ],
        "target": {
            "question_index": index,
            "question_number": index + 1,
            "question": questions[index].model_dump(mode="json"),
        },
        "read_only_flow_context": {
            "previous_question": (
                questions[index - 1].text if index > 0 else None
            ),
            "next_question": (
                questions[index + 1].text if index + 1 < len(questions) else None
            ),
            "total_questions": len(questions),
        },
    }

    try:
        return await _compose_question_with_openai(
            model=_COMPOSER_MODEL,
            instruction=payload.instruction,
            context=context,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Question card composer failed")
        raise HTTPException(
            status_code=502,
            detail=f"The question editor could not produce a valid draft: {exc}",
        ) from exc


@router.post("", response_model=ComposerDraft)
async def compose_study(
    study_id: UUID,
    payload: ComposeRequest,
    db: AsyncSession = Depends(get_db),
):
    """Generate an undoable draft; this endpoint never mutates the study."""

    study = await db.get(Study, study_id)
    if not study:
        raise HTTPException(status_code=404, detail="Study not found")

    agent: Agent | None = None
    if payload.agent_id:
        agent = await db.get(Agent, payload.agent_id)
        if not agent or agent.study_id != study_id:
            raise HTTPException(status_code=404, detail="Agent not found")

    current_guide = (
        payload.current_interview_guide.model_dump(mode="json")
        if payload.current_interview_guide
        else (agent.interview_guide if agent else None)
    )
    saved_guide = agent.interview_guide if agent else None
    edited_questions = _edited_question_context(
        saved=saved_guide,
        current=current_guide,
        explicit_ids=payload.edited_question_ids,
    )

    context = {
        "study": {
            "title": study.title,
            "description": study.description or "",
            "research_spec": (
                payload.current_research_spec.model_dump(mode="json")
                if payload.current_research_spec
                else study.research_spec
            ),
        },
        "agent": {
            "name": agent.name if agent else "AI interviewer",
            "interview_guide": current_guide,
            "saved_interview_guide": saved_guide,
            "system_prompt": (
                payload.current_system_prompt
                if payload.current_system_prompt is not None
                else (agent.system_prompt if agent else "")
            ),
            "welcome_message": (
                payload.current_welcome_message
                if payload.current_welcome_message is not None
                else (agent.welcome_message if agent else "")
            ),
        },
        "composer": {
            "conversation_history": [
                turn.model_dump(mode="json") for turn in payload.conversation_history
            ],
            "selected_question_index": payload.selected_question_index,
            "selected_question_number": (
                payload.selected_question_index + 1
                if payload.selected_question_index is not None
                else None
            ),
            "selected_question_id": payload.selected_question_id,
            "manually_edited_questions": edited_questions,
            "edit_history": [
                event.model_dump(mode="json") for event in payload.edit_history
            ],
        },
    }

    try:
        result = await _compose_with_openai(
            model=_COMPOSER_MODEL,
            instruction=payload.instruction,
            context=context,
        )
        result.summary = _concrete_revision_summary(
            before=context["agent"]["interview_guide"],
            result=result,
        )
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Study composer failed")
        raise HTTPException(
            status_code=502,
            detail=f"The study composer could not produce a valid draft: {exc}",
        ) from exc
