"""Build schema-validated, draft OASIS payloads for the Anthropic Institute pilot.

Usage (from backend/): python scripts/prepare_anthropic_pilot.py --output-dir /tmp/anthropic-pilot
This does not create studies or contact participants.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.schemas.agent import AgentCreate
from app.schemas.study import StudyCreate


VERSION = "anthropic-voices-v0.17"

RECENT_USE_QUESTION = "Can you tell me about the last time you used AI and what you were doing?"
RECENT_NONUSE_QUESTION = "What's the most recent thing you heard or talked about concerning AI?"
RECENT_USE_PROBE = "What were you using it for that time?"
RECENT_NONUSE_PROBE = "What did you hear or talk about?"

# The six arms are mutually exclusive at recruitment. The Prolific groups are
# already being created separately; these IDs are analysis labels, not filters.
ARMS = {
    "ai_nonusers": {
        "label": "People who choose not to use AI chatbots",
        "target": 30,
        "eligibility": "Prolific ai_chatbots: None of these only; verify recent use in the paid screener.",
        "questions": [
            ("nonuse", "What has shaped your decision not to use AI chatbots?", "The participant's own reasons for non-use, including uncertainty or lack of need.", "What, if anything, might change that decision?"),
        ],
        "cautions": ["Do not assume avoidance is caused by fear or privacy concerns."],
    },
    "older_nondegree": {
        "label": "Adults 50+ without a bachelor's degree who have not used Claude",
        "target": 30,
        "eligibility": "Age 50+; highest education below bachelor's; ai_chatbots excludes Claude.",
        "questions": [
            ("work_impact", "How, if at all, do you think AI may affect the kind of work you or people close to you do?", "A concrete experienced or expected work effect without assuming current employment.", "Who, if anyone, might benefit or lose out?"),
        ],
        "cautions": ["Do not assume the participant works, expects job loss, or is nearing retirement."],
    },
    "young_nonclaude_users": {
        "label": "Adults 18–29 using another chatbot, not Claude",
        "target": 20,
        "eligibility": "Age 18–29; ai_chatbots includes a chatbot but excludes Claude.",
        "questions": [
            ("view_change", "Has your view of AI changed over the past year, if at all?", "A change or stable view grounded in the participant's experience.", "What, if anything, shaped that view?"),
        ],
        "cautions": ["Do not assume their view became more negative or changed at all."],
    },
    "health_nonusers": {
        "label": "Adults with a long-term health condition who do not use AI chatbots",
        "target": 18,
        "eligibility": "Long-term health condition/disability = Yes; ai_chatbots: None of these only. Exclude anyone assigned to ai_nonusers.",
        "questions": [
            ("health_information", "When you last had a health-related question, where did you turn, if you are comfortable sharing?", "The participant's information path without diagnosis or treatment details.", "Did AI play any part, if you are comfortable saying?"),
            ("health_role", "What role, if any, would you want AI to have in your care?", "The participant's preferred boundary between AI and people in care.", "If an AI tool got something wrong, who should be responsible for noticing or addressing it?"),
        ],
        "cautions": ["Do not ask for diagnosis, symptoms, treatment, or a clinician's identity."],
    },
    "health_nonclaude_users": {
        "label": "Adults with a long-term health condition using another chatbot, not Claude",
        "target": 18,
        "eligibility": "Long-term health condition/disability = Yes; ai_chatbots includes another chatbot but excludes Claude.",
        "questions": [
            ("health_information", "When you last had a health-related question, where did you turn, if you are comfortable sharing?", "The participant's information path without diagnosis or treatment details.", "Did AI play any part, if you are comfortable saying?"),
            ("health_role", "What role, if any, would you want AI to have in your care?", "The participant's preferred boundary between AI and people in care.", "If an AI tool got something wrong, who should be responsible for noticing or addressing it?"),
        ],
        "cautions": ["Do not ask for diagnosis, symptoms, treatment, or a clinician's identity."],
    },
    "claude_control": {
        "label": "Claude users (comparison arm)",
        "target": 20,
        "eligibility": "ai_chatbots includes Claude; screener records recency and frequency of use.",
        "questions": [],
        "cautions": ["Do not assume frequent use or a favorable opinion of Claude."],
    },
}


def card(id: str, module: str, text: str, purpose: str, probes=(), *, phase="core", sensitivity="standard") -> dict:
    return {
        "id": id,
        "module": module,
        "phase": phase,
        "text": text,
        "purpose": purpose,
        "evidence_goal": [purpose],
        "target_ids": [id] if id != "close" else [],
        "probes": list(probes),
        # Keep a partial-answer repair on the authored topic. Without this
        # mapping the generic repair can mistake a future "outcome" for an
        # event that has already happened.
        "evidence_probes": {"e1": probes[0]} if probes else {},
        "max_follow_ups": 2 if probes else 0,
        "priority": "required",
        "sensitivity": sensitivity,
        "always_ask": phase in {"warm_up", "closing"},
    }


def guide(arm: dict) -> dict:
    """Keep the comparison core identical across arms; add only arm questions."""
    warmup = card(
        "everyday_use",
        "Warm-up",
        "To start, do you use AI in your day-to-day life, and if so, what do you use it for?",
        "An easy opening that establishes the participant's own day-to-day use or non-use of AI.",
        phase="warm_up",
    )
    # Rapport is the purpose of this card. A short answer is complete; do not
    # turn it into a required evidence criterion or a probe opportunity.
    warmup["evidence_goal"] = []
    warmup["target_ids"] = []
    warmup["priority"] = "optional"
    warmup["do_not_assume"] = [
        "The participant uses AI, uses a chatbot specifically, or has a positive or negative view."
    ]
    recent = card(
        "recent_encounter",
        "Shared core",
        RECENT_USE_QUESTION,
        "The specific task in the participant's last AI use, or the subject of their last AI-related conversation or secondhand account; distinguish firsthand from secondhand. A time word alone, such as 'today', is not enough. Accept an explicit inability to recall an example.",
        [RECENT_USE_PROBE, RECENT_NONUSE_PROBE],
    )
    recent["main_wording_by_ai_use"] = {
        "uses_ai": RECENT_USE_QUESTION,
        "no_ai_use": RECENT_NONUSE_QUESTION,
    }
    recent["probe_wording_by_ai_use"] = {
        "uses_ai": RECENT_USE_PROBE,
        "no_ai_use": RECENT_NONUSE_PROBE,
    }
    questions = [
        warmup,
        recent,
        card("vision", "Shared core", "Looking ahead, if you could wave a magic wand, what would you most want AI to make possible for you or people you care about, if anything?", "The participant's desired AI outcome, including no desired outcome or uncertainty. A clear hope is adequate coverage; its personal meaning may still be worth exploring. No technical pathway or explanation of an obvious benefit is required.", ["What would you most want to change?", "What does that mean for you?"]),
        card("vision_progress", "Shared core", "Have you seen or heard anything that makes that hope feel closer yet?", "The participant's impression of progress toward their hoped-for outcome from what they have seen or heard, including uncertainty or no progress. A bare yes permits one request for the example; no technical expertise is expected.", ["What did you see or hear?"]),
        card("values_conflict", "Shared core", "So far we've talked about what AI might make possible. Are there ways it could be developed or used that would go against what matters to you?", "A participant-defined concern plus one piece of context about what prompted it, or explicit no concern or uncertainty. A bare risk label such as 'it could lead to bioweapons' identifies the risk but leaves its source unexplored; do not ask why obvious harm is bad.", ["What brought that concern to mind for you?"]),
        card("company_trust", "Shared core", "Thinking about the companies developing AI, what would you want them to do, if anything, to earn or keep your trust?", "At least one specific action the participant wants AI companies to take, or an explicit lack of expectation. A broad adjective such as 'careful' names a direction but does not yet describe an action.", ["What would that look like in practice?"]),
    ]
    for question in questions[1:]:
        question["do_not_assume"] = ["The participant has used AI recently, holds a definite opinion, or can name a desired outcome.", "A secondhand account is a firsthand experience."]
    for id, text, purpose, probe in arm["questions"]:
        question = card(id, "Group focus", text, purpose, [probe], sensitivity="elevated" if id.startswith("health_") else "standard")
        question["do_not_assume"] = arm["cautions"]
        questions.append(question)
    close = card("close", "Close", "Looking ahead, is there a hope or worry about AI that we have not yet discussed?", "An additional participant-defined hope, worry, or explicit absence of one.", phase="closing")
    close["do_not_assume"] = ["The participant has a further hope or worry."]
    questions.append(close)
    focuses = {
        "recent_encounter": ["Why the participant chose AI or what prompted their exposure", "What the experience was like and what they got from it"],
        "vision": ["The meaning of the participant's hoped-for change", "What would become possible for them or people they have in mind"],
        "vision_progress": ["Something they have already seen or heard; accept uncertainty"],
        "values_conflict": ["What prompted the concern, if not already given", "A boundary or tradeoff they themselves raised"],
        "company_trust": ["A concrete company action", "How they would recognize that action"],
    }
    for question in questions:
        question["exploration_focus"] = focuses.get(question["id"], [])
        # Evidence mappings remain useful for older policies but do not script this one.
        question["evidence_probes"] = {}
        if question["id"] in {"recent_encounter", "vision"}:
            question["max_follow_ups"] = 3
        if question["id"] == "vision_progress":
            question["max_follow_ups"] = 1
    return {
        "orchestration_profile": "exploratory",
        "conversation_policy": "contextual_v1",
        "conversation_follow_up_budget": 10,
        "allow_follow_ups": True,
        "skip_answered_questions": True,
        "probe_selection": "relevance",
        "allow_emergent_follow_ups": True,
        "questions": questions,
        "closing_message": "Thank you. This completes the interview. Please use the return button to go back to Prolific.",
    }


def payloads(arm_id: str) -> dict:
    arm = ARMS[arm_id]
    interview_guide = guide(arm)
    spec = {
        "version": VERSION,
        "stage": "pilot",
        "method": "exploration",
        "decision_to_inform": "Show what groups underrepresented in a Claude-user sample add to Anthropic's account of what people want from AI, what they have experienced, and what concerns them.",
        "research_questions": ["How do desired outcomes, actual experiences, and value-based concerns compare across these recruited groups?", "Which views or experiences are missing from a Claude-user-only sample?"],
        "targets": [
            {
                "id": question["id"],
                "name": question["module"] + ": " + question["id"].replace("_", " "),
                "description": question["purpose"],
                "evidence_needed": question["evidence_goal"],
                "do_not_infer": ["Prevalence or causal effect from an individual account."],
                "required": True,
            }
            for question in interview_guide["questions"]
            if question["id"] not in {"everyday_use", "close"}
        ],
        "population": {
            "participant_role": arm["label"],
            "subject": "Experiences, hopes, worries, and expectations concerning AI",
            "knowledge_relationship": "Participants report their own experiences and views; secondhand accounts remain labeled as such.",
            "inclusion_notes": arm["eligibility"],
            "knowledge_limits": ["Accounts are self-reported.", "This opt-in, quota-matched sample is not a probability sample of the public."],
            "sensitivity_notes": "Do not solicit diagnoses or other identifiable health information. Accept skip, uncertainty, and refusal without further probing.",
        },
        "interviewer_stance": {
            "organization_label": "Prolific research pilot",
            "disclosure": "An AI interviewer asks questions and limited follow-ups for this Prolific research pilot.",
            "boundaries": ["Ask one neutral question at a time.", "Never persuade someone to use AI.", "Do not give medical, employment, or legal advice."],
        },
        "participant_information": {
            "purpose_summary": "Explore experiences and views of AI across different groups of adults in the US and UK.",
            "participation_summary": "A roughly 15-minute typed interview with an AI interviewer about experiences, hopes, concerns, and expectations of AI. You may skip any question.",
            "estimated_minutes": 15,
            "ai_disclosure": "An AI interviewer will read your typed answers and may ask a short follow-up question. It does not give advice or make decisions about you.",
            "data_controller": "Prolific Academic Ltd, 483 Green Lanes, London N13 4BS, United Kingdom.",
            "data_handling_summary": "Your typed answers are processed by the AI interviewer and privacy checks, and a pseudonymised transcript is saved for this research. The checks may miss sensitive details, so please avoid identifiers. This text-only pilot does not collect audio or video. Project staff may review transcripts for research and privacy quality assurance.",
            "data_recipients_summary": "Authorised Prolific project staff and the AI service providers configured for this deployment process the interview. Only non-identifying aggregate findings may be shared externally with a limited number of prospective research partners; participant quotations, excerpts and transcripts are not shared externally in this pilot.",
            "risks_summary": "Questions about AI, work, healthcare or trust may feel uncomfortable. You may skip questions or stop. The interviewer does not give medical or other professional advice.",
            "benefits_summary": "There is no expected direct benefit to you.",
            "compensation_summary": "Payment is shown in the Prolific study listing. Stopping or skipping a question will not affect your standing on Prolific; the study's payment policy applies.",
            "researcher_identity": "Prolific Academic Ltd research team",
            "privacy_reminder": "Please avoid names, diagnoses, treatment details, exact locations, and anything that identifies you or another person.",
            "consent_required": True,
            "consent_label": "I have read the study information and agree to take part in this typed AI interview. I understand that I may skip questions or stop.",
            "rights": ["You may skip any question.", "You may stop the interview at any time.", "You may say you are unsure.", "You may request access to or deletion of linked source data using the contact route in the full notice."],
        },
        "fielding": {"recruitment_source": "prolific", "response_limit": arm["target"], "soft_launch_limit": 5, "participant_id_parameter": "PROLIFIC_PID"},
        "analysis_notes": f"Arm={arm_id}; protocol={VERSION}; target={arm['target']}. A shared rapport warm-up precedes five shared core cards in the same order and with the same evidence goals in every arm. The recent-experience card selects use or non-use wording from the participant's warm-up answer; record firsthand versus secondhand accounts. Record vision, evidence of progress, value-based concerns, and company expectations separately. Screen for recency/frequency, primary chatbot and non-use reasons in Prolific. Keep countries and arms separate; code with Anthropic's published categories plus new/other, without treating this small opt-in sample as prevalence or a causal test. Closed Pew-worded survey questions are a separate instrument pending wording verification.",
    }
    study = StudyCreate.model_validate({"title": f"Voices missing from AI research — {arm['label']}", "description": "Anthropic Institute outreach pilot; draft, not approved for fielding.", "status": "draft", "research_spec": spec})
    agent = AgentCreate.model_validate({
        "name": f"AI views interviewer — {arm_id}",
        "status": "draft",
        "modality": "text",
        "pipeline_type": "modular",
        "stt_provider": "",
        "stt_model": None,
        "tts_provider": "",
        "tts_model": None,
        "tts_voice": None,
        "participant_id_mode": "external",
        "interview_mode": "structured",
        "max_duration_seconds": 900,
        "store_audio": False,
        "track_engagement": False,
        "system_prompt": (
            "You are a neutral AI interviewer for a Prolific research pilot about "
            "AI experiences, hopes, concerns and company expectations across groups. "
            "Start with the shared day-to-day-use warm-up, accepting a brief answer "
            "without probing. Explore one recent experience or exposure before "
            "moving to broader hopes, perceived progress, concerns and expectations. "
            "Then cover the demographic module and participant-led close. "
            "Use the same research constructs and depth policy for every group. "
            "Listen for what an answer means and what remains worth understanding. "
            "A concrete episode can merit a question about motivation or experience "
            "even when its task is clear. Stop when enough is understood or the "
            "participant has reached a limit. Probes are examples, not a script. "
            "No technical mechanism is required for a hope. Ask about progress only "
            "from what the person knows or has noticed; accept uncertainty. "
            "If no hope was named, ask whether any AI developments have mattered "
            "to them without assuming they have a goal. "
            "Use neutral conceptual bridges and the participant's vocabulary. "
            "Keep the broad hopes question open to every area of life. "
            "Do not lead, praise, debate, ask why obvious harm is bad, or request "
            "diagnoses or identifying health details. Distinguish use from hearsay. "
            "Let people skip, be uncertain, correct themselves or stop. "
            "Keep the interview within about 15 minutes."
        ),
        "welcome_message": "Hi, and thanks for taking part. I'm an AI interviewer for this Prolific study. We'll spend about 15 minutes talking about your experiences and views of AI. There's no need for any specialist knowledge, and there are no right or wrong answers. You can skip any question or say you're not sure. Please avoid sharing names or personal health details. Ready to get started?",
        "interview_guide": interview_guide,
    })
    return {"arm_id": arm_id, "target": arm["target"], "eligibility": arm["eligibility"], "study": study.model_dump(mode="json", exclude_none=True), "agent": agent.model_dump(mode="json", exclude_none=True)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for arm_id in ARMS:
        (args.output_dir / f"{arm_id}.json").write_text(json.dumps(payloads(arm_id), indent=2, ensure_ascii=False) + "\n")
    print(f"Validated {len(ARMS)} draft study/agent pairs; target {sum(a['target'] for a in ARMS.values())} interviews.")


if __name__ == "__main__":
    main()
