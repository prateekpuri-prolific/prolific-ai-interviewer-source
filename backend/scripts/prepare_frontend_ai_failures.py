"""Build the frontend AI failure interview payload without contacting participants.

Run from backend/: PYTHONPATH=. ../.venv/bin/python scripts/prepare_frontend_ai_failures.py
"""

from __future__ import annotations

import json
from pathlib import Path

from app.schemas.agent import AgentCreate
from app.schemas.study import StudyCreate


OUT = Path(__file__).resolve().parents[1] / "studies" / "frontend_ai_failures_v02" / "interview.json"


def question(id, module, text, purpose, evidence, probes=(), *, phase="core", follow_ups=2,
             priority="required", depends_on=(),
             evidence_probes=None):
    return {
        "id": id, "module": module, "phase": phase, "text": text,
        "purpose": purpose, "target_ids": [id], "evidence_goal": list(evidence),
        "probes": list(probes), "evidence_probes": evidence_probes or {},
        "max_follow_ups": follow_ups,
        "priority": priority, "sensitivity": "standard", "visit_scope": "all",
        "depends_on": list(depends_on), "always_ask": False,
        "do_not_assume": [
            "The participant can paste confidential source code, private screenshots, customer data, or credentials.",
            "The AI caused the problem rather than missing context or an integration step.",
        ],
    }


QUESTIONS = [
    question("focal_episode", "Find a failure",
             "Think of one recent UI implementation or debugging attempt where you personally changed frontend code with AI and needed to correct its output. What were you trying to build or change, and when did this happen? If you cannot recall one, say so.",
             "Anchor the interview in one remembered firsthand episode with a bounded deliverable.",
             ["One specific AI assisted frontend attempt", "Intended UI or code change", "Approximate recency"],
             ["What would you have committed or shown when the task was finished?"],
             phase="warm_up"),
    question("starting_state", "Task and inputs",
             "Before you asked the AI, what already existed in the app, and what had to change?",
             "Separate starting state from the requested transformation.",
             ["Existing component, page, flow, or code state", "Requested change or new behavior"],
             ["Was this a new component, a change to existing code, a bug fix, or something else?",
              "Which nearby pieces of the app did the change have to work with?"],
             depends_on=["focal_episode"]),
    question("input_inventory", "Task and inputs",
             "What did you give the AI for this attempt? Please describe the kinds of inputs and their approximate size, without sharing the actual material.",
             "Record the full input envelope rather than treating the prompt alone as the task.",
             ["Instructions or request given", "Code or repository context supplied", "Design, visual, behavior, or test references supplied", "Approximate scale or amount"],
             ["What did your initial request ask the AI to do?",
              "Could it see the relevant files or repository?",
              "Did it have a design, behavior, or test reference?",
              "About how many files, screens, or states were involved?"],
             follow_ups=1, depends_on=["focal_episode"],
             evidence_probes={"e1": "What did your initial request ask the AI to do?",
                              "e2": "Could it see the relevant files or repository?",
                              "e3": "Did it have a design, behavior, or test reference?",
                              "e4": "About how many files, screens, or states were involved?"}),
    question("tool_and_workflow", "Task and inputs",
             "Which AI tool did you use, and how was it working with you: chat, editor completion, repo agent, browser agent, or another setup?",
             "Identify the product and level of access without guessing the underlying model.",
             ["Tool or product", "Mode of interaction and access", "Model or version if known"],
             ["Could it read the repo, edit files, run commands or tests, or see the browser?"],
             follow_ups=1, depends_on=["focal_episode"]),
    question("requirements", "Task and inputs",
             "What requirements did the result have to satisfy for this particular frontend task?",
             "Capture constraints that determine correctness and which the AI knew about.",
             ["Functional behavior", "Visual or design system expectations", "Relevant responsive, accessibility, browser, performance, or test constraints", "Which requirements the AI was told or could inspect"],
             ["What interaction had to work?",
              "What existing visual reference or design rule mattered?",
              "Which viewport, accessibility, browser, or test condition mattered most?",
              "Could the AI see that requirement, or did you have to tell it?"],
             follow_ups=1, depends_on=["focal_episode"],
             evidence_probes={"e1": "What interaction had to work?",
                              "e2": "What existing visual reference or design rule mattered?",
                              "e3": "Which viewport, accessibility, browser, or test condition mattered most?",
                              "e4": "Could the AI see that requirement, or did you have to tell it?"}),
    question("ai_actions", "What AI did",
             "Walk me through what the AI actually did or returned in that attempt, in order.",
             "Recover the observable sequence from request through output.",
             ["AI action or output", "Whether it edited, generated, explained, or tested", "Any follow-up instruction or retry before the failure was noticed"],
             [], follow_ups=0, depends_on=["focal_episode"]),
    question("failure_location", "Exact failure",
             "Where exactly did the result fall short? Describe one specific thing the AI produced or did, and what happened instead of what you needed. If there was no observed problem, say so.",
             "Obtain a concrete observed defect with expected versus actual behavior.",
             ["Specific location, UI state, behavior, or code path", "Observed result", "Expected result", "Why the difference mattered"],
             ["What would a user or reviewer actually see or do when this breaks?",
              "In which viewport, browser, state, or interaction does it show up?",
              "What should have happened in that same state?",
              "What could a user or reviewer not do because of that difference?",
              "Can you give a made-up, nonconfidential example with the same structure?"],
             follow_ups=3, depends_on=["ai_actions"],
             evidence_probes={"e1": "In which viewport, browser, state, or interaction does it show up?",
                              "e2": "What would a user or reviewer actually see or do when this breaks?",
                              "e3": "What should have happened in that same state?",
                              "e4": "What could a user or reviewer not do because of that difference?"}),
    question("detection", "Finding and fixing it",
             "How did you first notice the problem, and what did you have to check to confirm it?",
             "Capture real verification rather than assuming a test suite caught the issue.",
             ["Detection method", "Check or reproduction steps", "Whether the AI's own checks caught it"],
             [], follow_ups=0, depends_on=["failure_location"]),
    question("repair", "Finding and fixing it",
             "What did you do next to make the work usable, and how much back-and-forth or manual work did that take?",
             "Record concrete correction burden and whether the attempt was abandoned.",
             ["Fix, workaround, retry, or abandonment", "Number or nature of iterations", "Approximate time or effort and remaining risk"],
             ["What did you have to explain to the AI again?",
              "Which parts did you end up doing yourself?",
              "Would doing it without AI have been faster in this case?"],
             follow_ups=3, depends_on=["failure_location"]),
    question("right_result", "What right looks like",
             "Imagine the AI got this task right on the first useful attempt. What exactly would it have delivered or changed?",
             "Define the desired artifact and behavior in participant terms.",
             ["Expected deliverable or code change", "Correct user-visible behavior", "Fit with existing app or design"],
             ["What would be different in the specific state that failed?",
              "What should it have left alone?",
              "What information should it have asked you for if it could not know?"],
             follow_ups=3, depends_on=["repair"]),
    question("acceptance_checks", "What right looks like",
             "How would you decide the result was ready to merge or ship? Walk me through the checks you would use for this task.",
             "Make the success criterion observable and scorable.",
             ["Functional checks or tests", "Visual, responsive, accessibility, or browser checks when relevant", "Who decides acceptability", "What level of remaining manual review is acceptable"],
             ["Which exact test, interaction, or comparison would have caught this failure?",
              "What would a reviewer need to inspect even if tests passed?",
              "What would make you trust it enough to use without a rewrite?",
              "Who makes the final merge or ship decision for this work?",
              "How much manual review would still be acceptable?"],
             follow_ups=3, depends_on=["right_result"],
             evidence_probes={"e1": "Which exact test, interaction, or comparison would have caught this failure?",
                              "e2": "What would a reviewer need to inspect even if tests passed?",
                              "e3": "Who makes the final merge or ship decision for this work?",
                              "e4": "How much manual review would still be acceptable?"}),
    question("repeatability", "Pattern and impact",
             "Has this same kind of failure happened on other frontend tasks or with other AI tools, or was this a one-off?",
             "Distinguish recurring capability gap from a single incident.",
             ["Repeat, one-off, or unknown", "What varies between instances", "Other tool comparison if firsthand"],
             ["What kinds of tasks tend to trigger it?",
              "What is different when AI does succeed?"],
             follow_ups=2, depends_on=["failure_location"]),
    question("impact", "Pattern and impact",
             "How much does this limit your use of AI for frontend work today?",
             "Understand practical consequence rather than inferring importance from annoyance.",
             ["Frequency or avoided work", "Effect on time, quality, or confidence"],
             ["Do you still use AI for this task, use it only for part of it, or avoid it?"],
             follow_ups=1, depends_on=["failure_location"]),
    question("closing", "Close",
             "Is there another frontend task where AI falls short in a different way that we should investigate next?",
             "Capture a second lead without diluting the focal failure.",
             ["Another task or explicit none"], [], phase="closing", follow_ups=0,
             priority="optional"),
]


def build():
    targets = [
        {"id": q["id"], "name": q["module"] + ": " + q["id"].replace("_", " "),
         "description": q["purpose"], "evidence_needed": q["evidence_goal"],
         "do_not_infer": ["A population-wide rate or technical root cause from one account."],
         "required": q["priority"] == "required"}
        for q in QUESTIONS if q["id"] != "closing"
    ]
    spec = {
        "version": "frontend-ai-failures-v0.6", "stage": "pilot", "method": "exploration",
        "privacy_profile": "work_high_risk",
        "clarification_policy": {
            "participant_definition": "A firsthand AI-assisted frontend implementation or debugging attempt where you personally wrote or edited UI code.",
            "in_scope": ["Design-led UI work when the participant personally implemented or debugged frontend code", "Styling, interactions, accessibility, responsive behavior, and browser-visible UI states"],
            "out_of_scope": ["Standalone visual design or animation without frontend implementation", "Backend-only API or infrastructure changes"],
            "ambiguous_cases": ["A design task that may also have involved changing UI code"],
            "ambiguity_handling": "Ask whether the participant personally changed frontend code. If yes, accept the episode; otherwise invite one coding example once.",
        },
        "decision_to_inform": "Identify concrete, recurring frontend development failures worth improving or testing, with an observable definition of successful AI assistance.",
        "research_questions": [
            "Which frontend tasks do developers personally try to complete with AI, and what inputs and constraints do those tasks involve?",
            "What exactly does the AI do wrong, in which state or environment, and how is the failure detected and repaired?",
            "What deliverable, behavior, and verification checks would make AI assistance successful?",
            "Which failures recur, and what work or AI use do they affect?",
        ],
        "population": {
            "participant_role": "Adults doing hands-on frontend development who have personally tried generative AI for a frontend task.",
            "subject": "A recent firsthand attempt to use AI for a frontend implementation or debugging task.",
            "knowledge_relationship": "Developers describe their own tasks, observations, and acceptance checks; they do not speak for an employer or all frontend developers.",
            "inclusion_notes": "This draft follows the first ten-person Prolific pilot. Future recruitment should require a recent firsthand AI-assisted frontend implementation or debugging attempt and personally editing UI code. Recruitment filters are self-reported and need an eligibility check before another paid wave.",
            "knowledge_limits": ["Self-reported episodes may omit context or misremember sequence.", "A participant's explanation of model cause is an interpretation, not a diagnosis.", "One episode does not establish prevalence or model-wide failure."],
            "terminology_notes": "Use the developer's own terms. Frontend includes UI implementation, styling, interactions, state, accessibility, responsiveness, visual matching, and browser behavior.",
            "sensitivity_notes": "Ask for concrete work details and examples. Do not request pasted confidential source materials, credentials, customer data, or personal contact information.",
        },
        "interviewer_stance": {
            "organization_label": "Prolific research team",
            "disclosure": "An AI interviewer asks about one firsthand frontend development attempt. It does not assess the developer's skill.",
            "boundaries": ["Ask one question at a time and stay with the selected episode.", "Distinguish what happened from what the participant thinks caused it.", "Never blame the participant, argue that the AI was right, or offer coding advice.", "Do not collect confidential materials or identifying details.", "Accept skips, uncertainty, and stopping without pressure."],
        },
        "participant_information": {
            "purpose_summary": "Explore specific frontend development tasks that generative AI still struggles to help with.",
            "participation_summary": "A roughly 25-minute typed interview about one real frontend task. You will describe the task, what information the AI had, what it did wrong, how you checked or fixed it, and what a correct result would look like. You may skip any question.",
            "estimated_minutes": 25,
            "ai_disclosure": "An AI interviewer reads your typed answers and may ask follow-up questions. It does not evaluate you or your work.",
            "data_controller": "Prolific Academic Ltd, 483 Green Lanes, London N13 4BS, United Kingdom.",
            "data_handling_summary": "Your typed responses are processed by the AI interviewer. Automated checks remove clear high-risk details such as contact information, account numbers, and credentials; ordinary work descriptions remain in the transcript for research and quality review. The checks may miss some details. A session identifier supports withdrawal requests. Do not paste confidential work materials. No microphone, camera, or audio recording is used.",
            "data_recipients_summary": "Authorised Prolific project staff may review interview transcripts. Google Cloud hosts the application and database in London; OpenAI processes typed answers for interviewing through its standard API, which may process data outside the UK. Only non-identifying aggregate findings may be shared externally; transcripts, participant-level responses, and quotations will not be shared externally during this pilot.",
            "risks_summary": "Discussing frustrating work may be mildly uncomfortable. Do not paste confidential source materials or customer data. You may skip or stop.",
            "benefits_summary": "There is no expected direct benefit to you.",
            "compensation_summary": "Compensation terms, if any, are stated in the recruitment invitation. Honest participation is welcome even if you cannot recall a clear failure.",
            "privacy_reminder": "Describe your work and the AI's failure in detail. Do not paste confidential source materials, credentials, personal contact information, or customer data.",
            "researcher_identity": "Prateek Puri, Prolific Academic Ltd research team",
            "researcher_contact": "Email prateek.puri@prolific.com with the study title for research questions or a data request.",
            "complaints_contact": "Use the contact and rights-request routes in Prolific's Privacy and Legal Notices: https://www.prolific.com/privacy-and-legal",
            "privacy_notice_url": "https://www.prolific.com/privacy-and-legal",
            "data_retention_period": "Linked interview transcripts are retained for up to six months after this research project is completed, then deleted from the application. Backups expire on a rolling seven-day schedule. Non-identifying aggregate findings may be retained longer.",
            "withdrawal_instructions": "You may stop at any time. To request withdrawal of linked interview data, email the researcher with the study title and any session identifier you received, preferably within one month. Later data-rights requests will also be considered. Data already irreversibly combined into non-identifying aggregates cannot be removed.",
            "rights": ["You may skip any question.", "You may stop the interview at any time.", "You may say you are unsure."],
            "consent_required": True,
            "consent_label": "I have read the study information and agree to take part in this typed AI interview.",
        },
        "fielding": {"recruitment_source": "direct_link", "response_limit": 10,
                     "soft_launch_limit": 10},
        "targets": targets,
        "analysis_notes": "Analyze one focal firsthand episode per transcript. Preserve task, existing state, intended change, every input type and missing context, tool access, requirements, observed AI actions, exact expected-versus-actual failure, reproduction state, detection, correction burden, right-result specification, acceptance checks, recurrence, and impact. Distinguish observed fact from participant causal interpretation. Secondary tasks are leads only. Do not infer prevalence or model-wide inability from this convenience sample.",
    }
    agent = {
        "name": "Frontend AI failure interviewer", "status": "draft", "modality": "text",
        "pipeline_type": "modular", "llm_model": "openai/gpt-5.6-terra", "language": "en",
        "stt_provider": "", "stt_model": None,
        "tts_provider": "", "tts_model": None, "tts_voice": None,
        "max_duration_seconds": 1800, "participant_id_mode": "random",
        "widget_title": "Frontend AI Failures — Developer Interview",
        "widget_description": "A detailed typed interview about one frontend task AI struggled to complete.",
        "widget_primary_color": "#111827", "widget_show_progress": True,
        "store_audio": False, "track_engagement": False, "adaptive_enabled": False,
        "interview_mode": "structured",
        "welcome_message": "Hi, I'm an AI interviewer working with the Prolific research team. I'd like to understand one frontend task where AI fell short: the task, inputs, exact failure, and what a correct result would look like. This takes about 25 minutes. Please describe the work in detail, but do not paste confidential source material, credentials, personal contact information, or customer data. You can skip a question or stop at any time. Are you ready to begin?",
        "system_prompt": "You are a neutral research interviewer studying firsthand frontend development attempts with generative AI. The goal is unusually precise failure evidence. Select one recent concrete episode, then stay with it. Build a clear chain: task and starting state -> all input types and missing context -> AI actions and output -> one exact observed defect with expected versus actual behavior -> how it was detected and repaired -> what a correct result and acceptance checks would be. Ask one question at a time. Follow the structured guide, but use the participant's earlier answers to skip covered material and ask only for missing detail. Never ask someone to repeat an answer; if a detail is uncertain or withheld, move on. Prioritize the four core questions: task, inputs, exact failure, and right result. If the selected episode has no observed AI shortcoming, do not manufacture one; invite a different firsthand attempt once, and if none exists, close. Never turn lack of trust alone into an observed error. For UI failures ask about viewport, interaction, state, and visible behavior only when relevant; for code or architecture failures ask about interface, dependency, tests, and integration only when relevant. Probe for a small illustrative example if a failure remains abstract. Distinguish observed behavior from the participant's theory of its cause; do not diagnose the model. Do not assume missing context would have fixed the issue. Accept ordinary work context. Do not request pasted confidential source material, credentials, personal contact details, or customer data. Do not troubleshoot, recommend tools, evaluate the participant, or argue with their account. If the participant cannot recall a detail, record uncertainty and move on. Accept skips, corrections, and stopping. End naturally when the focal episode is adequately covered or time is up.",
        "interview_guide": {
            "orchestration_profile": "semi_structured", "max_quality_repairs": 4,
            "allow_follow_ups": True, "skip_answered_questions": True,
            "probe_selection": "relevance", "allow_emergent_follow_ups": False,
            "questions": QUESTIONS,
            "closing_message": "Thank you for walking through that example. This completes the interview. Please use the return button to go back to Prolific.",
        },
    }
    return {
        "study": StudyCreate.model_validate({"title": "Frontend developers: where AI falls short",
                                             "description": "Revised frontend task failure interview, pilot v0.6, based on the 10-person fielding review.",
                                             "status": "draft", "research_spec": spec}).model_dump(mode="json"),
        "agent": AgentCreate.model_validate(agent).model_dump(mode="json"),
    }


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(), indent=2, ensure_ascii=False) + "\n")
    print(OUT)
