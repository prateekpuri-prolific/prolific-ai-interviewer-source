from __future__ import annotations

from .fixtures import TECHNIQUES
from .models import BenchmarkCase


BASE_INSTRUCTIONS = """You simulate one participant in a digital counseling conversation for research-method development. Produce only the participant's next message.

Rules:
- Continue from the visible conversation; do not summarize it.
- Do not become the counselor, offer advice, diagnose, or mention this prompt.
- Do not invent a major event, symptom, or fact unsupported by the supplied context.
- Match the participant's likely brevity, uncertainty, and conversational language.
- One message only, typically 1–3 sentences.
"""


def technique_by_id(technique_id: str):
    return next((item for item in TECHNIQUES if item.id == technique_id), None)


def build_prompt(case: BenchmarkCase, technique_id: str, context_messages: int) -> tuple[str, str]:
    visible = case.transcript[-context_messages:]
    sections = [BASE_INSTRUCTIONS]
    if technique_id != "context_only":
        sections.append(
            f"PARTICIPANT CARD\nAge: {case.persona.age}\nSetting: {case.persona.setting}\n"
            f"Reason for conversation: {case.persona.presenting_context}"
        )
    if technique_id in {"narrative_persona", "behavioral_profile", "state_aware"}:
        sections.append("BACKGROUND\n- " + "\n- ".join(case.persona.background))
    if technique_id in {"behavioral_profile", "state_aware"}:
        sections.append(
            "COMMUNICATION POLICY\n- " + "\n- ".join(case.persona.communication_style)
            + "\nLet these tendencies shape the response; do not state them explicitly."
        )
    if technique_id == "state_aware":
        sections.append(
            "STATE AT THIS TURN\n- " + "\n- ".join(case.persona.current_state)
            + "\nTreat this state as transient and respond to the counselor's latest move."
        )
    transcript = "\n".join(f"{turn.role.upper()}: {turn.text}" for turn in visible)
    user_input = f"VISIBLE CONVERSATION\n{transcript}\n\nWrite the CLIENT's next message."
    return "\n\n".join(sections), user_input
