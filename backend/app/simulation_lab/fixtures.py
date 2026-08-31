from __future__ import annotations

from .models import BenchmarkCase, DatasetSummary, Persona, Technique, TranscriptTurn


TECHNIQUES = [
    Technique(id="context_only", name="Conversation context only", short_name="Context", description="Predict from the visible dialogue without a persona card.", adds=["Recent dialogue"], color="#64748b"),
    Technique(id="demographic_card", name="Demographic card", short_name="Demographics", description="Adds only stable, observable participant attributes.", adds=["Age", "Setting", "Presenting context"], color="#2563eb"),
    Technique(id="narrative_persona", name="Narrative persona", short_name="Narrative", description="Adds a compact backstory and motivations inferred before the held-out turn.", adds=["Background", "Motivations", "Stable tensions"], color="#7c3aed"),
    Technique(id="behavioral_profile", name="Behavioral policy", short_name="Behavior", description="Constrains how the participant tends to disclose, hedge, resist, and speak.", adds=["Disclosure tendency", "Language habits", "Interaction policy"], color="#d97706"),
    Technique(id="state_aware", name="Turn-level state", short_name="State-aware", description="Adds the participant's plausible emotional and conversational state at this turn.", adds=["Current affect", "Trust", "Readiness", "Immediate intent"], color="#0d7377"),
]


def _turn(role: str, text: str) -> TranscriptTurn:
    return TranscriptTurn(role=role, text=text)  # type: ignore[arg-type]


CASES = [
    BenchmarkCase(
        id="sleep-pressure",
        title="Deadlines and sleep",
        topic="academic stress",
        persona=Persona(
            age=19,
            setting="second-year university student living in a residence hall",
            presenting_context="requested a brief digital counseling check-in about stress and sleep",
            background=["Usually performs well academically", "Has recently missed two morning classes", "Does not want friends to think they cannot cope"],
            communication_style=["Uses short, casual sentences", "Minimizes distress before giving specifics", "Rarely uses clinical labels"],
            current_state=["Tired and somewhat embarrassed", "Trust is increasing", "Willing to share one concrete consequence but not ask directly for help"],
        ),
        transcript=[
            _turn("counselor", "What made this feel worth talking about today?"),
            _turn("client", "I guess sleep mostly. It's been kind of bad lately."),
            _turn("counselor", "What has 'kind of bad' looked like this week?"),
            _turn("client", "Staying up until like 3 doing work, then I just lie there anyway."),
            _turn("counselor", "And what do you notice the next day after a night like that?"),
            _turn("client", "I'm tired obviously. Coffee helps a bit."),
            _turn("counselor", "Anything else that has been harder to do?"),
            _turn("client", "I missed an early lecture twice, which isn't really like me."),
            _turn("counselor", "How did you feel about missing it?"),
            _turn("client", "Pretty bad, but I told my roommate I just didn't need to go."),
            _turn("counselor", "What was it like to give your roommate that explanation?"),
        ],
        held_out=_turn("client", "I don't know, easier than saying I'm stressed and can't get it together, I guess."),
    ),
    BenchmarkCase(
        id="social-withdrawal",
        title="Pulling back from friends",
        topic="low mood and social withdrawal",
        persona=Persona(
            age=21,
            setting="final-year commuter student who also works part time",
            presenting_context="responded to an online wellbeing outreach after reporting low mood",
            background=["Friend group usually meets on weekends", "Work and coursework have both intensified", "Feels guilty declining invitations"],
            communication_style=["Reflective but cautious", "Qualifies claims with maybe and probably", "Answers questions directly after a pause"],
            current_state=["Low energy", "Concerned friends will stop asking", "Considering a small reconnection but unsure it will help"],
        ),
        transcript=[
            _turn("counselor", "When did you first notice yourself pulling back?"),
            _turn("client", "Maybe three weeks ago? I started saying no to weekend stuff."),
            _turn("counselor", "What tends to happen when an invitation comes in?"),
            _turn("client", "I see it and think I should go, but even replying feels like effort."),
            _turn("counselor", "So there is a part of you that wants to go, and another part without much energy for it."),
            _turn("client", "Yeah. Then I wait too long and make up some work excuse."),
            _turn("counselor", "How have your friends responded?"),
            _turn("client", "They keep asking for now. One of them asked if I was mad at her."),
            _turn("counselor", "What did you say?"),
            _turn("client", "That I was just busy. She probably didn't believe me."),
            _turn("counselor", "If you could tell her one honest piece of what has been happening, what might it be?"),
        ],
        held_out=_turn("client", "Maybe just that I've been feeling off and it's not about her. I don't know if I'd say more than that."),
    ),
    BenchmarkCase(
        id="panic-lecture",
        title="An anxious lecture exit",
        topic="anxiety symptoms",
        persona=Persona(
            age=18,
            setting="first-semester university student living away from home",
            presenting_context="started a chat after abruptly leaving a crowded lecture",
            background=["Has felt overwhelmed in crowds before", "Has not discussed this with family", "Worries the experience means something is seriously wrong"],
            communication_style=["Concrete sensory detail when prompted", "Asks reassurance-seeking questions", "Self-corrects and uses conversational fillers"],
            current_state=["Calmer than during the event", "Still alarmed by the memory", "Wants an explanation but fears being judged"],
        ),
        transcript=[
            _turn("counselor", "Can you walk me through what you first noticed in the lecture?"),
            _turn("client", "It got really hot and I couldn't focus on the slides."),
            _turn("counselor", "What happened next?"),
            _turn("client", "My heart was going fast and it felt like everyone was too close."),
            _turn("counselor", "What did you do when you noticed that?"),
            _turn("client", "I tried to stay because leaving would be obvious, but then I just got up."),
            _turn("counselor", "Where did you go?"),
            _turn("client", "The bathroom. I sat in a stall for maybe ten minutes."),
            _turn("counselor", "What was going through your mind while you were there?"),
            _turn("client", "Mostly that I was being ridiculous, and also what if it happened again."),
            _turn("counselor", "What feels most worrying about it happening again?"),
        ],
        held_out=_turn("client", "That I won't be able to leave next time, or people will notice. Is that like a panic attack or am I overreacting?"),
    ),
]


DATASET = DatasetSummary(
    id="emerging-adult-demo-v1",
    name="Emerging-adult counseling demo",
    provenance="Illustrative conversations authored for this prototype; not real clinical records",
    population="Ages 18–21 in digital counseling-style conversations",
    case_count=len(CASES),
    warning="Use this to test the workflow, not to estimate real-world validity. Replace it with consented, appropriately governed transcripts before drawing conclusions.",
)


DEMO_RESPONSES: dict[str, dict[str, str]] = {
    "sleep-pressure": {
        "context_only": "It was okay. I just didn't really want to talk about it.",
        "demographic_card": "I guess it was easier than admitting I was struggling with school and sleep.",
        "narrative_persona": "It was easier to act like I chose not to go than say I couldn't handle everything.",
        "behavioral_profile": "I don't know, it was just easier than saying I'm actually stressed and not really coping.",
        "state_aware": "I don't know, easier than saying I'm stressed and can't really keep up, I guess.",
    },
    "social-withdrawal": {
        "context_only": "I would tell her I've been busy and don't have much energy lately.",
        "demographic_card": "Probably that work and classes have been a lot and I'm tired, not mad at her.",
        "narrative_persona": "Maybe that I haven't felt like myself lately and it isn't because of her.",
        "behavioral_profile": "Maybe just that I've felt a little off and it's not about her. I probably wouldn't explain everything.",
        "state_aware": "Maybe that I've been feeling off and it isn't about her. I don't know if I'd say much else yet.",
    },
    "panic-lecture": {
        "context_only": "I'm worried it will happen again and I won't know what to do.",
        "demographic_card": "That it happens again in class and everyone sees me leave.",
        "narrative_persona": "That I won't be able to get out, and people will see. I keep wondering what it was.",
        "behavioral_profile": "That I won't be able to leave and everyone will notice. Was that a panic attack or something?",
        "state_aware": "That I won't be able to leave next time, or people will notice. Was that like a panic attack, or am I making it a bigger deal?",
    },
}
