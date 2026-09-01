"""
Tests for the structured-interview InterviewGuideProcessor.

We use Pipecat's bundled ``run_test`` helper to spin up a 3-processor pipeline
(source → InterviewGuideProcessor → sink), feed it scripted frames, and
inspect the downstream output.
"""

from __future__ import annotations

from typing import List

import pytest

from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    TextFrame,
    TranscriptionFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.tests.utils import SleepFrame, run_test

from app.pipeline.interview_guide import (
    DEFAULT_MAX_FOLLOW_UPS,
    InterviewGuideProcessor,
    StructuredOutputFilter,
    TextStructuredController,
    TurnDecision,
    build_structured_prompt,
    classify_participant_turn,
    conditional_module_skip_index,
    control_marker_prefix,
    resolve_role_guide,
    enforce_one_question_per_turn,
    looks_like_clarification,
    looks_like_more_procedural_help,
    looks_like_procedural_resume,
    participant_interruption_response,
    procedural_resume_response,
    resolve_orchestration_settings,
    strip_control_markers,
    strip_progress_marker,
)


# ── Helpers ────────────────────────────────────────────────────────────────


def _appended_messages(frames: List[Frame]) -> List[dict]:
    out: List[dict] = []
    for f in frames:
        if isinstance(f, LLMMessagesAppendFrame):
            out.extend(f.messages)
    return out


def _tf(text: str) -> TranscriptionFrame:
    return TranscriptionFrame(text=text, user_id="u", timestamp="t")


def _paced(*frames: Frame) -> List[Frame]:
    """Interleave a short sleep between frames. TranscriptionFrame is a
    queued DataFrame while the speaking frames are SystemFrames that jump
    the queue, so without pacing the test pushes them out of order — a race
    that doesn't exist in real sessions where frames arrive seconds apart."""
    out: List[Frame] = []
    for f in frames:
        out.append(f)
        out.append(SleepFrame(sleep=0.05))
    return out


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def two_question_guide() -> dict:
    return {
        "questions": [
            {
                "text": "Tell me about your typical day.",
                "probes": ["Walk me through this morning."],
                "max_follow_ups": 2,
                "transition": "Great, let's switch gears.",
            },
            {
                "text": "What's your biggest pain point right now?",
                "probes": ["Why is that the most important?"],
                "max_follow_ups": 1,
            },
        ],
        "closing_message": "Thanks for your time.",
    }


# ── build_structured_prompt sanity ──────────────────────────────────────────


class TestBuildStructuredPrompt:
    def test_prompt_includes_each_question(self, two_question_guide):
        out = build_structured_prompt("Be friendly.", two_question_guide)
        assert "Be friendly." in out
        assert "Tell me about your typical day." in out
        assert "What's your biggest pain point right now?" in out
        assert "INTERVIEW PROTOCOL" in out
        assert "Thanks for your time." in out

    def test_empty_guide_returns_base_prompt_unchanged(self):
        out = build_structured_prompt("Be friendly.", {"questions": []})
        assert out == "Be friendly."

    def test_default_closing_when_omitted(self):
        out = build_structured_prompt(
            "Hi.",
            {"questions": [{"text": "Q?", "probes": [], "max_follow_ups": 1}]},
        )
        assert "concludes our interview" in out

    def test_anti_repeat_and_neutrality_rules_present(self, two_question_guide):
        """Regression: production logs showed the model repeating Q1 and
        validating answers ('great', 'interesting'). The protocol must
        explicitly forbid both."""
        out = build_structured_prompt("Hi.", two_question_guide)
        assert "NEVER ask the same question twice" in out
        assert "great" in out  # part of the don't-validate rule
        assert "Do NOT acknowledge or evaluate" in out

    def test_scope_clarifications_are_minimal_and_do_not_advance(self, two_question_guide):
        out = build_structured_prompt("Hi.", two_question_guide)

        assert "scope clarification policy" in out
        assert "do not volunteer" in out
        assert "do not guess" in out
        assert "return to the original question" in out

    def test_probe_list_is_ordered_and_marked_verbatim(self, two_question_guide):
        out = build_structured_prompt("Hi.", two_question_guide)
        # Probes must be numbered and the prompt must say to use them
        # verbatim rather than invent new ones.
        assert "use these in order" in out
        assert "do NOT invent your own" in out
        assert "1. Walk me through this morning." in out

    def test_max_followups_is_a_hard_cap(self, two_question_guide):
        out = build_structured_prompt("Hi.", two_question_guide)
        assert "Maximum follow-ups for this question: 2" in out
        assert "you MUST move on" in out
        assert "ceiling, not a target" in out

    def test_follow_ups_can_be_disabled_for_a_fixed_survey(
        self, two_question_guide
    ):
        guide = {**two_question_guide, "allow_follow_ups": False}
        out = build_structured_prompt("Hi.", guide, emit_progress=True)

        assert "Follow-ups are disabled" in out
        assert "Ask ONLY the predefined main questions" in out
        assert "Maximum follow-ups for this question: 0" in out
        assert "1. Walk me through this morning." not in out
        assert '"action":"main|clarify|close"' in out
        assert "Never use `probe`" in out

    def test_evidence_skipping_is_independent_from_follow_ups(self):
        guide = {
            "allow_follow_ups": False,
            "skip_answered_questions": True,
            "questions": [
                {
                    "id": "task",
                    "text": "What task did you try?",
                    "evidence_goal": ["One concrete task"],
                    "always_ask": True,
                },
                {
                    "id": "inputs",
                    "text": "What inputs did you provide?",
                    "evidence_goal": ["Inputs supplied"],
                },
            ],
        }

        out = build_structured_prompt("Hi.", guide, emit_progress=True)

        assert "EVIDENCE-AWARE FIXED SURVEY OVERRIDE" in out
        assert "protected; always ask" in out
        assert "EVERY one of its evidence items is `covered`" in out
        assert '"action":"main|probe|clarify|close"' in out
        assert "permission for exploratory" in out

    def test_closing_waits_for_final_answer(self, two_question_guide):
        out = build_structured_prompt("Hi.", two_question_guide)
        assert "the participant has given\ntheir final answer" in out
        assert "Never deliver the closing message in the same turn as a question" in out

    def test_ordered_probe_selection_is_default(self, two_question_guide):
        # No probe_selection key → ordered wording.
        out = build_structured_prompt("Hi.", two_question_guide)
        assert "use these in order" in out
        assert "Ask probes from the list in order" in out

    def test_relevance_probe_selection_changes_wording(self, two_question_guide):
        relevance_guide = {**two_question_guide, "probe_selection": "relevance"}
        out = build_structured_prompt("Hi.", relevance_guide)
        # Ordered-specific wording is gone; relevance menu wording is present.
        assert "use these in order" not in out
        assert "a menu" in out
        assert "not a checklist" in out
        # The numbered probe list and follow-up cap are unchanged.
        assert "1. Walk me through this morning." in out
        assert "Maximum follow-ups for this question: 2" in out

    def test_emergent_probe_can_replace_menu_probe(self, two_question_guide):
        guide = {
            **two_question_guide,
            "probe_selection": "relevance",
            "allow_emergent_follow_ups": True,
        }
        out = build_structured_prompt("Hi.", guide)
        assert "replace a listed probe" in out
        assert "exact detail" in out
        assert "You do not need to exhaust the listed probes" in out

    def test_prompt_contains_evidence_based_rapport_rules(self, two_question_guide):
        out = build_structured_prompt("Hi.", two_question_guide)
        assert "rapport as a process" in out
        assert "Never fabricate a personal experience" in out
        assert "content-grounded acknowledgement" in out
        assert "warm-up-to-core signpost" in out

    def test_control_marker_catalog_is_added_with_progress(self):
        guide = {
            "questions": [
                {
                    "id": "recent_event",
                    "text": "What happened?",
                    "evidence_goal": ["Sequence", "Outcome"],
                    "probes": ["What happened next?"],
                }
            ]
        }
        out = build_structured_prompt("Hi.", guide, emit_progress=True)
        assert "Turn decision marker" in out
        assert "recent_event:main" in out
        assert "recent_event:e1" in out
        assert "recent_event:p1" in out
        assert "unknown|declined|not_applicable" in out

    def test_prompt_uses_narrow_focused_advance_for_partial_next_question(
        self, two_question_guide
    ):
        out = build_structured_prompt("Hi.", two_question_guide, emit_progress=True)

        assert "first unresolved question in guide order is already `partial`" in out
        assert "broad main wording" in out
        assert "in-order focused advance" in out
        assert 'Use `action="main"` only when the participant has supplied NO evidence' in out
        assert "internally inconsistent and forbidden" in out
        assert "sole exception to the normal rule that" in out
        assert "do not close a rich-episode interview after only one" in out


class TestResolveRoleGuide:
    def test_parent_and_educator_variants(self):
        guide = {
            "questions": [
                {
                    "text": "Tell me about young people.",
                    "probes": ["Generic?"],
                    "role_variants": {
                        "parent": {"text": "Tell me about your children.", "probes": ["Which child?"]},
                        "educator": "Tell me about your students.",
                    },
                }
            ]
        }
        parent = resolve_role_guide(guide, {"role": "guardian"})
        educator = resolve_role_guide(guide, {"role": "teacher"})
        assert parent["questions"][0]["text"] == "Tell me about your children."
        assert parent["questions"][0]["probes"] == ["Which child?"]
        assert educator["questions"][0]["text"] == "Tell me about your students."
        assert guide["questions"][0]["text"] == "Tell me about young people."

    def test_returning_wave_takes_precedence(self):
        guide = {
            "questions": [
                {
                    "text": "Baseline?",
                    "role_variants": {"parent": "Parent baseline?"},
                    "returning_role_variants": {"parent": "What changed since last time?"},
                }
            ]
        }
        resolved = resolve_role_guide(guide, {"role": "parent", "wave": 2})
        assert resolved["questions"][0]["text"] == "What changed since last time?"

    def test_visit_scope_selects_first_time_or_returning_warm_up(self):
        guide = {
            "questions": [
                {"id": "intro", "text": "Tell me about your role.", "visit_scope": "first_time"},
                {"id": "reconnect", "text": "What changed?", "visit_scope": "returning"},
                {"id": "core", "text": "Tell me about a recent example.", "visit_scope": "all"},
            ]
        }

        first = resolve_role_guide(guide, {"role": "parent", "wave": 1})
        returning = resolve_role_guide(guide, {"role": "parent", "wave": 2})
        no_profile = resolve_role_guide(guide, None)

        assert [q["id"] for q in first["questions"]] == ["intro", "core"]
        assert [q["id"] for q in returning["questions"]] == ["reconnect", "core"]
        assert [q["id"] for q in no_profile["questions"]] == ["intro", "core"]


def _llm_response(*chunks: str) -> List[Frame]:
    frames: List[Frame] = [LLMFullResponseStartFrame()]
    frames.extend(TextFrame(text=c) for c in chunks)
    frames.append(LLMFullResponseEndFrame())
    return frames


def _spoken_text(frames: List[Frame]) -> str:
    return "".join(f.text for f in frames if type(f) is TextFrame)


class TestStructuredOutputFilter:
    @pytest.mark.asyncio
    async def test_text_after_first_question_is_cut(self):
        """Regression replaying session 3bb8ac78: probe + leaked transition
        label + the NEXT main question, all crammed into one turn. Only the
        probe may reach the participant."""
        filt = StructuredOutputFilter()
        down, _ = await run_test(
            processor=filt,
            frames_to_send=_llm_response(
                "What was that experience like for you at the time?\n\n",
                "(Transition: Thanks, that's really helpful background.)\n\n",
                "Thinking about a specific recent moment, could you walk ",
                "me through what happened?",
            ),
        )
        spoken = _spoken_text(down)
        assert "What was that experience like for you at the time?" in spoken
        assert "Transition" not in spoken
        assert "helpful background" not in spoken
        assert "specific recent moment" not in spoken

    @pytest.mark.asyncio
    async def test_leading_transition_label_is_unwrapped(self):
        filt = StructuredOutputFilter()
        down, _ = await run_test(
            processor=filt,
            frames_to_send=_llm_response(
                "(Transition: Thanks, that's useful.) ",
                "Stepping back, what would have made a difference?",
            ),
        )
        spoken = _spoken_text(down)
        assert "Thanks, that's useful." in spoken
        assert "(Transition" not in spoken
        assert "Stepping back, what would have made a difference?" in spoken

    @pytest.mark.asyncio
    async def test_statement_only_turn_passes_through(self):
        filt = StructuredOutputFilter()
        down, _ = await run_test(
            processor=filt,
            frames_to_send=_llm_response(
                "Thank you for your time. ",
                "This concludes our interview.",
            ),
        )
        spoken = _spoken_text(down)
        assert "Thank you for your time." in spoken
        assert "This concludes our interview." in spoken

    @pytest.mark.asyncio
    async def test_preamble_before_question_is_kept(self):
        """A clarification answer followed by the repeated question is a
        legitimate turn shape — nothing before the first question may be
        dropped, and the cut starts only after it."""
        filt = StructuredOutputFilter()
        down, _ = await run_test(
            processor=filt,
            frames_to_send=_llm_response(
                "I mean a specific instance from your own life. ",
                "Could you share one? ",
                "Also, what first got you started?",
            ),
        )
        spoken = _spoken_text(down)
        assert "I mean a specific instance from your own life." in spoken
        assert "Could you share one?" in spoken
        assert "what first got you started" not in spoken

    @pytest.mark.asyncio
    async def test_chunks_split_mid_sentence(self):
        """Streamed tokens rarely align with sentence boundaries."""
        filt = StructuredOutputFilter()
        down, _ = await run_test(
            processor=filt,
            frames_to_send=_llm_response(
                "What were you thi", "nking or feeling at that point",
                "? Stepping ba", "ck, what would help?",
            ),
        )
        spoken = _spoken_text(down)
        assert "What were you thinking or feeling at that point?" in spoken
        assert "Stepping back" not in spoken

    @pytest.mark.asyncio
    async def test_consecutive_responses_reset_state(self):
        filt = StructuredOutputFilter()
        frames = _llm_response(
            "First question? This trailing text is cut.",
        ) + _llm_response(
            "Second turn statement. Second question?",
        )
        down, _ = await run_test(processor=filt, frames_to_send=frames)
        spoken = _spoken_text(down)
        assert "First question?" in spoken
        assert "trailing text" not in spoken
        assert "Second turn statement." in spoken
        assert "Second question?" in spoken


class TestStripProgressMarker:
    def test_extracts_and_removes_tag(self):
        cleaned, n = strip_progress_marker("[[Q3]] What changed for you?")
        assert n == 3
        assert "[[Q3]]" not in cleaned
        assert "What changed for you?" in cleaned

    def test_no_marker_returns_none(self):
        cleaned, n = strip_progress_marker("Just a normal answer.")
        assert n is None
        assert cleaned == "Just a normal answer."

    def test_tolerates_whitespace_and_case(self):
        _, n = strip_progress_marker("[[ q 2 ]] next question")
        assert n == 2

    def test_highest_number_wins_when_multiple(self):
        _, n = strip_progress_marker("[[Q1]] foo [[Q4]] bar")
        assert n == 4


class TestTurnDecisionMarker:
    def test_parses_and_strips_hidden_state(self):
        text = (
            '[[TURN {"action":"probe","question":2,'
            '"intent":"episode:p1","resolved":{"episode:e1":"covered"},'
            '"focal_entity":"current_tool","provenance":"direct"}]] '
            "[[Q2]] What happened next?"
        )
        cleaned, marker, decision = strip_control_markers(text)
        assert cleaned.strip() == "What happened next?"
        assert marker == 2
        assert decision == TurnDecision(
            action="probe",
            question=2,
            intent="episode:p1",
            resolved={"episode:e1": "covered"},
            focal_entity="current_tool",
            provenance="direct",
        )

    def test_invalid_json_is_stripped_without_becoming_state(self):
        cleaned, marker, decision = strip_control_markers(
            '[[TURN {not-json}]] [[Q1]] Hello?'
        )
        assert "TURN" not in cleaned
        assert marker == 1
        assert decision is None

    def test_context_prefix_keeps_reviewed_state_hidden(self):
        prefix = control_marker_prefix(
            1, TurnDecision(action="main", question=1, intent="intro:main")
        )
        assert prefix.startswith("[[TURN ")
        assert prefix.endswith("[[Q1]]")

    def test_thread_action_is_valid_hidden_state(self):
        cleaned, marker, decision = strip_control_markers(
            '[[TURN {"action":"thread","question":3,'
            '"intent":"impact:p1","resolved":{"impact:e1":"partial"}}]] '
            "What happened immediately after that?"
        )

        assert cleaned.strip() == "What happened immediately after that?"
        assert marker is None
        assert decision == TurnDecision(
            action="thread",
            question=3,
            intent="impact:p1",
            resolved={"impact:e1": "partial"},
        )

    def test_valid_turn_json_without_closing_brackets_is_parsed_and_hidden(self):
        cleaned, marker, decision = strip_control_markers(
            '[[TURN {"action":"thread","question":3,'
            '"intent":"impact:p1","resolved":{"impact:e1":"partial"}}\n'
            "What changed after that?"
        )

        assert cleaned.strip() == "What changed after that?"
        assert marker is None
        assert decision is not None
        assert decision.action == "thread"
        assert decision.resolved == {"impact:e1": "partial"}

    def test_malformed_turn_line_never_leaks_to_participant(self):
        cleaned, marker, decision = strip_control_markers(
            '[[TURN {"action":"main","question":2,broken\n[[Q2]] What happened?'
        )

        assert "TURN" not in cleaned
        assert "broken" not in cleaned
        assert cleaned.strip() == "What happened?"
        assert marker == 2
        assert decision is None


class TestProgressMarkerPrompt:
    def test_instruction_added_only_when_enabled(self, two_question_guide):
        on = build_structured_prompt("Hi.", two_question_guide, emit_progress=True)
        off = build_structured_prompt("Hi.", two_question_guide, emit_progress=False)
        assert "Progress marker" in on
        assert "[[Qn]]" in on
        assert "2 main questions" in on
        assert "every later" in on  # emphasises tagging beyond the first question
        assert "Progress marker" not in off


class TestProgressBarReporting:
    @pytest.mark.asyncio
    async def test_marker_is_stripped_and_reported(self):
        seen: List[tuple] = []

        async def cb(current, total):
            seen.append((current, total))

        filt = StructuredOutputFilter(progress_callback=cb, total_questions=3)
        down, _ = await run_test(
            processor=filt,
            frames_to_send=_llm_response(
                "[[Q2]] What would you change about that?",
            ),
        )
        spoken = _spoken_text(down)
        assert "[[Q2]]" not in spoken
        assert "What would you change about that?" in spoken
        assert seen == [(2, 3)]

    @pytest.mark.asyncio
    async def test_turn_marker_is_never_spoken(self):
        filt = StructuredOutputFilter(total_questions=2)
        down, _ = await run_test(
            processor=filt,
            frames_to_send=_llm_response(
                '[[TURN {"action":"main","question":1,'
                '"intent":"intro:main","resolved":{}}]] '
                "[[Q1]] Tell me about your day?"
            ),
        )
        spoken = _spoken_text(down)
        assert "TURN" not in spoken
        assert "intro:main" not in spoken
        assert "Tell me about your day?" in spoken

    @pytest.mark.asyncio
    async def test_marker_split_across_chunks(self):
        seen: List[tuple] = []

        async def cb(current, total):
            seen.append((current, total))

        filt = StructuredOutputFilter(progress_callback=cb, total_questions=4)
        down, _ = await run_test(
            processor=filt,
            frames_to_send=_llm_response("[[", "Q", "3", "]] ", "How did that feel?"),
        )
        spoken = _spoken_text(down)
        assert "Q3" not in spoken
        assert "How did that feel?" in spoken
        assert seen == [(3, 4)]

    @pytest.mark.asyncio
    async def test_progress_is_monotonic_and_clamped(self):
        seen: List[tuple] = []

        async def cb(current, total):
            seen.append((current, total))

        filt = StructuredOutputFilter(progress_callback=cb, total_questions=2)
        frames = (
            _llm_response("[[Q2]] Second question?")
            + _llm_response("[[Q1]] A backward jump that must be ignored?")
            + _llm_response("[[Q9]] Out of range, clamp to total?")
        )
        await run_test(processor=filt, frames_to_send=frames)
        # Q2 reported once; backward Q1 ignored; Q9 clamps to 2 which is not an
        # advance, so it's also ignored.
        assert seen == [(2, 2)]


class TestLooksLikeClarification:
    @pytest.mark.parametrize(
        "text",
        [
            "What do you mean by a concrete example?",
            "Sorry, I didn't get the whole question.",
            "Sorry.",
            "Could you repeat that?",
            "At what point do you mean?",
            "Pardon?",
            "I don't understand the question.",
            "Does Facebook count?",
        ],
    )
    def test_english_clarifications_detected(self, text):
        assert looks_like_clarification(text) is True

    @pytest.mark.parametrize(
        ("language", "text"),
        [
            ("de", "Wie meinst du das genau in diesem Zusammenhang bitte"),
            ("de", "Das habe ich leider akustisch nicht verstanden gerade eben"),
            ("de", "Kannst du die ganze Frage bitte noch einmal wiederholen"),
            ("es", "Perdona, no entendí bien lo que me estabas preguntando ahí"),
            ("es", "¿Puedes repetir la pregunta una vez más por favor entonces?"),
            ("fr", "Désolé, je n'ai pas compris ce que vous venez de demander là"),
            ("fr", "Pouvez-vous répéter la question s'il vous plaît encore une fois"),
            ("pt", "Desculpa, não entendi direito o que você estava perguntando agora"),
            ("nl", "Wat bedoel je daar precies mee in deze context dan"),
            ("it", "Scusa, non ho capito bene cosa mi stavi chiedendo adesso"),
            ("zh", "不好意思，我没听清楚你刚才问的那个问题是什么内容"),
            ("ja", "すみません、今の質問がよく聞き取れませんでしたのでお願いします"),
            ("ko", "죄송한데 방금 하신 질문을 잘 못 들었어요 한번만 더요"),
            ("ar", "عذرا لم أفهم السؤال الذي طرحته للتو بشكل واضح"),
            ("hi", "माफ़ कीजिए, मुझे आपका सवाल ठीक से समझ नहीं आया अभी"),
            # English baseline always applies, regardless of agent language.
            ("de", "Sorry, I didn't get the whole question at all there"),
            # Region suffixes are ignored.
            ("de-DE", "Das habe ich leider gar nicht verstanden gerade"),
        ],
    )
    def test_localized_clarifications_detected(self, language, text):
        assert looks_like_clarification(text, language) is True

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "I really like soccer and I do research.",
            "I kind of started analyzing data to figure out the best team.",
            "Would I check every item before using it?",
            "Will I correct them manually?",
            # Long answers ending in "?" are not clarifications.
            "I spent years collecting match data, building models, and "
            "comparing predictions to actual outcomes — does that count as "
            "a concrete example of what I find exciting about it?",
        ],
    )
    def test_answers_not_flagged(self, text):
        assert looks_like_clarification(text) is False

    def test_other_language_patterns_not_active_in_english(self):
        # A German answer mentioning "wiederholen" descriptively should not
        # trip the detector when the agent is configured for English… but
        # for a German agent it should.
        text = "Ich würde das Experiment gerne wiederholen"
        assert looks_like_clarification(text, "en") is False
        assert looks_like_clarification(text, "de") is True


class TestParticipantTurnControl:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Is this recorded?", "procedural"),
            ("How will my data be used?", "procedural"),
            ("Is this data gonna be private?", "procedural"),
            ("Is this gonna be privately protected?", "procedural"),
            ("How is what I share kept secure?", "procedural"),
            ("What's your data retention policy?", "procedural"),
            (
                "Will this be shared with anyone -- > the data I provide you?",
                "procedural",
            ),
            ("Do you share my answers with anyone?", "procedural"),
            ("Who will you share my data with?", "procedural"),
            ("What do you know about this study?", "procedural"),
            ("Wait, you moved on before I answered.", "flow_repair"),
            ("I haven't answered your last question.", "flow_repair"),
            ("I'm not done.", "continuation"),
            ("I mean, I think...", "continuation"),
            ("At that point I.", "continuation"),
            ("Does Facebook count?", "clarification"),
            (
                "Do you mean brainstorming a question or actually choosing it?",
                "clarification",
            ),
            ("I think AI saves me time.", "answer"),
            (
                "I had to go back to the source material and rewrite it.",
                "answer",
            ),
            ("I want to stop using AI for email.", "answer"),
            ("I want to finish this interview.", "stop"),
        ],
    )
    def test_adversarial_turns_are_classified_conservatively(
        self, text, expected
    ):
        assert classify_participant_turn(text) == expected

    def test_recording_answer_uses_public_protocol_then_waits(self):
        response = participant_interruption_response(
            "Is this recorded and who sees it?",
            {
                "data_handling_summary": (
                    "Raw audio is not saved; a transcript is stored for the "
                    "research team."
                )
            },
            "What kinds of work do you deliberately keep AI out of?",
        )
        assert "Raw audio is not saved" in response
        assert "What kinds of work do you deliberately keep AI out of?" not in response
        assert "Would you like to continue" in response
        assert response.count("?") == 1

    def test_privacy_answer_includes_the_study_privacy_reminder(self):
        response = participant_interruption_response(
            "Is this data gonna be private?",
            {
                "data_handling_summary": (
                    "Audio is not saved; researchers review the transcript."
                ),
                "privacy_reminder": "Please do not share identifying details.",
            },
            "What work have you used AI for?",
        )
        assert "Audio is not saved" in response
        assert "Please do not share identifying details" in response
        assert response.endswith("you'd like to ask first?")

    def test_data_sharing_curveball_uses_protocol_and_pauses(self):
        response = participant_interruption_response(
            "Will this be shared with anyone -- > the data I provide you?",
            {
                "data_handling_summary": (
                    "Audio is not saved; the research team reviews the transcript."
                ),
                "privacy_reminder": "Please keep confidential details general.",
            },
            "What was at stake in getting that task right?",
        )
        assert response.startswith("Audio is not saved")
        assert "What was at stake" not in response
        assert response.endswith("you'd like to ask first?")

    def test_combined_recipient_and_retention_question_answers_both(self):
        response = participant_interruption_response(
            "Who will be able to see my answers and how long will the transcript be kept?",
            {
                "data_handling_summary": "Audio is not saved.",
                "data_recipients_summary": (
                    "Authorized researchers and the study's service providers."
                ),
                "data_retention_period": (
                    "Pseudonymized transcripts are retained for 12 months."
                ),
            },
            "Are you ready to begin?",
        )
        assert "Authorized researchers" in response
        assert "12 months" in response
        assert "Are you ready to begin?" not in response
        assert response.endswith("you'd like to ask first?")

    def test_protected_wording_uses_data_handling_facts(self):
        response = participant_interruption_response(
            "Before I continue, is this gonna be privately protected?",
            {
                "purpose_summary": "We are studying professional AI use.",
                "data_handling_summary": (
                    "Audio is not saved; the research team reviews transcripts."
                ),
                "privacy_reminder": "Do not share confidential work details.",
            },
            "What happened in the task?",
        )
        assert response.startswith("Audio is not saved")
        assert "Do not share confidential work details" in response
        assert "We are studying professional AI use" not in response
        assert "What happened in the task?" not in response

    def test_study_information_question_uses_approved_public_facts(self):
        response = participant_interruption_response(
            "What do you know about this study?",
            {
                "purpose_summary": "We are studying professional AI use.",
                "data_handling_summary": "Audio is not saved.",
            },
            "What happened in the task?",
        )
        assert "We are studying professional AI use" in response
        assert "Audio is not saved" in response
        assert response.endswith("you'd like to ask first?")

    def test_procedural_pause_requires_explicit_resume(self):
        assert looks_like_procedural_resume("Yeah, that answers it.") is True
        assert looks_like_procedural_resume("Coding and website design.") is False
        assert looks_like_more_procedural_help("Not really.") is True
        response = procedural_resume_response("What work have you used AI for?")
        assert response == "Okay—let's continue. What work have you used AI for?"

    def test_incomplete_turn_invites_continuation_without_reasking(self):
        response = participant_interruption_response(
            "I mean, I think...",
            {},
            "What happened next?",
        )
        assert response == "Please continue—I'm listening."


# ── InterviewGuideProcessor state machine ──────────────────────────────────


class TestInterviewGuideProcessor:
    def test_initial_state(self, two_question_guide):
        proc = InterviewGuideProcessor(two_question_guide)
        snap = proc.snapshot()
        assert snap["current_question_index"] == 0
        assert snap["bot_turns_on_question"] == 0
        assert snap["nudge_pending"] is False
        assert snap["closed"] is False
        assert snap["total_questions"] == 2
        assert snap["user_has_spoken"] is False

    def test_voice_fixed_mode_rejects_mislabeled_generated_followup(
        self, two_question_guide
    ):
        guide = {**two_question_guide, "allow_follow_ups": False}
        proc = InterviewGuideProcessor(guide)

        review = proc.review_turn(
            "Wouldn't a more organized morning solve that problem?",
            1,
            TurnDecision(action="main", question=1, intent="q1:main"),
            "My mornings vary.",
        )

        assert review.replaced is True
        assert review.reason == "fixed_question_enforced"
        assert review.text == "Tell me about your typical day."
        assert review.marker == 1

    def test_voice_listed_probe_uses_server_owned_wording(self):
        guide = {
            "questions": [
                {
                    "id": "episode",
                    "text": "What happened?",
                    "probes": ["What happened immediately before that?"],
                }
            ]
        }
        proc = InterviewGuideProcessor(guide)
        proc.review_turn(
            "What happened?",
            1,
            TurnDecision(action="main", question=1, intent="episode:main"),
        )

        review = proc.review_turn(
            "The AI obviously failed, so why did you use it?",
            None,
            TurnDecision(action="probe", question=1, intent="episode:p1"),
            "The output surprised me.",
        )

        assert review.replaced is True
        assert review.reason == "listed_probe_wording_corrected"
        assert review.text == "What happened immediately before that?"

    @pytest.mark.asyncio
    async def test_no_nudge_within_budget(self, two_question_guide):
        proc = InterviewGuideProcessor(two_question_guide)
        # Q1 has max_follow_ups=2, so budget is 1 initial + 2 follow-ups = 3 bot turns.
        # We start with the spoken welcome (BotStopped), then the first user reply
        # opens the budget window. 2 counted bot turns is well under budget.
        frames_to_send = [
            BotStoppedSpeakingFrame(),  # welcome — NOT counted
            UserStoppedSpeakingFrame(),  # user has now spoken
            BotStoppedSpeakingFrame(),  # turn 1
            UserStoppedSpeakingFrame(),
            BotStoppedSpeakingFrame(),  # turn 2
            UserStoppedSpeakingFrame(),
        ]
        down, _ = await run_test(
            processor=proc,
            frames_to_send=frames_to_send,
            expected_down_frames=[
                BotStoppedSpeakingFrame,
                UserStoppedSpeakingFrame,
                BotStoppedSpeakingFrame,
                UserStoppedSpeakingFrame,
                BotStoppedSpeakingFrame,
                UserStoppedSpeakingFrame,
            ],
        )
        assert _appended_messages(down) == []
        assert proc.current_question_index == 0

    @pytest.mark.asyncio
    async def test_welcome_message_does_not_count_against_budget(
        self, two_question_guide
    ):
        """Bot turns before the participant speaks (the spoken welcome) must
        not eat into question 1's follow-up budget."""
        proc = InterviewGuideProcessor(two_question_guide)
        # 3 bot turns BEFORE any user turn — historically this would have
        # exhausted Q1 (budget = 3). After the fix, none of these count.
        frames_to_send = [
            BotStoppedSpeakingFrame(),
            BotStoppedSpeakingFrame(),
            BotStoppedSpeakingFrame(),
            UserStoppedSpeakingFrame(),
        ]
        down, _ = await run_test(
            processor=proc,
            frames_to_send=frames_to_send,
        )
        assert _appended_messages(down) == []
        assert proc.current_question_index == 0

    @pytest.mark.asyncio
    async def test_nudge_fires_after_budget_exhausted(self, two_question_guide):
        proc = InterviewGuideProcessor(two_question_guide)
        # The processor only counts ONE bot turn per intervening user turn,
        # so we have to alternate User/Bot to spend the budget. Q1 budget =
        # 1 + max_follow_ups=2 = 3 bot turns.
        frames_to_send = [
            UserStoppedSpeakingFrame(),  # user opens conversation
            BotStoppedSpeakingFrame(),  # ask Q1 (turn 1)
            UserStoppedSpeakingFrame(),
            BotStoppedSpeakingFrame(),  # probe (turn 2)
            UserStoppedSpeakingFrame(),
            BotStoppedSpeakingFrame(),  # probe (turn 3) -> budget hit
            UserStoppedSpeakingFrame(),  # nudge fires here
        ]
        down, _ = await run_test(
            processor=proc,
            frames_to_send=frames_to_send,
        )
        msgs = _appended_messages(down)
        assert len(msgs) == 1, f"expected 1 nudge, got {msgs}"
        # Marked user-role note — mid-conversation system messages are
        # rejected by some chat APIs (OpenAI gpt-5.x).
        assert msgs[0]["role"] == "user"
        assert "question 2" in msgs[0]["content"]
        assert "biggest pain point" in msgs[0]["content"]
        assert "Great, let's switch gears." in msgs[0]["content"]
        assert proc.current_question_index == 1

    @pytest.mark.asyncio
    async def test_silence_prompt_does_not_count_as_probe(
        self, two_question_guide
    ):
        """Regression: silence prompts ('take your time') are TTSSpeakFrames
        that emit BotStoppedSpeakingFrame too. They must NOT count against
        the question budget, otherwise a couple of silence nudges would
        prematurely advance Q1 without the model ever asking a real probe.
        """
        proc = InterviewGuideProcessor(two_question_guide)
        frames_to_send = [
            UserStoppedSpeakingFrame(),  # user opens
            BotStoppedSpeakingFrame(),  # real bot ask -> turn 1
            # User goes quiet — simulate three silence prompts firing in a
            # row without any user turn in between. None should count.
            BotStoppedSpeakingFrame(),
            BotStoppedSpeakingFrame(),
            BotStoppedSpeakingFrame(),
            UserStoppedSpeakingFrame(),  # user finally responds
            BotStoppedSpeakingFrame(),  # real probe -> turn 2
            UserStoppedSpeakingFrame(),
        ]
        down, _ = await run_test(
            processor=proc,
            frames_to_send=frames_to_send,
        )
        # Budget = 3, only 2 real turns counted, no nudge.
        assert _appended_messages(down) == []
        assert proc.current_question_index == 0
        assert proc.snapshot()["bot_turns_on_question"] == 2

    @pytest.mark.asyncio
    async def test_closing_nudge_after_final_question(self, two_question_guide):
        proc = InterviewGuideProcessor(two_question_guide)
        frames_to_send = [
            UserStoppedSpeakingFrame(),
            # Exhaust Q1 budget (3 counted bot turns, alternating with user)
            BotStoppedSpeakingFrame(),
            UserStoppedSpeakingFrame(),
            BotStoppedSpeakingFrame(),
            UserStoppedSpeakingFrame(),
            BotStoppedSpeakingFrame(),
            UserStoppedSpeakingFrame(),  # advance Q1 → Q2
            # Q2 budget = 1 + 1 = 2 counted bot turns
            BotStoppedSpeakingFrame(),
            UserStoppedSpeakingFrame(),
            BotStoppedSpeakingFrame(),
            UserStoppedSpeakingFrame(),  # closing nudge
        ]
        down, _ = await run_test(
            processor=proc,
            frames_to_send=frames_to_send,
        )
        msgs = _appended_messages(down)
        assert len(msgs) == 2, f"expected 2 nudges, got {len(msgs)}: {msgs}"
        closing = msgs[1]["content"]
        assert "completed all the questions" in closing
        assert "Thanks for your time." in closing
        assert proc.is_finished is True

    @pytest.mark.asyncio
    async def test_default_max_follow_ups_when_unset(self):
        guide = {
            "questions": [{"text": "Open?", "probes": []}],
            "closing_message": "Bye.",
        }
        proc = InterviewGuideProcessor(guide)
        # Budget = 1 + DEFAULT_MAX_FOLLOW_UPS counted bot turns. Alternate
        # so each one is actually counted.
        frames: List[Frame] = [UserStoppedSpeakingFrame()]
        for _ in range(1 + DEFAULT_MAX_FOLLOW_UPS):
            frames.append(BotStoppedSpeakingFrame())
            frames.append(UserStoppedSpeakingFrame())
        down, _ = await run_test(
            processor=proc,
            frames_to_send=frames,
        )
        msgs = _appended_messages(down)
        assert len(msgs) == 1
        assert "completed all the questions" in msgs[0]["content"]

    @pytest.mark.asyncio
    async def test_clarification_reply_does_not_burn_budget(
        self, two_question_guide
    ):
        """Regression: the agent's answer to "what do you mean?" was counted
        as a probe, so clarification exchanges exhausted the follow-up budget
        and forced an early advance."""
        proc = InterviewGuideProcessor(two_question_guide)
        frames_to_send = _paced(
            _tf("I work in research."),
            UserStoppedSpeakingFrame(),
            BotStoppedSpeakingFrame(),  # turn 1 (real probe)
            _tf("What do you mean by a concrete example?"),
            UserStoppedSpeakingFrame(),
            BotStoppedSpeakingFrame(),  # clarification answer — NOT counted
            _tf("Sorry, I didn't get the whole question."),
            UserStoppedSpeakingFrame(),
            BotStoppedSpeakingFrame(),  # repeat — NOT counted
        )
        down, _ = await run_test(
            processor=proc,
            frames_to_send=frames_to_send,
        )
        assert _appended_messages(down) == []
        assert proc.current_question_index == 0
        assert proc.snapshot()["bot_turns_on_question"] == 1

    @pytest.mark.asyncio
    async def test_closing_nudge_held_while_participant_asks_to_repeat(
        self, two_question_guide
    ):
        """Regression: at the final question, a "sorry, I didn't get that"
        triggered the close — the agent repeated the question and delivered
        the closing message in the same turn, never waiting for the answer.
        The nudge must be held until a substantive user turn."""
        proc = InterviewGuideProcessor(two_question_guide)
        frames_to_send = _paced(
            _tf("Hello."),
            UserStoppedSpeakingFrame(),
            # Exhaust Q1 budget (3 counted bot turns)
            BotStoppedSpeakingFrame(),
            _tf("I like soccer."),
            UserStoppedSpeakingFrame(),
            BotStoppedSpeakingFrame(),
            _tf("I analyze match statistics."),
            UserStoppedSpeakingFrame(),
            BotStoppedSpeakingFrame(),
            _tf("Mostly on weekends."),
            UserStoppedSpeakingFrame(),  # advance Q1 → Q2
            # Q2 budget = 1 + 1 = 2 counted bot turns
            BotStoppedSpeakingFrame(),
            _tf("Data quality is my pain point."),
            UserStoppedSpeakingFrame(),
            BotStoppedSpeakingFrame(),
            # Participant didn't catch the last question — must NOT close.
            _tf("Sorry, I didn't get the whole question."),
            UserStoppedSpeakingFrame(),
            BotStoppedSpeakingFrame(),  # agent repeats — uncounted
            # Now the real answer arrives — close fires here.
            _tf("Cleaning the raw data takes most of my time."),
            UserStoppedSpeakingFrame(),
        )
        down, _ = await run_test(
            processor=proc,
            frames_to_send=frames_to_send,
        )
        msgs = _appended_messages(down)
        assert len(msgs) == 2, f"expected advance + close, got {msgs}"
        closing = msgs[1]["content"]
        assert "completed all the questions" in closing
        assert "Thanks for your time." in closing
        assert "never combine a question with the closing message" in closing
        assert proc.is_finished is True

    def test_one_question_per_turn_rules_present(self, two_question_guide):
        out = build_structured_prompt("Hi.", two_question_guide)
        assert "at most ONE question mark" in out
        assert 'Never\n  read labels like "Transition:" aloud' in out

    def test_snapshot_mutates_with_state(self, two_question_guide):
        proc = InterviewGuideProcessor(two_question_guide)
        proc.current_question_index = 1
        proc._bot_turns_on_question = 2  # type: ignore[attr-defined]
        snap = proc.snapshot()
        assert snap["current_question_index"] == 1
        assert snap["bot_turns_on_question"] == 2
        assert snap["total_questions"] == 2

    def test_stats_reports_adherence(self, two_question_guide):
        proc = InterviewGuideProcessor(two_question_guide)
        stats = proc.stats()
        assert stats["channel"] == "voice"
        assert stats["total_questions"] == 2
        assert stats["completed"] is False
        assert stats["forced_advances"] == 0


# ── enforce_one_question_per_turn (non-streaming twin of the filter) ────────


class TestEnforceOneQuestion:
    def test_text_after_first_question_is_cut(self):
        cleaned, dropped = enforce_one_question_per_turn(
            "What was that experience like for you at the time?\n\n"
            "(Transition: Thanks, that's really helpful background.)\n\n"
            "Thinking about a specific recent moment, could you walk me "
            "through what happened?"
        )
        assert "What was that experience like for you at the time?" in cleaned
        assert "Transition" not in cleaned
        assert "helpful background" not in cleaned
        assert "specific recent moment" not in cleaned
        assert dropped > 0

    def test_leading_transition_label_is_unwrapped(self):
        cleaned, _ = enforce_one_question_per_turn(
            "(Transition: Thanks, that's useful.) "
            "Stepping back, what would have made a difference?"
        )
        assert "Thanks, that's useful." in cleaned
        assert "(Transition" not in cleaned
        assert "Stepping back, what would have made a difference?" in cleaned

    def test_statement_only_passes_through(self):
        cleaned, dropped = enforce_one_question_per_turn(
            "Thank you for your time. This concludes our interview."
        )
        assert "Thank you for your time." in cleaned
        assert "This concludes our interview." in cleaned
        assert dropped == 0

    def test_preamble_before_question_is_kept(self):
        cleaned, _ = enforce_one_question_per_turn(
            "I mean a specific instance from your own life. "
            "Could you share one? Also, what first got you started?"
        )
        assert "I mean a specific instance from your own life." in cleaned
        assert "Could you share one?" in cleaned
        assert "what first got you started" not in cleaned

    def test_no_trailing_punctuation(self):
        cleaned, dropped = enforce_one_question_per_turn(
            "What do you think about that"
        )
        assert cleaned == "What do you think about that"
        assert dropped == 0


# ── TextStructuredController (text-chat guardrail parity) ───────────────────


class TestTextStructuredController:
    def test_initial_state(self, two_question_guide):
        c = TextStructuredController(two_question_guide)
        assert c.total_questions == 2
        assert c.is_finished is False
        snap = c.snapshot()
        assert snap["current_question_index"] == 0
        assert snap["nudge_pending"] is False

    def test_legacy_zero_probe_contradiction_resolves_to_repair_only(self):
        settings = resolve_orchestration_settings({
            "allow_follow_ups": True,
            "skip_answered_questions": True,
            "allow_emergent_follow_ups": False,
            "questions": [{"text": "Question?", "max_follow_ups": 0, "probes": []}],
        })

        assert settings["profile"] == "repair_only"
        assert settings["allow_follow_ups"] is False
        assert settings["allow_coverage_clarifications"] is True

    def test_rhetorical_answer_cannot_be_relabelled_as_clarification(self):
        guide = {
            "orchestration_profile": "repair_only",
            "questions": [
                {"id": "checking", "text": "How did you check it?", "evidence_goal": ["Checking process"]},
                {"id": "outcome", "text": "What happened next?", "evidence_goal": ["Outcome"]},
            ],
            "closing_message": "Done.",
        }
        c = TextStructuredController(guide)
        c.review_turn(
            "How did you check it?",
            1,
            TurnDecision(action="main", question=1, intent="checking:main"),
            "Ready.",
        )

        review = c.review_turn(
            "How did you check it?",
            None,
            TurnDecision(
                action="clarify",
                question=1,
                resolved={"checking:e1": "covered"},
            ),
            "Would I check every item before using it?",
        )

        assert review.replaced is True
        assert review.reason == "clarify_without_participant_request"
        assert review.text == "What happened next?"

    def test_marker_cannot_jump_over_protected_anchor(self):
        guide = {
            "orchestration_profile": "repair_only",
            "questions": [
                {"id": "anchor", "text": "Describe one recent task.", "always_ask": True},
                {"id": "detail", "text": "What tool did you use?"},
                {"id": "outcome", "text": "What happened?"},
            ],
        }
        c = TextStructuredController(guide)

        c.sync_to_marker(3)

        assert c.current_question_index == 0

    def test_dependency_is_server_enforced(self):
        guide = {
            "orchestration_profile": "repair_only",
            "questions": [
                {"id": "episode", "text": "Describe one recent task."},
                {"id": "inputs", "text": "What did you give it?", "depends_on": ["episode"]},
            ],
        }
        c = TextStructuredController(guide)

        review = c.review_turn(
            "What did you give it?",
            2,
            TurnDecision(action="main", question=2, intent="inputs:main"),
            "Ready.",
        )

        assert review.replaced is True
        assert review.reason == "unmet_anchor_or_dependency"
        assert review.text == "Describe one recent task."

    def test_disabled_follow_ups_advance_after_one_main_question(
        self, two_question_guide
    ):
        guide = {**two_question_guide, "allow_follow_ups": False}
        c = TextStructuredController(guide)

        c.register_bot_turn("A substantive answer")
        message = c.maybe_advance_message("Another substantive answer")

        assert message is not None
        assert "question 2" in message["content"]
        assert c.current_question_index == 1

    @staticmethod
    def _evidence_skip_guide() -> dict:
        return {
            "allow_follow_ups": False,
            "skip_answered_questions": True,
            "questions": [
                {
                    "id": "recent_task",
                    "text": "What recent task did you try?",
                    "evidence_goal": ["One concrete recent task"],
                    "always_ask": True,
                },
                {
                    "id": "future_task",
                    "text": "What other task would you want AI to handle?",
                    "evidence_goal": ["One desired future task"],
                    "always_ask": True,
                },
                {
                    "id": "workflow",
                    "text": "How do you complete that task today?",
                    "evidence_goal": ["Current workflow", "Manual effort"],
                },
            ],
            "closing_message": "Done.",
        }

    def test_evidence_skip_protects_task_anchor(self):
        c = TextStructuredController(self._evidence_skip_guide())
        c.review_turn(
            "What recent task did you try?",
            1,
            TurnDecision(action="main", question=1, intent="recent_task:main"),
        )
        c.register_bot_turn("Ready.")
        c.maybe_advance_message("I also want AI to coordinate vendor reviews.")

        review = c.review_turn(
            "How do you complete that task today?",
            3,
            TurnDecision(
                action="main",
                question=3,
                intent="workflow:main",
                resolved={"future_task:e1": "covered"},
                focal_entity="vendor_risk_review",
            ),
            "I also want AI to coordinate vendor reviews.",
        )

        assert review.replaced is True
        assert review.question_index == 1
        assert review.text == "What other task would you want AI to handle?"

    def test_evidence_skip_names_task_after_skipped_card(self):
        guide = self._evidence_skip_guide()
        guide["questions"][1]["always_ask"] = False
        c = TextStructuredController(guide)
        c.review_turn(
            "What recent task did you try?",
            1,
            TurnDecision(action="main", question=1, intent="recent_task:main"),
        )
        c.register_bot_turn("Ready.")
        c.maybe_advance_message("I want AI to coordinate vendor reviews.")

        review = c.review_turn(
            "How do you complete that task today?",
            3,
            TurnDecision(
                action="main",
                question=3,
                intent="workflow:main",
                resolved={"future_task:e1": "covered"},
                focal_entity="vendor_risk_review",
            ),
            "I want AI to coordinate vendor reviews.",
        )

        assert review.replaced is True
        assert review.reason == "task_context_after_skip"
        assert review.text.startswith("Thinking about the vendor risk review,")

    def test_evidence_skip_does_not_prefix_self_contained_anchor(self):
        guide = self._evidence_skip_guide()
        guide["questions"][1]["always_ask"] = False
        guide["questions"][2]["text"] = (
            "Thinking back six months, what work task feels easier now?"
        )
        c = TextStructuredController(guide)
        c.review_turn(
            "What recent task did you try?",
            1,
            TurnDecision(action="main", question=1, intent="recent_task:main"),
        )
        c.register_bot_turn("Ready.")
        c.maybe_advance_message("I want AI to coordinate vendor reviews.")

        review = c.review_turn(
            "Thinking back six months, what work task feels easier now?",
            3,
            TurnDecision(
                action="main",
                question=3,
                intent="workflow:main",
                resolved={"future_task:e1": "covered"},
                focal_entity="vendor_risk_review",
            ),
            "I want AI to coordinate vendor reviews.",
        )

        assert review.replaced is False
        assert review.text == "Thinking back six months, what work task feels easier now?"

    def test_fixed_skip_allows_one_partial_evidence_repair(self):
        c = TextStructuredController(self._evidence_skip_guide())
        c.review_turn(
            "What recent task did you try?",
            1,
            TurnDecision(action="main", question=1, intent="recent_task:main"),
        )
        c.register_bot_turn("Ready.")
        c.maybe_advance_message("I want AI to coordinate vendor reviews.")
        c.review_turn(
            "What other task would you want AI to handle?",
            2,
            TurnDecision(action="main", question=2, intent="future_task:main"),
        )
        c.register_bot_turn("I want AI to coordinate vendor reviews.")
        c.maybe_advance_message("I still do it manually.")

        review = c.review_turn(
            "What part still requires manual work?",
            None,
            TurnDecision(
                action="probe",
                question=3,
                intent="workflow:emergent:clarification",
                resolved={"workflow:e1": "partial"},
            ),
            "I still do it manually.",
        )

        assert review.replaced is False
        assert review.decision is not None
        assert review.decision.action == "probe"

    def test_none_skips_remaining_dependent_module_questions(self):
        guide = {
            "allow_follow_ups": False,
            "questions": [
                {
                    "id": "unmet_task",
                    "module": "Future unmet task",
                    "text": "What other task would you want AI to do?",
                    "skip_remaining_module_if_not_applicable": True,
                },
                {
                    "id": "inputs",
                    "module": "Future unmet task",
                    "text": "What inputs would it need?",
                },
                {
                    "id": "outputs",
                    "module": "Future unmet task",
                    "text": "What output would be useful?",
                },
            ],
            "closing_message": "That completes the interview. Thank you.",
        }
        c = TextStructuredController(guide)
        c.register_bot_turn("Ready.")

        message = c.maybe_advance_message("I'm drawing a blank on another task.")
        assert message is not None
        assert "question 2" in message["content"]

        review = c.review_turn(
            "What inputs would it need?",
            2,
            TurnDecision(
                action="main",
                question=2,
                intent="inputs:main",
                not_applicable_question=1,
            ),
            "I'm drawing a blank on another task.",
        )

        assert review.closed is True
        assert review.text == "That completes the interview. Thank you."
        assert review.reason == "module_not_applicable"
        assert c.current_question_index == 3
        assert c.is_finished is True

    def test_conditional_skip_stops_at_the_next_module(self):
        questions = [
            {
                "module": "Dependent details",
                "skip_remaining_module_if_not_applicable": True,
            },
            {"module": "Dependent details"},
            {"module": "Closing reflection"},
        ]

        decision = TurnDecision(
            action="main", question=2, not_applicable_question=1
        )
        assert conditional_module_skip_index(questions, decision, 1) == (0, 2)
        assert conditional_module_skip_index(
            questions, TurnDecision(action="main", question=2), 1
        ) is None

    def test_disabled_follow_ups_replace_model_probe(self, two_question_guide):
        guide = {**two_question_guide, "allow_follow_ups": False}
        c = TextStructuredController(guide)
        decision = TurnDecision(action="probe", question=1, intent="q1:p1")

        review = c.review_turn("What happened next?", None, decision)

        assert review.replaced is True
        assert review.reason == "fixed_question_enforced"
        assert review.text == "Tell me about your typical day."
        assert review.decision is not None
        assert review.decision.action == "main"

    @pytest.mark.parametrize(
        ("model_text", "marker", "decision"),
        [
            (
                "Wouldn't you agree that mornings are the hardest part?",
                1,
                TurnDecision(action="main", question=1, intent="q1:main"),
            ),
            (
                "What's your biggest pain point right now?",
                2,
                TurnDecision(action="main", question=2, intent="q2:main"),
            ),
            (
                "Thanks for your time.",
                None,
                TurnDecision(action="close"),
            ),
        ],
    )
    def test_adversarial_fixed_mode_emits_exact_current_question(
        self, two_question_guide, model_text, marker, decision
    ):
        """Previously, a generated follow-up mislabeled as `main`, a forward
        marker, or a premature close could bypass the no-follow-up toggle."""
        guide = {**two_question_guide, "allow_follow_ups": False}
        c = TextStructuredController(guide)

        review = c.review_turn(model_text, marker, decision, "I work in research.")

        assert review.replaced is True
        assert review.reason == "fixed_question_enforced"
        assert review.text == "Tell me about your typical day."
        assert review.marker == 1
        assert review.decision == TurnDecision(
            action="main", question=1, intent="q1:main"
        )
        assert c.current_question_index == 0

    def test_fixed_mode_allows_a_real_participant_clarification(self, two_question_guide):
        guide = {**two_question_guide, "allow_follow_ups": False}
        c = TextStructuredController(guide)

        review = c.review_turn(
            "By typical day, I mean the kind of workday you experience most often.",
            None,
            TurnDecision(action="clarify", question=1),
            "What do you mean by a typical day?",
        )

        assert review.replaced is False
        assert review.decision is not None
        assert review.decision.action == "clarify"
        assert review.text.startswith("By typical day")
        assert c.current_question_index == 0

    def test_fixed_mode_uses_exact_closing_after_final_answer(self):
        guide = {
            "allow_follow_ups": False,
            "questions": [{"id": "only", "text": "What happened?"}],
            "closing_message": "That completes the survey. Thank you.",
        }
        c = TextStructuredController(guide)
        c.review_turn(
            "What happened?",
            1,
            TurnDecision(action="main", question=1, intent="only:main"),
        )
        c.register_bot_turn("Ready.")
        assert c.maybe_advance_message("Here is my answer.") is not None

        review = c.review_turn(
            "That was fascinating—anything else you want to add?",
            None,
            TurnDecision(action="probe", question=1, intent="only:emergent:meaning"),
            "Here is my answer.",
        )

        assert review.replaced is True
        assert review.reason == "fixed_closing_enforced"
        assert review.text == "That completes the survey. Thank you."
        assert review.closed is True

    def test_adversarial_listed_probe_id_cannot_hide_leading_wording(self):
        """The model may select a safe listed intent; the server owns its text."""
        guide = {
            "questions": [
                {
                    "id": "episode",
                    "text": "What happened?",
                    "probes": ["What happened immediately before that?"],
                    "evidence_goal": ["Sequence"],
                }
            ]
        }
        c = TextStructuredController(guide)
        c.review_turn(
            "What happened?",
            1,
            TurnDecision(action="main", question=1, intent="episode:main"),
        )

        review = c.review_turn(
            "That sounds like the AI failed badly, so why did you trust it?",
            None,
            TurnDecision(action="probe", question=1, intent="episode:p1"),
            "It gave me a result I did not expect.",
        )

        assert review.replaced is True
        assert review.reason == "listed_probe_wording_corrected"
        assert review.text == "What happened immediately before that?"
        assert review.decision is not None
        assert review.decision.intent == "episode:p1"
        assert c.stats()["probe_wording_corrections"] == 1

    def test_adversarial_ordered_mode_blocks_probe_cherry_picking(self):
        guide = {
            "probe_selection": "ordered",
            "questions": [
                {
                    "id": "episode",
                    "text": "What happened?",
                    "probes": ["What happened first?", "What happened next?"],
                }
            ],
        }
        c = TextStructuredController(guide)
        c.review_turn(
            "What happened?",
            1,
            TurnDecision(action="main", question=1, intent="episode:main"),
        )

        review = c.review_turn(
            "What happened next?",
            None,
            TurnDecision(action="probe", question=1, intent="episode:p2"),
        )

        assert review.replaced is True
        assert review.reason == "ordered_probe_corrected"
        assert review.text == "What happened first?"
        assert review.decision is not None
        assert review.decision.intent == "episode:p1"
        assert c.stats()["probe_policy_corrections"] == 1

    def test_relevance_mode_selects_probe_but_cannot_rewrite_it(self):
        guide = {
            "probe_selection": "relevance",
            "questions": [
                {
                    "id": "episode",
                    "text": "What happened?",
                    "probes": ["What happened first?", "What happened next?"],
                }
            ],
        }
        c = TextStructuredController(guide)
        c.review_turn(
            "What happened?",
            1,
            TurnDecision(action="main", question=1, intent="episode:main"),
        )

        review = c.review_turn(
            "Surely that mistake made you angry—what happened next?",
            None,
            TurnDecision(action="probe", question=1, intent="episode:p2"),
        )

        assert review.replaced is True
        assert review.reason == "listed_probe_wording_corrected"
        assert review.text == "What happened next?"
        assert review.decision is not None
        assert review.decision.intent == "episode:p2"

    def test_exploratory_relevance_probe_can_use_grounded_wording(self):
        guide = {
            "orchestration_profile": "exploratory",
            "probe_selection": "relevance",
            "questions": [
                {
                    "id": "episode",
                    "text": "What happened?",
                    "probes": ["How did you assess the result?"],
                    "evidence_goal": ["Basis for assessment"],
                    "max_follow_ups": 1,
                }
            ],
        }
        c = TextStructuredController(guide)
        c.review_turn(
            "What happened?",
            1,
            TurnDecision(action="main", question=1, intent="episode:main"),
        )

        review = c.review_turn(
            "What did you see when you checked one citation against the report?",
            None,
            TurnDecision(action="probe", question=1, intent="episode:p1"),
            "I checked three citations against the reports and two were unsupported.",
        )

        assert review.replaced is False
        assert review.text == (
            "What did you see when you checked one citation against the report?"
        )
        assert c.stats()["adaptive_probe_wording_accepted"] == 1

    def test_exploratory_relevance_probe_still_blocks_leading_wording(self):
        guide = {
            "orchestration_profile": "exploratory",
            "probe_selection": "relevance",
            "questions": [
                {
                    "id": "episode",
                    "text": "What happened?",
                    "probes": ["How did you assess the result?"],
                    "evidence_goal": ["Basis for assessment"],
                    "max_follow_ups": 1,
                }
            ],
        }
        c = TextStructuredController(guide)
        c.review_turn(
            "What happened?",
            1,
            TurnDecision(action="main", question=1, intent="episode:main"),
        )

        review = c.review_turn(
            "Surely the unsupported citations made the assistant untrustworthy?",
            None,
            TurnDecision(action="probe", question=1, intent="episode:p1"),
            "I checked three citations and two were unsupported.",
        )

        assert review.replaced is True
        assert review.reason == "listed_probe_wording_corrected"
        assert review.text == "How did you assess the result?"

    def test_adversarial_emergent_probe_is_blocked_when_disabled(self):
        guide = {
            "allow_emergent_follow_ups": False,
            "questions": [
                {
                    "id": "episode",
                    "text": "What happened?",
                    "probes": ["What did you observe directly?"],
                }
            ],
        }
        c = TextStructuredController(guide)
        c.review_turn(
            "What happened?",
            1,
            TurnDecision(action="main", question=1, intent="episode:main"),
        )

        review = c.review_turn(
            "Why do you think the company designed it to manipulate you?",
            None,
            TurnDecision(
                action="probe", question=1, intent="episode:emergent:meaning"
            ),
        )

        assert review.replaced is True
        assert review.reason == "emergent_probes_disabled"
        assert review.text == "What did you observe directly?"
        assert review.decision is not None
        assert review.decision.intent == "episode:p1"
        assert c.stats()["emergent_probes_blocked"] == 1

    def test_same_question_thread_uses_safe_probe_when_emergent_is_disabled(self):
        guide = {
            "allow_emergent_follow_ups": False,
            "questions": [
                {
                    "id": "episode",
                    "text": "What happened?",
                    "probes": ["What did you observe directly?"],
                }
            ],
        }
        c = TextStructuredController(guide)
        c.review_turn(
            "What happened?",
            1,
            TurnDecision(action="main", question=1, intent="episode:main"),
        )

        review = c.review_turn(
            "Why do you think they intended to deceive you?",
            1,
            TurnDecision(
                action="thread", question=1, intent="episode:emergent:meaning"
            ),
        )

        assert review.replaced is True
        assert review.reason == "emergent_probes_disabled"
        assert review.text == "What did you observe directly?"
        assert review.decision is not None
        assert review.decision.action == "probe"
        assert review.decision.intent == "episode:p1"

    def test_disabled_emergent_probe_without_safe_fallback_advances(self):
        guide = {
            "allow_emergent_follow_ups": False,
            "questions": [
                {"id": "episode", "text": "What happened?", "probes": []}
            ],
            "closing_message": "Thanks.",
        }
        c = TextStructuredController(guide)
        c.review_turn(
            "What happened?",
            1,
            TurnDecision(action="main", question=1, intent="episode:main"),
        )

        review = c.review_turn(
            "Why do you think they wanted to deceive you?",
            None,
            TurnDecision(
                action="probe", question=1, intent="episode:emergent:meaning"
            ),
        )

        assert review.replaced is True
        assert review.reason == "emergent_probes_disabled"
        assert review.text == "Thanks."
        assert review.closed is True

    @pytest.mark.parametrize("invalid_intent", [None, "emergent_deception_cue"])
    def test_invalid_followup_intent_cannot_bypass_probe_policy(
        self, invalid_intent
    ):
        guide = {
            "allow_emergent_follow_ups": False,
            "questions": [
                {"id": "episode", "text": "What happened?", "probes": []}
            ],
            "closing_message": "Thanks.",
        }
        c = TextStructuredController(guide)
        c.review_turn(
            "What happened?",
            1,
            TurnDecision(action="main", question=1, intent="episode:main"),
        )

        review = c.review_turn(
            "What made you think they wanted to deceive you?",
            1,
            TurnDecision(
                action="probe", question=1, intent=invalid_intent
            ),
        )

        assert review.replaced is True
        assert review.reason == "invalid_followup_intent"
        assert review.text == "Thanks."
        assert review.closed is True
        assert c.stats()["probe_policy_corrections"] == 1

    def test_emergent_probe_remains_available_when_explicitly_enabled(self):
        guide = {
            "allow_emergent_follow_ups": True,
            "questions": [
                {"id": "episode", "text": "What happened?", "probes": []}
            ],
        }
        c = TextStructuredController(guide)
        c.review_turn(
            "What happened?",
            1,
            TurnDecision(action="main", question=1, intent="episode:main"),
        )

        review = c.review_turn(
            "What did 'unreliable' mean in that particular result?",
            None,
            TurnDecision(
                action="probe", question=1, intent="episode:emergent:clarification"
            ),
        )

        assert review.replaced is False
        assert review.text == "What did 'unreliable' mean in that particular result?"
        assert c.stats()["emergent_probes_blocked"] == 0

    def test_no_nudge_within_budget(self, two_question_guide):
        c = TextStructuredController(two_question_guide)
        # Q1 budget = 1 + max_follow_ups(2) = 3. Two counted turns is fine.
        for t in ("I work in research.", "Mostly data analysis."):
            assert c.maybe_advance_message(t) is None
            c.register_bot_turn(t)
        assert c.snapshot()["nudge_pending"] is False
        assert c.current_question_index == 0

    def test_advances_after_budget_exhausted(self, two_question_guide):
        c = TextStructuredController(two_question_guide)
        for t in ("answer one", "answer two", "answer three"):
            assert c.maybe_advance_message(t) is None
            c.register_bot_turn(t)
        assert c.snapshot()["nudge_pending"] is True
        msg = c.maybe_advance_message("answer four")
        assert msg is not None
        # Provider-safe user-role guidance note.
        assert msg["role"] == "user"
        assert "question 2" in msg["content"]
        assert "biggest pain point" in msg["content"]
        assert "Great, let's switch gears." in msg["content"]
        assert c.current_question_index == 1

    def test_clarification_reply_does_not_burn_budget(self, two_question_guide):
        c = TextStructuredController(two_question_guide)
        c.register_bot_turn("I work in research.")  # counted
        c.register_bot_turn("What do you mean by a concrete example?")  # skipped
        c.register_bot_turn("Sorry, I didn't get the whole question.")  # skipped
        assert c.snapshot()["bot_turns_on_question"] == 1
        assert c.snapshot()["nudge_pending"] is False

    def test_nudge_is_held_while_participant_asks_to_repeat(
        self, two_question_guide
    ):
        c = TextStructuredController(two_question_guide)
        for t in ("answer one", "answer two", "answer three"):
            c.register_bot_turn(t)
        assert c.snapshot()["nudge_pending"] is True
        # A clarification must not trigger the advance.
        assert c.maybe_advance_message("Sorry, could you repeat that?") is None
        assert c.current_question_index == 0
        assert c.snapshot()["nudge_pending"] is True
        # The next substantive turn advances.
        assert c.maybe_advance_message("Here is my real answer.") is not None
        assert c.current_question_index == 1

    def test_procedural_curveball_cannot_advance_even_with_bad_model_marker(
        self, two_question_guide
    ):
        c = TextStructuredController(two_question_guide)
        for answer in ("answer one", "answer two", "answer three"):
            c.register_bot_turn(answer)
        assert c.snapshot()["nudge_pending"] is True

        question = "Is this recorded?"
        assert c.maybe_advance_message(question) is None
        review = c.review_turn(
            "I don't know. What's your biggest pain point right now?",
            2,
            TurnDecision(action="main", question=2, intent="q2:main"),
            question,
        )
        c.register_bot_turn(question)

        assert review.marker is None
        assert review.decision.action == "clarify"
        assert c.current_question_index == 0
        assert c.snapshot()["nudge_pending"] is True
        assert c.snapshot()["bot_turns_on_question"] == 3

    def test_flow_repair_returns_to_exact_suspended_probe(
        self, two_question_guide
    ):
        c = TextStructuredController(two_question_guide)
        c.review_turn(
            "What part of this morning was hardest?",
            None,
            None,
        )
        response = participant_interruption_response(
            "You moved on before I answered.",
            {},
            c.current_prompt_text,
        )
        assert response.endswith("What part of this morning was hardest?")

    def test_cutoff_fragment_does_not_burn_followup_budget(
        self, two_question_guide
    ):
        c = TextStructuredController(two_question_guide)
        c.register_bot_turn("A complete answer.")
        c.register_bot_turn("I mean, I think...")
        assert c.snapshot()["bot_turns_on_question"] == 1

    def test_rich_warmup_answer_can_move_to_focused_next_question(self):
        guide = {
            "questions": [
                {
                    "id": "context",
                    "text": "What work have you used AI for?",
                    "evidence_goal": ["Work type", "Frequency"],
                    "probes": ["What do you avoid using it for?"],
                },
                {
                    "id": "success",
                    "text": "What recent task went well?",
                    "evidence_goal": ["Task", "Contribution"],
                    "probes": ["What did AI contribute?"],
                },
            ]
        }
        c = TextStructuredController(guide)
        c.review_turn(
            "What do you avoid using it for?",
            None,
            TurnDecision(
                action="probe",
                question=1,
                intent="context:p1",
                resolved={
                    "context:e1": "covered",
                    "success:e1": "partial",
                },
            ),
            "I used it to build a website, but I keep final claims human-led.",
        )
        review = c.review_turn(
            "What did AI contribute to the website?",
            None,
            TurnDecision(
                action="probe",
                question=2,
                intent="success:p1",
                resolved={"context:e1": "covered", "success:e1": "partial"},
            ),
            "I keep the interpretation for myself.",
        )
        assert review.replaced is True
        assert review.reason == "listed_probe_wording_corrected"
        assert review.text == "What did AI contribute?"
        assert c.current_question_index == 1

    def test_partial_early_answer_replaces_broad_main_with_listed_probe(self):
        guide = {
            "questions": [
                {
                    "id": "success",
                    "text": "Tell me about a success.",
                    "evidence_goal": ["Task", "Outcome"],
                    "probes": ["What changed because of it?"],
                }
            ]
        }
        c = TextStructuredController(guide)
        review = c.review_turn(
            "Tell me about a success.",
            1,
            TurnDecision(
                action="main",
                question=1,
                intent="success:main",
                resolved={"success:e1": "partial"},
            ),
            "AI helped me build a website.",
        )
        assert review.replaced is True
        assert review.reason == "broad_main_after_partial_answer"
        assert review.text == "What changed because of it?"
        assert review.decision.action == "probe"

    def test_next_main_can_advance_after_focused_probe_without_replaying_main(self):
        guide = {
            "questions": [
                {
                    "id": "success",
                    "text": "Tell me about a success.",
                    "evidence_goal": ["Task", "Outcome"],
                    "probes": ["What changed because of it?"],
                },
                {
                    "id": "failure",
                    "text": "Tell me about a failure.",
                    "evidence_goal": ["Failure"],
                    "probes": [],
                },
            ]
        }
        c = TextStructuredController(guide)
        c.review_turn(
            "What changed because of it?",
            None,
            TurnDecision(
                action="probe",
                question=1,
                intent="success:p1",
                resolved={"success:e1": "partial"},
            ),
            "AI helped me build a website.",
        )
        review = c.review_turn(
            "Tell me about a failure.",
            2,
            TurnDecision(
                action="main",
                question=2,
                intent="failure:main",
                resolved={"success:e2": "partial"},
            ),
            "It gave us something concrete to review.",
        )
        assert review.replaced is False
        assert review.text == "Tell me about a failure."
        assert c.current_question_index == 1

    def test_clarification_cannot_bridge_into_the_next_guide_card(self):
        guide = {
            "questions": [
                {
                    "id": "judgment",
                    "text": "Which decision did you keep for yourself?",
                    "evidence_goal": ["Decision", "Reason"],
                    "probes": ["What made you keep that decision for yourself?"],
                },
                {
                    "id": "future",
                    "text": "What should change in the future?",
                    "evidence_goal": ["Desired change"],
                    "probes": [],
                },
            ]
        }
        c = TextStructuredController(guide)
        review = c.review_turn(
            "I meant both. What should change in the future?",
            2,
            TurnDecision(
                action="main",
                question=2,
                intent="future:main",
            ),
            "Do you mean brainstorming a question or actually choosing it?",
        )
        assert review.replaced is True
        assert review.reason == "clarification_advance_blocked"
        assert review.marker is None
        assert review.text == "What made you keep that decision for yourself?"
        assert "future" not in review.text.lower()
        assert c.current_question_index == 0

    def test_closes_after_final_question(self, two_question_guide):
        c = TextStructuredController(two_question_guide)
        for t in ("a1", "a2", "a3"):
            c.register_bot_turn(t)
        assert c.maybe_advance_message("real answer") is not None  # → Q2
        c.register_bot_turn("real answer")  # Q2 turn 1
        c.register_bot_turn("more detail")  # Q2 turn 2 → budget hit
        assert c.snapshot()["nudge_pending"] is True
        msg = c.maybe_advance_message("final answer")
        assert msg is not None
        assert "completed all the questions" in msg["content"]
        assert "Thanks for your time." in msg["content"]
        assert c.is_finished is True
        # A closed controller no longer counts turns.
        c.register_bot_turn("anything")
        assert c.snapshot()["bot_turns_on_question"] == 2

    def test_marker_sync_is_forward_only(self, two_question_guide):
        c = TextStructuredController(two_question_guide)
        c.sync_to_marker(2)
        assert c.current_question_index == 1
        c.sync_to_marker(1)  # backward, ignored
        assert c.current_question_index == 1
        c.sync_to_marker(None)  # no marker, ignored
        assert c.current_question_index == 1
        c.sync_to_marker(99)  # out of range, ignored
        assert c.current_question_index == 1

    def test_marker_sync_clears_pending_nudge(self, two_question_guide):
        c = TextStructuredController(two_question_guide)
        for t in ("a1", "a2", "a3"):
            c.register_bot_turn(t)
        assert c.snapshot()["nudge_pending"] is True
        # The model advanced on its own (declared Q2), so trust it.
        c.sync_to_marker(2)
        assert c.current_question_index == 1
        assert c.snapshot()["nudge_pending"] is False

    def test_stats_reports_text_channel(self, two_question_guide):
        c = TextStructuredController(two_question_guide)
        stats = c.stats()
        assert stats["channel"] == "text"
        assert stats["total_questions"] == 2
        assert stats["completed"] is False
        assert stats["forced_advances"] == 0

    def test_resolved_question_probe_is_replaced_with_next_main(self):
        guide = {
            "questions": [
                {
                    "id": "episode",
                    "text": "What happened?",
                    "evidence_goal": ["Event sequence"],
                    "probes": ["What happened next?"],
                },
                {
                    "id": "impact",
                    "text": "What difference did that make?",
                    "evidence_goal": ["Consequence"],
                    "probes": [],
                },
            ],
            "closing_message": "Thanks.",
        }
        c = TextStructuredController(guide)
        first = c.review_turn(
            "What happened?",
            1,
            TurnDecision(action="main", question=1, intent="episode:main"),
        )
        assert first.replaced is False
        review = c.review_turn(
            "What happened next?",
            None,
            TurnDecision(
                action="probe",
                question=1,
                intent="episode:p1",
                resolved={"episode:e1": "covered"},
            ),
        )
        assert review.replaced is True
        assert review.reason == "question_resolved"
        assert review.text == "What difference did that make?"
        assert review.marker == 2
        assert c.current_question_index == 1
        assert c.stats()["resolved_probes_blocked"] == 1

    def test_unknown_and_not_applicable_are_terminal_evidence_states(self):
        guide = {
            "questions": [
                {
                    "id": "source",
                    "text": "How did you verify it?",
                    "evidence_goal": ["Verification basis", "Hidden cause"],
                    "probes": ["What caused it?"],
                }
            ],
            "closing_message": "Thanks.",
        }
        c = TextStructuredController(guide)
        review = c.review_turn(
            "What caused it?",
            None,
            TurnDecision(
                action="probe",
                question=1,
                intent="source:p1",
                resolved={
                    "source:e1": "covered",
                    "source:e2": "unknown",
                },
            ),
            "I checked the source, but I don't know what caused it.",
        )
        assert review.replaced is True
        assert review.closed is True
        assert review.text == "Thanks."
        stats = c.stats()
        assert stats["resolution_status_counts"]["covered"] == 1
        assert stats["resolution_status_counts"]["unknown"] == 1

    def test_explicit_no_concern_can_end_the_concern_branch(self):
        guide = {
            "questions": [
                {
                    "id": "concerns",
                    "text": "What, if anything, worries you?",
                    "evidence_goal": [
                        "Current concern",
                        "Change in concern",
                        "Concern-related event",
                    ],
                    "probes": ["Can you think of a specific time?"],
                },
                {
                    "id": "positive",
                    "text": "What feels helpful?",
                    "evidence_goal": ["Helpful use"],
                },
            ]
        }
        c = TextStructuredController(guide)
        review = c.review_turn(
            "What feels helpful?",
            2,
            TurnDecision(
                action="main",
                question=2,
                intent="positive:main",
                resolved={
                    "concerns:e1": "not_applicable",
                    "concerns:e2": "not_applicable",
                    "concerns:e3": "not_applicable",
                },
            ),
            "I don't have any concerns about it right now.",
        )

        assert review.replaced is False
        assert review.text == "What feels helpful?"
        assert c.stats()["resolution_status_counts"]["not_applicable"] == 3

    def test_unmentioned_future_evidence_cannot_be_marked_unknown(self):
        guide = {
            "questions": [
                {
                    "id": "intro",
                    "text": "What do you use?",
                    "evidence_goal": ["Current use"],
                },
                {
                    "id": "episode",
                    "text": "What happened recently?",
                    "evidence_goal": ["Event", "Outcome"],
                },
            ]
        }
        c = TextStructuredController(guide)
        review = c.review_turn(
            "What happened recently?",
            2,
            TurnDecision(
                action="main",
                question=2,
                intent="episode:main",
                resolved={
                    "intro:e1": "covered",
                    "episode:e1": "unknown",
                    "episode:e2": "unknown",
                },
            ),
            "I use several tools for research and drafting.",
        )
        assert review.replaced is False
        assert review.text == "What happened recently?"
        stats = c.stats()
        assert stats["terminal_resolutions_rejected"] == 2
        assert stats["resolution_status_counts"]["unknown"] == 0

    def test_duplicate_intent_is_blocked_even_when_rephrased(self):
        guide = {
            "questions": [
                {"id": "one", "text": "First?", "probes": ["Example?"]},
                {"id": "two", "text": "Second?", "probes": []},
            ]
        }
        c = TextStructuredController(guide)
        c.review_turn(
            "Can you share an example?",
            None,
            TurnDecision(action="probe", question=1, intent="one:p1"),
        )
        review = c.review_turn(
            "Could you give me a specific instance?",
            None,
            TurnDecision(action="probe", question=1, intent="one:p1"),
        )
        assert review.replaced is True
        assert review.reason == "duplicate_intent"
        assert review.text == "Second?"
        assert c.stats()["duplicate_intents_blocked"] == 1

    def test_unresolved_questions_cannot_be_skipped_or_revisited(self):
        guide = {
            "questions": [
                {"id": "one", "text": "First?", "evidence_goal": ["One"]},
                {"id": "two", "text": "Second?", "evidence_goal": ["Two"]},
                {"id": "three", "text": "Third?", "evidence_goal": ["Three"]},
            ]
        }
        c = TextStructuredController(guide)
        c.review_turn(
            "First?", 1, TurnDecision(action="main", question=1, intent="one:main")
        )
        jumped = c.review_turn(
            "Third?",
            3,
            TurnDecision(
                action="main",
                question=3,
                intent="three:main",
                resolved={"one:e1": "covered"},
            ),
            "Here is the first answer.",
        )
        assert jumped.replaced is True
        assert jumped.reason == "skipped_unresolved_question"
        assert jumped.text == "Second?"
        assert jumped.marker == 2

        backward = c.review_turn(
            "First again?",
            1,
            TurnDecision(action="main", question=1, intent="one:main"),
        )
        assert backward.replaced is True
        assert backward.reason == "backward_question"
        assert backward.text == "Third?"
        assert c.stats()["order_corrections"] == 2

    @staticmethod
    def _missing_decision_recovery_guide():
        return {
            "questions": [
                {"id": "context", "text": "What were you doing?", "evidence_goal": ["Context"]},
                {"id": "episode", "text": "What happened?", "evidence_goal": ["Episode"]},
                {"id": "impact", "text": "What changed?", "evidence_goal": ["Impact"]},
                {
                    "id": "close",
                    "text": "What have I missed?",
                    "phase": "close",
                    "evidence_goal": ["Final addition"],
                },
            ],
            "closing_message": "Thanks.",
        }

    def test_substantive_missing_marker_does_not_force_broad_replay(self):
        c = TextStructuredController(self._missing_decision_recovery_guide())
        c.review_turn(
            "What specifically changed your trust in the result?",
            None,
            None,
            (
                "I used it for a report, found an unsupported citation, checked "
                "the sources, switched tools, and now verify each important "
                "claim before I share the final document with anyone else."
            ),
        )

        recovered = c.review_turn(
            "What have I missed?",
            4,
            TurnDecision(
                action="main",
                question=4,
                intent="close:main",
                resolved={"episode:e1": "covered", "impact:e1": "covered"},
            ),
            (
                "Seeing the evidence beside the claim mattered most because it "
                "let me distinguish direct support from a merely related source "
                "before deciding whether I could trust and reuse the output."
            ),
        )

        assert recovered.replaced is False
        assert recovered.marker == 4
        assert recovered.text == "What have I missed?"
        assert c.current_question_index == 3
        assert c.stats()["missing_decision_recoveries"] == 1

    def test_short_missing_marker_does_not_bypass_unresolved_questions(self):
        c = TextStructuredController(self._missing_decision_recovery_guide())
        c.review_turn("Ready?", None, None, "Yes, ready.")

        review = c.review_turn(
            "What have I missed?",
            4,
            TurnDecision(
                action="main",
                question=4,
                intent="close:main",
                resolved={"episode:e1": "covered", "impact:e1": "covered"},
            ),
            (
                "Here is a detailed answer about what happened, what changed, "
                "and what I would want the product team to understand next."
            ),
        )

        assert review.replaced is True
        assert review.reason == "skipped_unresolved_question"
        assert review.text == "What were you doing?"
        assert c.stats()["missing_decision_recoveries"] == 0

    @staticmethod
    def _threading_guide():
        return {
            "allow_emergent_follow_ups": True,
            "questions": [
                {
                    "id": "context",
                    "text": "What were you trying to do?",
                    "evidence_goal": ["Initial purpose"],
                    "probes": ["What made that important then?"],
                },
                {
                    "id": "episode",
                    "text": "What happened in the interaction?",
                    "evidence_goal": ["Interaction sequence"],
                    "probes": ["What did the tool say or do?"],
                },
                {
                    "id": "impact",
                    "text": "What effect did that have?",
                    "evidence_goal": ["Immediate consequence", "Participant response"],
                    "probes": ["What did you do next?"],
                },
                {
                    "id": "sensitive",
                    "text": "How did this affect your health?",
                    "evidence_goal": ["Health consequence"],
                    "probes": ["What changed for you?"],
                    "sensitivity": "elevated",
                },
            ],
            "closing_message": "Thanks.",
        }

    def test_answer_grounded_thread_preserves_cursor_then_resumes(self):
        c = TextStructuredController(self._threading_guide())
        c.review_turn(
            "What were you trying to do?",
            1,
            TurnDecision(action="main", question=1, intent="context:main"),
        )

        threaded = c.review_turn(
            "You mentioned closing the chat immediately. What led you to do that?",
            None,
            TurnDecision(
                action="thread",
                question=3,
                intent="impact:p1",
                resolved={"impact:e2": "partial"},
            ),
            "I was checking a symptom, but the answer scared me so I closed it.",
        )

        assert threaded.replaced is True
        assert threaded.reason == "listed_probe_wording_corrected"
        assert threaded.text == "What did you do next?"
        assert threaded.marker is None
        assert threaded.question_index == 0
        assert c.current_question_index == 0
        assert c.stats()["threaded_probes_accepted"] == 1
        assert c.stats()["thread_pending"] is True

        resumed = c.review_turn(
            "What made checking that symptom important then?",
            None,
            TurnDecision(
                action="probe",
                question=1,
                intent="context:p1",
                resolved={"impact:e1": "covered", "impact:e2": "covered"},
            ),
            "I called a nurse line instead because I was worried.",
        )

        assert resumed.replaced is True
        assert resumed.reason == "listed_probe_wording_corrected"
        assert resumed.text == "What made that important then?"
        assert c.current_question_index == 0
        assert c.stats()["thread_pending"] is False
        assert c.stats()["thread_resumes"] == 1

    def test_rich_readiness_answer_can_thread_after_resolving_opening(self):
        c = TextStructuredController(self._threading_guide())

        review = c.review_turn(
            "You said you closed it immediately. What made you switch away?",
            None,
            TurnDecision(
                action="thread",
                question=3,
                intent="impact:p1",
                resolved={"context:e1": "covered", "impact:e2": "partial"},
            ),
            "I was checking a symptom, but the answer scared me so I closed it.",
        )

        assert review.replaced is True
        assert review.reason == "listed_probe_wording_corrected"
        assert review.text == "What did you do next?"
        assert review.question_index == 0
        assert c.current_question_index == 0
        assert c.stats()["threaded_probes_accepted"] == 1
        assert c.stats()["thread_pending"] is True

    def test_rich_bundle_can_take_one_depth_thread_on_a_resolved_card(self):
        c = TextStructuredController(self._threading_guide())

        review = c.review_turn(
            "What about the answer made you close the chat immediately?",
            None,
            TurnDecision(
                action="thread",
                question=3,
                intent="impact:p1",
                resolved={
                    "context:e1": "covered",
                    "episode:e1": "covered",
                    "impact:e1": "covered",
                    "impact:e2": "covered",
                },
            ),
            (
                "I checked a symptom in an assistant, saw a severe diagnosis, "
                "closed it, called a nurse, and now verify health answers before "
                "acting on them because the experience frightened me."
            ),
        )

        assert review.replaced is True
        assert review.reason == "listed_probe_wording_corrected"
        assert review.text == "What did you do next?"
        assert review.marker is None
        assert c.stats()["threaded_probes_accepted"] == 1
        assert c.stats()["depth_threads_accepted"] == 1

    def test_resolved_card_cannot_be_threaded_without_a_rich_bundle(self):
        c = TextStructuredController(self._threading_guide())
        c.review_turn(
            "What were you trying to do?",
            1,
            TurnDecision(action="main", question=1, intent="context:main"),
        )

        review = c.review_turn(
            "What did you do next?",
            None,
            TurnDecision(
                action="thread",
                question=3,
                intent="impact:p1",
                resolved={"impact:e1": "covered", "impact:e2": "covered"},
            ),
            (
                "I closed the tool and took a short break before returning to "
                "the task later, but I have not explained the original task or "
                "the interaction itself yet."
            ),
        )

        assert review.replaced is True
        assert review.reason == "thread_target_resolved"
        assert c.stats()["depth_threads_accepted"] == 0

    def test_vague_topic_hop_cannot_justify_thread(self):
        c = TextStructuredController(self._threading_guide())
        c.review_turn(
            "What were you trying to do?",
            1,
            TurnDecision(action="main", question=1, intent="context:main"),
        )

        review = c.review_turn(
            "What did you do next?",
            None,
            TurnDecision(action="thread", question=3, intent="impact:p1"),
            "There was the answer, and the impact, and privacy too—anyway, lots happened.",
        )

        assert review.replaced is True
        assert review.reason == "thread_without_target_evidence"
        assert review.text == "What happened in the interaction?"
        assert review.marker == 2
        assert c.stats()["threading_blocked"] == 1

    def test_thread_cannot_chain_when_participant_keeps_topic_hopping(self):
        c = TextStructuredController(self._threading_guide())
        c.review_turn(
            "What were you trying to do?",
            1,
            TurnDecision(action="main", question=1, intent="context:main"),
        )
        c.review_turn(
            "What did you do next?",
            None,
            TurnDecision(
                action="thread",
                question=3,
                intent="impact:p1",
                resolved={"impact:e2": "partial"},
            ),
            "I stopped using it and did something else.",
        )

        chained = c.review_turn(
            "What changed for you?",
            None,
            TurnDecision(
                action="thread",
                question=4,
                intent="sensitive:p1",
                resolved={"sensitive:e1": "partial"},
            ),
            "Then there was my health, and privacy, and a different app.",
        )

        assert chained.replaced is True
        assert chained.reason == "thread_chain"
        assert chained.text == "What happened in the interaction?"
        assert c.current_question_index == 1
        stats = c.stats()
        assert stats["threaded_probes_accepted"] == 1
        assert stats["threading_blocked"] == 1
        assert stats["thread_resumes"] == 1

    def test_elevated_downstream_question_cannot_be_threaded_early(self):
        c = TextStructuredController(self._threading_guide())
        c.review_turn(
            "What were you trying to do?",
            1,
            TurnDecision(action="main", question=1, intent="context:main"),
        )

        review = c.review_turn(
            "What changed for you?",
            None,
            TurnDecision(
                action="thread",
                question=4,
                intent="sensitive:p1",
                resolved={"sensitive:e1": "partial"},
            ),
            "It also affected my health, though I do not want to get into that yet.",
        )

        assert review.replaced is True
        assert review.reason == "thread_unsafe_target"
        assert review.text == "What happened in the interaction?"

    def test_partial_thread_does_not_skip_later_question(self):
        c = TextStructuredController(self._threading_guide())
        c.review_turn(
            "What were you trying to do?",
            1,
            TurnDecision(action="main", question=1, intent="context:main"),
        )
        c.review_turn(
            "What did you do next?",
            None,
            TurnDecision(
                action="thread",
                question=3,
                intent="impact:p1",
                resolved={"context:e1": "covered", "impact:e2": "partial"},
            ),
        )
        c.review_turn(
            "What happened in the interaction?",
            2,
            TurnDecision(action="main", question=2, intent="episode:main"),
        )

        attempted_jump = c.review_turn(
            "How did this affect your health?",
            4,
            TurnDecision(
                action="main",
                question=4,
                intent="sensitive:main",
                resolved={"episode:e1": "covered"},
            ),
        )

        assert attempted_jump.replaced is True
        assert attempted_jump.reason == "skipped_unresolved_question"
        assert attempted_jump.text == "What effect did that have?"
        assert attempted_jump.marker == 3

    def test_fully_covered_downstream_question_is_skipped_after_thread(self):
        c = TextStructuredController(self._threading_guide())
        c.review_turn(
            "What were you trying to do?",
            1,
            TurnDecision(action="main", question=1, intent="context:main"),
        )
        c.review_turn(
            "What did you do next?",
            None,
            TurnDecision(
                action="thread",
                question=3,
                intent="impact:p1",
                resolved={"context:e1": "covered", "impact:e2": "partial"},
            ),
        )
        c.review_turn(
            "What happened in the interaction?",
            2,
            TurnDecision(
                action="main",
                question=2,
                intent="episode:main",
                resolved={"impact:e1": "covered", "impact:e2": "covered"},
            ),
        )

        skipped = c.review_turn(
            "How did this affect your health?",
            4,
            TurnDecision(
                action="main",
                question=4,
                intent="sensitive:main",
                resolved={"episode:e1": "covered"},
            ),
        )

        assert skipped.replaced is False
        assert skipped.question_index == 3
        assert c.current_question_index == 3
        assert c.stats()["covered_questions_skipped"] >= 1

    def test_partial_downstream_card_accepts_narrow_focused_advance(self):
        c = TextStructuredController(self._threading_guide())
        c.review_turn(
            "What were you trying to do?",
            1,
            TurnDecision(action="main", question=1, intent="context:main"),
        )
        c.review_turn(
            "You said you closed it. What did you do next?",
            None,
            TurnDecision(
                action="thread",
                question=3,
                intent="impact:p1",
                resolved={"context:e1": "covered", "impact:e2": "partial"},
            ),
            "I was checking a symptom, but the answer scared me so I closed it.",
        )
        c.review_turn(
            "What happened in the interaction?",
            2,
            TurnDecision(action="main", question=2, intent="episode:main"),
        )

        focused = c.review_turn(
            "What about the answer made it feel alarming?",
            None,
            TurnDecision(
                action="probe",
                question=3,
                intent="impact:emergent:meaning",
                resolved={"episode:e1": "covered", "impact:e1": "partial"},
            ),
            "It gave me a severe diagnosis without explaining how likely it was.",
        )

        assert focused.replaced is False
        assert focused.text == "What about the answer made it feel alarming?"
        assert focused.marker == 3
        assert focused.question_index == 2
        assert c.current_question_index == 2
        stats = c.stats()
        assert stats["focused_advances_accepted"] == 1
        assert stats["thread_resumes"] == 1
        assert stats["thread_pending"] is False

    def test_time_limited_final_question_skips_card_already_asked(
        self, two_question_guide
    ):
        c = TextStructuredController(two_question_guide)
        c.register_bot_turn("yes")

        final = c.begin_time_limited_final_question()

        assert final == ("What's your biggest pain point right now?", 2)
        assert c.current_question_index == 1

    def test_time_close_marks_protocol_complete(self, two_question_guide):
        c = TextStructuredController(two_question_guide)

        closing = c.close_for_time()

        assert closing == two_question_guide["closing_message"]
        assert c.is_finished is True
        assert c.stats()["completed"] is True
