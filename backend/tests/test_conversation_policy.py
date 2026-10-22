"""Decision and delivery regressions for the contextual pilot policy."""
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from app.pipeline.conversation_policy import (
    ContextualInterviewController, ConversationRepair, Observation, Opportunity,
    TopicUpdate, QuestionCheck, ModuleBoundary,
)
from app.pipeline.interview_guide import TurnDecision


def controller(topic="recent_encounter", answer="I was learning about the Roman empire"):
    guide = json.loads((Path(__file__).parents[1] / "studies/anthropic_voices_v01/claude_control.json").read_text())["agent"]["interview_guide"]
    c = ContextualInterviewController(guide)
    index = c.ids[topic]
    for s in c.states[:index + 1]:
        s.asked = True
    c.current_question_index = c._last_question_index = index
    c._last_text = c.main_wording(index)
    c.note_answer(answer, 6)
    return c


def observation(c, *, coverage="adequate", purpose="Why they chose AI to learn history", priority="high", opportunities=True):
    h = c.history[-1]
    return Observation(
        updates=[TopicUpdate(topic_id=h["topic_id"], coverage=coverage, quote=h["answer"],
                             meaning="They used AI to learn history.", exploration="useful" if opportunities else "settled", correction=False)],
        opportunities=[Opportunity(topic_id=h["topic_id"], move="motivation", purpose=purpose,
                                   anchor_sequence=h["sequence"], anchor_quote=h["answer"], priority=priority)] if opportunities else [],
        ai_use="unknown", ai_use_quote="", chatbot_use="unclear", chatbot_use_quote="", chatbot_use_sequence=None, boundaries=[],
    )


def probe_move(c):
    move = next(m for m in c.eligible_moves() if m["action"] == "probe")
    return move, TurnDecision("probe", move["question"], move["intent"])


def test_covered_roman_history_can_receive_a_new_motivation_probe():
    c = controller()
    c.apply_observation(observation(c))
    move, decision = probe_move(c)
    text = "What made you choose AI for that?"
    assert c.validate_turn(text, decision, c.history[-1]["answer"]) == move
    review = c.accept_turn(text, decision, move)
    assert review.text == text
    assert c.states[1].coverage == "adequate"
    assert c.states[1].follow_ups == 1


def test_high_value_opportunity_prevents_premature_topic_jump():
    c = controller()
    c.apply_observation(observation(c))
    move = next(m for m in c.eligible_moves() if m["action"] == "main")
    with pytest.raises(ConversationRepair, match="valuable"):
        c.validate_turn(move["wording"], TurnDecision("main", move["question"], move["intent"]), c.history[-1]["answer"])


def test_rich_answer_moves_on_without_a_probe_quota():
    c = controller(answer="I learned history with AI because I could ask follow-ups. It explained clearly and I checked the facts in a book.")
    c.apply_observation(observation(c, opportunities=False))
    assert [m["action"] for m in c.eligible_moves()] == ["main"]
    assert c.eligible_moves()[0]["intent"] == "vision:main"


def test_observer_receives_actual_delivered_question_and_prior_exchange():
    c = controller()
    c._last_text = "What made you choose AI for that?"
    c.note_answer("I could ask follow-up questions.", 8)
    context = c.observer_context()
    assert context["actual_question"] == c._last_text
    assert context["conversation"][0]["answer"] == "I was learning about the Roman empire"
    assert context["latest_answer"] == "I could ask follow-up questions."


@pytest.mark.parametrize("coverage", ["unknown", "declined", "not_applicable"])
def test_participant_limit_blocks_probes_even_if_observer_suggests_one(coverage):
    c = controller(topic="vision_progress", answer="I don't know.")
    c.apply_observation(observation(c, coverage=coverage))
    assert all(m["action"] != "probe" for m in c.eligible_moves())


def test_uncertainty_is_not_spread_to_future_topics():
    c = controller(topic="vision_progress", answer="I don't know.")
    obs = observation(c, coverage="unknown")
    obs.updates[0].topic_id = "company_trust"
    c.apply_observation(obs)
    assert c.states[c.ids["company_trust"]].coverage == "missing"


def test_fabricated_source_is_rejected():
    c = controller()
    obs = observation(c)
    obs.updates[0].quote = "I do not use AI"
    obs.opportunities[0].anchor_quote = "AI disappointed me"
    c.apply_observation(obs)
    assert c.states[1].coverage == "missing"
    assert c.opportunities == {}
    assert c.rejected_sources == 2


def test_explicit_correction_can_reopen_coverage_but_unrelated_update_cannot():
    c = controller()
    c.apply_observation(observation(c, opportunities=False))
    c.note_answer("Actually I misremembered. It was my friend using it, not me.", 8)
    obs = observation(c, coverage="partial", opportunities=False)
    c.apply_observation(obs)
    assert c.states[1].coverage == "adequate"
    obs.updates[0].correction = True
    c.apply_observation(obs)
    assert c.states[1].coverage == "partial"
    assert c.states[1].sources[-1]["correction"]


@pytest.mark.parametrize("limit", ["topic", "session", "time"])
def test_budgets_reserve_later_topics_without_claiming_coverage(limit):
    c = controller(answer="Today")
    c.apply_observation(observation(c, coverage="partial", purpose="What they used AI to do"))
    if limit == "topic": c.states[1].follow_ups = 3
    if limit == "session": c.follow_ups = 10
    if limit == "time": c.remaining_seconds = 60
    assert all(m["action"] != "probe" for m in c.eligible_moves())
    assert c.states[1].coverage == "partial"


def test_nonuser_anchor_uses_exposure_wording():
    c = controller(topic="everyday_use", answer="No, I don't use AI tools.")
    assert c.ai_use == "no_ai_use"
    assert "hear" in c.eligible_moves()[0]["wording"].lower()


def test_no_thread_into_sensitive_topic():
    c = controller()
    c.questions[4]["sensitivity"] = "elevated"
    obs = observation(c)
    obs.opportunities[0].topic_id = c.questions[4]["id"]
    c.apply_observation(obs)
    assert not c.opportunities


def test_thread_answer_is_attributed_to_threaded_topic_and_cannot_chain():
    c = controller(answer="I learned history, but I worry AI could enable bioweapons.")
    obs = observation(c)
    obs.opportunities[0].topic_id = "values_conflict"
    c.states[c.ids["values_conflict"]].coverage = "partial"
    c.apply_observation(obs)
    move = next(m for m in c.eligible_moves() if m["action"] == "thread")
    c.accept_turn("What brought that concern to mind?", TurnDecision("thread", move["question"], move["intent"]), move)
    c.note_answer("I don't know.", 8)
    assert c.history[-1]["topic_id"] == "values_conflict"
    c.apply_observation(observation(c, coverage="unknown"))
    assert c.states[c.ids["values_conflict"]].coverage == "unknown"
    assert c.current_question_index == c.ids["recent_encounter"]
    assert not any(m["action"] == "thread" for m in c.eligible_moves())


def test_observer_failure_keeps_coverage_missing_and_does_not_repeat_current_anchor():
    c = controller()
    c.observation_failed()
    review = c.fallback_turn("observer unavailable")
    assert review.decision.intent == "vision:main"
    assert c.states[1].coverage == "missing"
    assert c.observer_failures == 1
    assert c.states[1].stopped_reason == "generation_failure_incomplete"


def test_unknown_optional_detail_preserves_hope_but_ends_probing():
    c = controller(topic="vision", answer="I want AI to cure diseases.")
    c.apply_observation(observation(c))
    c.note_answer("I don't know how it would do that.", 8)
    c.apply_observation(observation(c, coverage="unknown"))
    assert c.states[c.ids["vision"]].coverage == "adequate"
    assert c.states[c.ids["vision"]].exploration == "participant_limit"
    assert not any(m["action"] == "probe" for m in c.eligible_moves())


def test_thread_respects_anchor_dependencies():
    c = controller()
    c.questions[c.ids["vision_progress"]]["depends_on"] = ["vision"]
    obs = observation(c)
    obs.opportunities[0].topic_id = "vision_progress"
    c.states[c.ids["vision_progress"]].coverage = "partial"
    c.apply_observation(obs)
    assert not c.opportunities


def test_clarification_does_not_consume_budget_or_change_active_question():
    c = controller()
    original = c.current_prompt_text
    d = TurnDecision("clarify", 2, None)
    move = c.validate_turn("I mean tools such as AI chatbots. Shall we continue?", d, "What do you mean by AI?")
    c.accept_turn("I mean AI chatbots. Shall we continue?", d, move)
    assert c.current_prompt_text == original
    assert c.follow_ups == 0


def test_time_limit_leaves_incomplete_coverage_visible():
    c = controller()
    c.close_for_time()
    assert c.closed
    assert c.states[1].coverage == "missing"
    assert c.states[1].stopped_reason == "time_incomplete"


@pytest.mark.asyncio
async def test_semantic_repeat_gets_one_rewrite_before_delivery(monkeypatch):
    import app.api.text_chat as api
    import app.pipeline.conversation_policy as policy
    c = controller()
    c.apply_observation(observation(c))
    move, decision = probe_move(c)
    prefix = '[[TURN ' + json.dumps({"action": "probe", "question": 2, "intent": decision.intent}) + ']] '
    mock = AsyncMock(side_effect=[
        {"content": prefix + "What were you using AI for?", "prompt_tokens": 1, "completion_tokens": 1},
        {"content": prefix + "What made you choose AI for that?", "prompt_tokens": 1, "completion_tokens": 1},
    ])
    monkeypatch.setattr(api, "_call_llm", mock)
    monkeypatch.setattr(api, "_get_key", AsyncMock(return_value="test"))
    monkeypatch.setattr(api, "_openai_api_base", AsyncMock(return_value=None))
    check = AsyncMock(side_effect=[
        (QuestionCheck(acceptable=False, issue="already_answered", correction="The learning purpose is known; ask about choosing AI."), {"prompt_tokens": 1, "completion_tokens": 1}),
        (QuestionCheck(acceptable=True, issue="none", correction=""), {"prompt_tokens": 1, "completion_tokens": 1}),
    ])
    monkeypatch.setattr(policy, "check_question", check)
    result, review = await api._contextual_agent_turn(c, [{"role": "user", "content": c.history[-1]["answer"]}],
                                                   model="test", study_id=None, participant_text=c.history[-1]["answer"])
    assert review.text == "What made you choose AI for that?"
    assert c.follow_ups == 1
    assert mock.await_count == 2
    assert len([e for e in c.events if e["action"] == "question_rejected"]) == 1
    assert result["prompt_tokens"] == 4


@pytest.mark.asyncio
async def test_repair_is_bounded_when_renderer_keeps_proposing_invalid_moves(monkeypatch):
    import app.api.text_chat as api
    c = controller()
    mock = AsyncMock(return_value={"content": "Bad untagged output", "prompt_tokens": 1, "completion_tokens": 1})
    monkeypatch.setattr(api, "_call_llm", mock)
    _, review = await api._contextual_agent_turn(c, [], model="test", study_id=None, participant_text="History")
    assert mock.await_count == 2
    assert review.decision.intent == "vision:main"
    assert c.states[1].coverage == "missing"


def test_schema_requires_exploratory_policy_and_unique_stable_ids():
    from app.schemas.agent import InterviewGuide
    with pytest.raises(ValueError, match="exploratory"):
        InterviewGuide.model_validate({"conversation_policy": "contextual_v1", "questions": [{"text": "Hello?"}]})


@pytest.mark.asyncio
@pytest.mark.parametrize('issue', ['already_answered', 'participant_limit'])
async def test_failed_repair_cannot_deliver_an_anchor_rejected_as_out_of_bounds(monkeypatch, issue):
    import app.api.text_chat as api
    import app.pipeline.conversation_policy as policy
    c = controller(topic='vision', answer='I would rather not talk about my hopes further.')
    bad = '[[TURN {"action":"main","question":4,"intent":"vision_progress:main"}]] Have you seen progress toward that hope?'
    monkeypatch.setattr(api, '_call_llm', AsyncMock(return_value={'content': bad, 'prompt_tokens': 1, 'completion_tokens': 1}))
    monkeypatch.setattr(api, '_get_key', AsyncMock(return_value='test'))
    monkeypatch.setattr(api, '_openai_api_base', AsyncMock(return_value=None))
    monkeypatch.setattr(policy, 'check_question', AsyncMock(return_value=(
        QuestionCheck(acceptable=False, issue=issue, correction='This anchor is out of bounds.'),
        {'prompt_tokens': 1, 'completion_tokens': 1})))
    _, review = await api._contextual_agent_turn(c, [], model='test', study_id=None, participant_text=c.history[-1]['answer'])
    assert review.decision.intent == 'values_conflict:main'
    assert c.states[3].coverage == 'missing'
    assert c.states[3].stopped_reason == 'question_check:' + issue


def test_two_interrogatives_are_rejected_even_with_one_question_mark():
    c = controller()
    c.apply_observation(observation(c))
    _, decision = probe_move(c)
    with pytest.raises(ConversationRepair, match='one clean'):
        c.validate_turn('What prompted you to choose AI, and how did it go?', decision, c.history[-1]['answer'])


def test_fallback_skips_covered_topics_after_a_partial_anchor():
    c = controller()
    c.states[2].coverage = 'partial'
    c.states[3].coverage = 'adequate'
    review = c.fallback_turn('unavailable')
    assert review.decision.intent == 'values_conflict:main'
    assert c.states[2].coverage == 'partial'


def test_subject_refusal_defers_sensitive_module_without_claiming_coverage():
    guide = json.loads((Path(__file__).parents[1] / "studies/anthropic_voices_v01/health_nonusers.json").read_text())["agent"]["interview_guide"]
    c = ContextualInterviewController(guide)
    index = c.ids["health_information"]
    c.current_question_index = c._last_question_index = index
    c.states[index].asked = True
    c._last_text = c.main_wording(index)
    c.note_answer("I prefer not to discuss personal circumstances.", 20)
    obs = observation(c, coverage="declined", opportunities=False)
    obs.boundaries = [ModuleBoundary(module="Group focus", quote=c.history[-1]["answer"])]
    c.apply_observation(obs)
    assert c.eligible_moves()[0]["intent"] == "close:main"
    assert c.states[c.ids["health_role"]].coverage == "missing"
    assert c.states[c.ids["health_role"]].stopped_reason == "participant_boundary"


def test_fallback_does_not_deliver_unchecked_sensitive_anchor():
    c = controller(topic="company_trust")
    c.questions[c.ids["close"]]["sensitivity"] = "elevated"
    review = c.fallback_turn("checker unavailable")
    assert review.closed
    assert c.states[c.ids["close"]].stopped_reason == "generation_failure_incomplete"


def test_recent_main_keeps_approved_single_episode_wording():
    c = controller(topic="everyday_use")
    move = next(m for m in c.eligible_moves() if m["action"] == "main")
    c.validate_turn(move["wording"], TurnDecision("main", move["question"], move["intent"]), c.history[-1]["answer"])


@pytest.mark.parametrize("text", ["I don't have anything more to add about that.", "I don’t know; I haven’t described a specific experience like that."])
def test_submitted_text_is_not_treated_as_a_cut_off_audio_fragment(text):
    from app.pipeline.interview_guide import classify_participant_turn, participant_interruption_response
    assert classify_participant_turn(text, infer_incomplete=False) == "answer"
    assert participant_interruption_response(text, infer_incomplete=False) is None
    assert classify_participant_turn("I'm not done.", infer_incomplete=False) == "continuation"


@pytest.mark.asyncio
async def test_approved_anchor_does_not_have_to_collect_whole_topic_in_one_question(monkeypatch):
    import app.api.text_chat as api
    import app.pipeline.conversation_policy as policy
    c = controller(topic='vision_progress', answer="I don't know about progress.")
    move = next(m for m in c.eligible_moves() if m['action'] == 'main')
    raw = '[[TURN ' + json.dumps({'action': 'main', 'question': move['question'], 'intent': move['intent']}) + ']] ' + move['wording']
    renderer = AsyncMock(return_value={'content': raw, 'prompt_tokens': 1, 'completion_tokens': 1})
    monkeypatch.setattr(api, '_call_llm', renderer)
    monkeypatch.setattr(api, '_get_key', AsyncMock(return_value='test'))
    monkeypatch.setattr(api, '_openai_api_base', AsyncMock(return_value=None))
    monkeypatch.setattr(policy, 'check_question', AsyncMock(return_value=(
        QuestionCheck(acceptable=False, issue='wrong_purpose', correction='Ask for the concern and its source together.'),
        {'prompt_tokens': 1, 'completion_tokens': 1})))
    _, review = await api._contextual_agent_turn(c, [], model='test', study_id=None, participant_text=c.history[-1]['answer'])
    assert renderer.await_count == 1
    assert review.text == move['wording']
    assert any(e['action'] == 'critic_scope_override' for e in c.events)


def test_repetition_rejection_allows_anchor_rewording_before_final_fallback():
    c = controller(topic='everyday_use')
    move = next(m for m in c.eligible_moves() if m['action'] == 'main')
    c.defer_rejected_anchor(move, 'already_answered')
    assert move in c.eligible_moves()
    c.defer_rejected_anchor(move, 'already_answered', final=True)
    assert move not in c.eligible_moves()


def nonuse_controller():
    guide = json.loads((Path(__file__).parents[1] / "studies/anthropic_voices_v01/ai_nonusers.json").read_text())["agent"]["interview_guide"]
    return ContextualInterviewController(guide)


def test_other_ai_use_cannot_create_chatbot_nonuse_premise():
    c = nonuse_controller()
    c.ai_use = "uses_ai"
    wording = c.main_wording(c.ids["nonuse"])
    assert wording == "Which AI tool have you been using?"
    assert "not" not in wording


def test_confirmed_chatbot_use_removes_future_nonuse_module_with_evidence():
    c = nonuse_controller()
    c._last_text = "Have you used conversational AI?"
    c.note_answer("Yes, I use a chatbot.", 4)
    obs = observation(c, opportunities=False)
    obs.chatbot_use = "uses_chatbots"
    obs.chatbot_use_quote = "I use a chatbot"
    obs.chatbot_use_sequence = 4
    c.apply_observation(obs)
    state = c.states[c.ids['nonuse']]
    assert state.coverage == 'not_applicable'
    assert state.sources[0]['sequence'] == 4
    c.current_question_index = c.ids['nonuse'] - 1
    c.states[c.current_question_index].asked = True
    assert c.eligible_moves()[-1]['intent'] == 'close:main'


def test_fabricated_chatbot_use_evidence_cannot_skip_module():
    c = nonuse_controller()
    c.note_answer("I make pictures with AI.", 4)
    obs = observation(c, opportunities=False)
    obs.chatbot_use = 'uses_chatbots'
    obs.chatbot_use_quote = 'I use chatbots'
    obs.chatbot_use_sequence = 4
    c.apply_observation(obs)
    assert c.chatbot_use == 'unclear'
    assert c.states[c.ids['nonuse']].coverage == 'missing'


def test_partial_thought_survives_renderer_failure_once():
    c = controller(topic='vision', answer='I wish AI would tell us when it is')
    obs = observation(c, coverage='partial', purpose='Clarify the unfinished thought')
    obs.opportunities[0].move = 'clarify_meaning'
    c.apply_observation(obs)
    review = c.fallback_turn('invalid output twice')
    assert review.decision.action == 'probe'
    assert review.question_index == c.ids['vision']
    assert c.states[c.ids['vision']].coverage == 'partial'
    assert c.follow_ups == 1
    assert c.fallback_turn('failed again').decision.intent == 'vision_progress:main'


def test_exhausted_opportunity_does_not_reappear_in_fallback():
    c = controller(topic='vision', answer='I already explained what I meant.')
    obs = observation(c, coverage='partial')
    obs.opportunities[0].move = 'clarify_meaning'
    c.apply_observation(obs)
    move, _ = probe_move(c)
    c.retire_rejected_opportunity(move, 'already_answered')
    assert c.fallback_turn('checker rejected purpose').decision.intent == 'vision_progress:main'
    assert c.states[c.ids['vision']].coverage == 'partial'


def test_finished_session_does_not_claim_complete_coverage():
    c = controller()
    c.close_for_time()
    stats = c.stats()
    assert stats['session_finished']
    assert not stats['coverage_complete']
    assert stats['coverage_by_topic']['recent_encounter'] == 'missing'


def test_slow_answers_reserve_time_for_remaining_anchors():
    c = controller()
    c.remaining_seconds = 800
    c.note_answer('First detail', 8)
    c.remaining_seconds = 720
    c.note_answer('Second detail', 10)
    c.remaining_seconds = 640
    c.note_answer('Third detail', 12)
    assert c.seconds_per_exchange == 80
    c.remaining_seconds = 350
    assert not c._probe_allowed(c.ids['recent_encounter'])


@pytest.mark.asyncio
async def test_missing_marker_repair_keeps_selected_clarification(monkeypatch):
    import app.api.text_chat as api
    import app.pipeline.conversation_policy as policy
    c = controller(topic='vision', answer='I wish AI would tell us when it is')
    obs = observation(c, coverage='partial', purpose='Clarify what they want disclosed')
    obs.opportunities[0].move = 'clarify_meaning'
    c.apply_observation(obs)
    move, decision = probe_move(c)
    raw = '[[TURN ' + json.dumps({'action': 'probe', 'question': decision.question, 'intent': decision.intent}) + ']] What should it disclose, and what would that change?'
    monkeypatch.setattr(api, '_call_llm', AsyncMock(side_effect=[
        {'content': raw, 'prompt_tokens': 1, 'completion_tokens': 1},
        {'content': 'What would you want it to disclose?', 'prompt_tokens': 1, 'completion_tokens': 1},
    ]))
    monkeypatch.setattr(api, '_get_key', AsyncMock(return_value='test'))
    monkeypatch.setattr(api, '_openai_api_base', AsyncMock(return_value=None))
    checker = AsyncMock(return_value=(QuestionCheck(acceptable=True, issue='none', correction=''), {'prompt_tokens': 1, 'completion_tokens': 1}))
    monkeypatch.setattr(policy, 'check_question', checker)
    _, review = await api._contextual_agent_turn(c, [], model='test', study_id=None, participant_text=c.history[-1]['answer'])
    assert review.text == 'What would you want it to disclose?'
    assert review.question_index == c.ids['vision']
    assert checker.await_count == 1
    assert any(e['action'] == 'repair_marker_recovered' for e in c.events)


@pytest.mark.asyncio
async def test_retired_move_cannot_be_resurrected_by_marker_recovery(monkeypatch):
    import app.api.text_chat as api
    import app.pipeline.conversation_policy as policy
    c = controller()
    c.apply_observation(observation(c))
    move, decision = probe_move(c)
    raw = '[[TURN ' + json.dumps({'action': 'probe', 'question': decision.question, 'intent': decision.intent}) + ']] Why did you choose AI?'
    monkeypatch.setattr(api, '_call_llm', AsyncMock(side_effect=[
        {'content': raw, 'prompt_tokens': 1, 'completion_tokens': 1},
        {'content': 'What made you choose it?', 'prompt_tokens': 1, 'completion_tokens': 1},
    ]))
    monkeypatch.setattr(api, '_get_key', AsyncMock(return_value='test'))
    monkeypatch.setattr(api, '_openai_api_base', AsyncMock(return_value=None))
    monkeypatch.setattr(policy, 'check_question', AsyncMock(return_value=(
        QuestionCheck(acceptable=False, issue='already_answered', correction='Motivation was already explained.', purpose_exhausted=True),
        {'prompt_tokens': 1, 'completion_tokens': 1})))
    _, review = await api._contextual_agent_turn(c, [], model='test', study_id=None, participant_text=c.history[-1]['answer'])
    assert review.decision.intent == 'vision:main'
    assert c.follow_ups == 0


def test_opening_context_is_preserved_without_pretending_anchor_was_answered():
    c = nonuse_controller()
    c.note_opening_answer('I have only tried AI in research studies.', 2, 'Ready to start?')
    assert not c.needs_evidence_judge
    assert not any(s.asked for s in c.states)
    c._last_text = c.main_wording(0)
    c.note_answer('Not in everyday life.', 4)
    assert c.observer_context()['conversation'][0]['answer'] == 'I have only tried AI in research studies.'
    assert c.question_check_context('Do you use AI?', {})['conversation'][0]['sequence'] == 2


def test_uncertain_vision_has_progress_wording_without_a_presumed_goal():
    c = controller(topic='vision', answer='I am unsure.')
    c.apply_observation(observation(c, coverage='unknown', opportunities=False))
    move = c.eligible_moves()[-1]
    assert move['intent'] == 'vision_progress:main'
    assert 'hope' not in move['wording']
    assert 'without assuming' in move['purpose']


def test_later_chatbot_correction_can_reopen_nonuse_without_stale_overwrite():
    c = nonuse_controller()
    c.note_answer('I use a chatbot.', 4)
    obs = observation(c, opportunities=False)
    obs.chatbot_use = 'uses_chatbots'; obs.chatbot_use_quote = 'I use a chatbot'; obs.chatbot_use_sequence = 4
    c.apply_observation(obs)
    c.note_answer('Actually I meant ordinary search, not a chatbot.', 6)
    correction = observation(c, opportunities=False)
    correction.chatbot_use = 'no_chatbot_use'; correction.chatbot_use_quote = c.history[-1]['answer']; correction.chatbot_use_sequence = 6
    c.apply_observation(correction)
    assert c.states[c.ids['nonuse']].coverage == 'missing'
    c.apply_observation(obs)
    assert c.chatbot_use == 'no_chatbot_use'
    assert c.states[c.ids['nonuse']].coverage == 'missing'


def test_episode_gap_survives_adequate_coverage_and_blocks_topic_jump():
    c = controller(answer='I asked AI to help with fractions.')
    obs = observation(c, opportunities=False)
    obs.experience_gap = Opportunity(topic_id='recent_encounter', move='evaluation',
        purpose='Understand how that episode went', anchor_sequence=6,
        anchor_quote=c.history[-1]['answer'], priority='medium')
    c.apply_observation(obs)
    move, decision = probe_move(c)
    assert move['priority'] == 'high'
    assert c.states[c.ids['recent_encounter']].coverage == 'adequate'
    assert c.validate_turn('How did that go?', decision, c.history[-1]['answer']) == move
    main = next(m for m in c.eligible_moves() if m['action'] == 'main')
    with pytest.raises(ConversationRepair, match='valuable'):
        c.validate_turn(main['wording'], TurnDecision('main', main['question'], main['intent']), '')


@pytest.mark.parametrize('boundary', ['time', 'declined', 'fabricated', 'already_moved_on'])
def test_episode_gap_cannot_override_limits_or_grounding(boundary):
    c = controller(answer='I tried it for a recipe.')
    obs = observation(c, coverage='declined' if boundary == 'declined' else 'adequate', opportunities=False)
    obs.experience_gap = Opportunity(topic_id='recent_encounter', move='evaluation',
        purpose='Understand the outcome', anchor_sequence=6,
        anchor_quote='invented account' if boundary == 'fabricated' else c.history[-1]['answer'], priority='high')
    if boundary == 'time': c.remaining_seconds = 30
    if boundary == 'already_moved_on': c.current_question_index = c.ids['vision']
    c.apply_observation(obs)
    assert all(m['action'] not in {'probe', 'thread'} for m in c.eligible_moves())


def test_research_only_use_preserves_everyday_nonuse_and_its_reason():
    c = nonuse_controller()
    c.current_question_index = c._last_question_index = c.ids['nonuse']
    c.states[c.ids['nonuse']].asked = True
    c._last_text = 'What has shaped whether you use AI chatbots?'
    c.note_answer('I use chatbots only in research studies. I prefer thinking things through myself.', 16)
    obs = observation(c, opportunities=False)
    obs.chatbot_use = 'research_only'; obs.chatbot_use_quote = c.history[-1]['answer']; obs.chatbot_use_sequence = 16
    c.apply_observation(obs)
    assert c.chatbot_use == 'research_only'
    assert c.states[c.ids['nonuse']].coverage == 'adequate'
    assert c.states[c.ids['nonuse']].stopped_reason != 'confirmed_chatbot_use'
    assert c.eligible_moves()[-1]['intent'] == 'close:main'


def test_research_only_correction_reopens_incorrect_nonuse_skip():
    c = nonuse_controller()
    c.note_answer('I use a chatbot.', 4)
    obs = observation(c, opportunities=False)
    obs.chatbot_use = 'uses_chatbots'; obs.chatbot_use_quote = c.history[-1]['answer']; obs.chatbot_use_sequence = 4
    c.apply_observation(obs)
    c.note_answer('Only in research studies, never outside them.', 6)
    obs = observation(c, opportunities=False)
    obs.chatbot_use = 'research_only'; obs.chatbot_use_quote = c.history[-1]['answer']; obs.chatbot_use_sequence = 6
    c.apply_observation(obs)
    assert c.states[c.ids['nonuse']].coverage == 'missing'
    assert c.main_wording(c.ids['nonuse']).startswith('Outside research studies')


def test_main_tag_shorthand_cannot_authorize_an_ineligible_move():
    c = controller()
    c.apply_observation(observation(c))
    main = next(m for m in c.eligible_moves() if m['action'] == 'main')
    shorthand = TurnDecision('main', main['question'], 'vision')
    decision = c.canonical_decision(shorthand)
    assert decision.intent == 'vision:main'
    with pytest.raises(ConversationRepair, match='valuable'):
        c.validate_turn(main['wording'], decision, '')
    wrong = TurnDecision('main', 1, 'everyday_use')
    assert c.canonical_decision(wrong) == wrong
    with pytest.raises(ConversationRepair, match='not eligible'):
        c.validate_turn('Do you use AI?', wrong, '')


def test_unclear_chatbot_use_cannot_retire_nonuse_from_generic_ai_evidence():
    c = nonuse_controller()
    c.current_question_index = c._last_question_index = c.ids['nonuse']
    c.note_answer('I use AI for learning.', 8)
    obs = observation(c, coverage='not_applicable', opportunities=False)
    c.apply_observation(obs)
    assert c.chatbot_use == 'unclear'
    assert c.states[c.ids['nonuse']].coverage == 'missing'
    assert c.main_wording(c.ids['nonuse']) == 'Have you used AI chatbots that you can have a conversation with?'


def test_recent_experience_thread_can_continue_without_nested_topic_jump():
    c = controller(topic='everyday_use', answer='Sometimes friends show me AI.')
    c.states[c.ids['recent_encounter']].coverage = 'partial'
    op = Opportunity(topic_id='recent_encounter', move='develop_experience', purpose='Identify an episode',
        anchor_sequence=6, anchor_quote=c.history[-1]['answer'], priority='high')
    obs = observation(c, opportunities=False); obs.opportunities = [op]
    c.apply_observation(obs)
    move = next(m for m in c.eligible_moves() if m['action'] == 'thread')
    c.accept_turn('What were your friends doing that time?', TurnDecision('thread', 2, move['intent']), move)
    c.note_answer('They were checking a claim about a film.', 8)
    obs = observation(c, opportunities=False)
    obs.experience_gap = Opportunity(topic_id='recent_encounter', move='evaluation', purpose='Understand the experience',
        anchor_sequence=8, anchor_quote=c.history[-1]['answer'], priority='high')
    c.apply_observation(obs)
    move = next(m for m in c.eligible_moves() if m['action'] == 'thread')
    assert move['topic_id'] == 'recent_encounter'
    assert c.validate_turn('How did that go?', TurnDecision('thread', 2, move['intent']), '') == move
    c.states[c.ids['recent_encounter']].follow_ups = 3
    assert not any(m['action'] == 'thread' for m in c.eligible_moves())


@pytest.mark.asyncio
@pytest.mark.parametrize('quote,accepted', [('learn how to repair bicycles', False), ('invented participant answer', True)])
async def test_checker_reuse_evidence_is_grounded_and_overrides_inconsistent_verdict(monkeypatch, quote, accepted):
    import app.pipeline.conversation_policy as policy
    result = QuestionCheck(acceptable=True, issue='none', correction='', answer_reuse_quote=quote)
    monkeypatch.setattr(policy, '_structured_call', AsyncMock(return_value=(result, {})))
    check, _ = await policy.check_question(payload={'conversation': [{'answer': 'It should help people learn how to repair bicycles.'}]})
    assert check.acceptable is accepted
    if not accepted: assert check.issue == 'already_answered'
