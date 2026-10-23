import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from app.pipeline.interview_guide import classify_participant_turn, TurnDecision
from app.pipeline.conversation_policy import ContextualInterviewController, ConversationRepair, QuestionCheck
from app.api import text_chat as api


@pytest.mark.parametrize('text,kind', [
    ('I don’t understand your question closer to what ?', 'clarification'),
    ("I don't understand your question closer to what ?", 'clarification'),
    ('I forgot what the question was', 'clarification'),
    ('I don’t know what idea that was personal, but I forgot what we are the questions', 'clarification'),
    ('let me think', 'continuation'), ('give me a minute', 'continuation'),
    ('I do not know', 'answer'), ('I think it could help', 'answer'),
    ('Could AI do the dishes?', 'answer'),
])
def test_real_interruptions_and_substantive_answers_are_distinct(text,kind):
    assert classify_participant_turn(text, 'en', infer_incomplete=False) == kind


def controller():
    d=json.loads((Path(__file__).parents[1]/'studies/anthropic_voices_v01/claude_control.json').read_text())
    c=ContextualInterviewController(d['agent']['interview_guide'])
    c.current_question_index=c._last_question_index=3
    c._last_text='Have you seen anything that makes that hope feel closer?'
    for s in c.states[:4]:s.asked=True
    return c


def test_clarification_cannot_be_overridden_by_topic_change():
    c=controller();c.turn_kind='clarification';move=next(m for m in c.eligible_moves() if m['action']=='main')
    with pytest.raises(ConversationRepair,match='preserve'):
        c.validate_turn(move['wording'],TurnDecision('main',move['question'],move['intent']),'I don’t understand your question closer to what ?')
    review=c.accept_turn('Have you heard of any progress toward the health outcome you mentioned?',TurnDecision('clarify'),{})
    assert c.current_question_index==3 and c.follow_ups==0 and not review.closed


async def test_clarification_repair_survives_two_bad_generations(monkeypatch):
    c=controller();c.turn_kind='clarification'
    monkeypatch.setattr(api,'_call_llm',AsyncMock(return_value={'content':'[[TURN {"action":"close","question":null,"intent":null}]] Goodbye.'}))
    _,review=await api._contextual_agent_turn(c,[],model='test',study_id=None,participant_text='I don’t understand your question closer to what ?')
    assert review.decision.action=='clarify' and not c.closed
    assert c.current_prompt_text in review.text
    assert c.follow_ups==0


def test_bad_wording_cannot_delete_unanswered_anchor():
    c=controller();c.current_question_index=0;c.states[0].coverage='adequate'
    for state in c.states[1:]:state.asked=False
    move=next(m for m in c.eligible_moves() if m['action']=='main')
    check=QuestionCheck(acceptable=False,issue='unsupported_premise',correction='Ask about AI generally',purpose_exhausted=True)
    c.defer_rejected_anchor(move,check.issue,final=True,check=check)
    review=c.fallback_turn('wording failed')
    assert review.decision.question==2
    assert c.states[1].coverage=='missing'
    assert 'last' in review.text.lower() or 'recent' in review.text.lower()


async def test_pause_ignores_keepalives_and_does_not_consume_answers():
    ws=AsyncMock()
    ws.receive_text=AsyncMock(side_effect=[json.dumps({'type':'keepalive'}),json.dumps({'type':'message','text':'not a resume'}),json.dumps({'type':'resume'})])
    result=await api._wait_for_participant_resume(ws,1)
    assert result=={'type':'resume'}
    ws.send_json.assert_awaited_once_with({'type':'paused','seconds_remaining':1})


async def test_pause_has_a_real_deadline():
    ws=AsyncMock()
    async def no_response():await asyncio.sleep(1)
    ws.receive_text.side_effect=no_response
    with pytest.raises(asyncio.TimeoutError):await api._wait_for_participant_resume(ws,.01)


async def test_participant_can_finish_during_break():
    ws=AsyncMock();ws.receive_text.return_value=json.dumps({'type':'finish'})
    assert (await api._wait_for_participant_resume(ws,1))['type']=='finish'


def test_no_hoped_for_outcome_skips_progress_without_inventing_a_goal():
    c=controller();c.current_question_index=2
    c.states[c.ids['vision']].coverage='not_applicable'
    c.states[c.ids['vision']].sources=[{'sequence':10,'quote':'I do not wish for anything','meaning':'No desired outcome'}]
    c.states[c.ids['vision_progress']].asked=False
    assert c._next_index()!=c.ids['vision_progress']
    assert c.states[c.ids['vision_progress']].coverage=='not_applicable'


async def test_full_chat_break_clarification_and_addendum_preserve_question(
    monkeypatch, db_session_factory, make_study, make_agent
):
    from tests.test_text_chat import _SyntheticParticipantSocket
    import app.pipeline.conversation_policy as policy
    from app.models.agent import InterviewMode
    from app.models.session import Session
    from sqlalchemy import select
    study=make_study(title='Synthetic recovery integration')
    agent=make_agent(study_id=study.id,welcome_message='Ready to begin?')
    agent.interview_mode=InterviewMode.STRUCTURED
    agent.interview_guide={'conversation_policy':'contextual_v1','orchestration_profile':'exploratory',
        'questions':[{'id':'everyday_use','text':'What do you use AI for?','phase':'warm_up','always_ask':True,'max_follow_ups':0},
                     {'id':'vision','text':'What would you hope AI could do?','phase':'core','always_ask':True,'max_follow_ups':0}],
        'closing_message':'Thank you. This completes the interview.'}
    async with db_session_factory() as db:
        db.add_all([study,agent]);await db.commit()
    socket=_SyntheticParticipantSocket([
        {'type':'message','text':'Yes'},
        {'type':'pause'},{'type':'keepalive'},{'type':'resume'},
        {'type':'message','text':'let me think'},
        {'type':'message','text':'I use it to write emails.'},
        {'type':'addendum','text':'Also to plan meals.'},
        {'type':'message','text':'I don’t understand your question'},
        {'type':'message','text':'I hope it can translate conversations.'},
    ])
    llm=AsyncMock(side_effect=[
        {'content':'[[TURN {"action":"main","question":1,"intent":"everyday_use:main"}]] What do you use AI for?'},
        {'content':'[[TURN {"action":"main","question":2,"intent":"vision:main"}]] What would you hope AI could do?'},
        {'content':'[[TURN {"action":"close","question":null,"intent":null}]] Thank you.'},
    ])
    observer=AsyncMock(return_value=(policy.Observation(updates=[],opportunities=[],ai_use='unknown',ai_use_quote='',
        chatbot_use='unclear',chatbot_use_quote='',chatbot_use_sequence=None,boundaries=[]),{'prompt_tokens':0,'completion_tokens':0}))
    monkeypatch.setattr(api,'async_session_factory',db_session_factory)
    for name in ('register_session','unregister_session','publish_transcript_event'):
        monkeypatch.setattr(api,name,AsyncMock())
    monkeypatch.setattr(api,'validate_agent_pipeline_config',AsyncMock(return_value=[]))
    monkeypatch.setattr(api,'_call_llm',llm)
    monkeypatch.setattr(api,'_get_key',AsyncMock(return_value='test'))
    monkeypatch.setattr(api,'_openai_api_base',AsyncMock(return_value=None))
    monkeypatch.setattr(api,'_redact_participant_text',AsyncMock(side_effect=api.redact_text))
    from app.pipeline import turn_router
    async def route(**kwargs):
        text=kwargs['payload']['latest_turn']
        repairs={'let me think':('continuation',"Take your time."),
                 'I don’t understand your question':('clarification','What would you like AI to make possible for you?')}
        kind,reply=repairs.get(text,('answer',''))
        return turn_router.TurnRoute(kind=kind,answer_quote=text if kind=='answer' else '',reply=reply,information_fields=[]), {'prompt_tokens':0,'completion_tokens':0}
    monkeypatch.setattr(turn_router,'route_participant_turn',route)
    monkeypatch.setattr(policy,'observe_conversation',observer)
    monkeypatch.setattr(policy,'check_question',AsyncMock(return_value=(policy.QuestionCheck(acceptable=True,issue='none',correction=''),{'prompt_tokens':0,'completion_tokens':0})))
    await api.text_chat_ws(socket,agent.widget_key,pid=None,external_study_id=None,external_session_id=None,
        consent=True,response_mode='text',mic_check_status=None)
    assert not socket.incoming
    assert [m['type'] for m in socket.sent if m['type'] in {'paused','resumed'}]==['paused','resumed']
    assert observer.await_count==3
    payloads=[call.kwargs['payload'] for call in observer.call_args_list]
    assert [p['active_topic'] for p in payloads]==['everyday_use','everyday_use','vision']
    assert payloads[1]['latest_answer']=='Also to plan meals.'
    assert payloads[1]['resume_topic']=='vision'
    assert payloads[2]['actual_question']=='What would you like AI to make possible for you?'
    assert any('Thanks for adding that. What would you hope AI could do?'==m.get('text') for m in socket.sent)
    async with db_session_factory() as db:
        saved=await db.scalar(select(Session).where(Session.agent_id==agent.id))
        assert saved.status.value=='completed'
        assert saved.completion_metadata['timing']['participant_pause_seconds']>=0
        assert saved.completion_metadata['protocol']['follow_ups']==0


async def test_grounded_cutoff_invites_completion_without_a_model_probe(monkeypatch):
    from app.pipeline.conversation_policy import Observation
    c=controller();c.note_answer('It gave me a list of things to wa',10)
    c.apply_observation(Observation(updates=[],opportunities=[],unfinished_answer_quote='things to wa',
        ai_use='unknown',ai_use_quote='',chatbot_use='unclear',chatbot_use_quote='',chatbot_use_sequence=None,boundaries=[]))
    model=AsyncMock();monkeypatch.setattr(api,'_call_llm',model)
    _,review=await api._contextual_agent_turn(c,[],model='test',study_id=None,participant_text=c.history[-1]['answer'])
    assert review.decision.action=='clarify' and 'finish your thought' in review.text
    assert c.follow_ups==0 and c.current_question_index==3
    model.assert_not_called()


def test_explicit_no_outcome_can_mark_dependent_progress_inapplicable():
    from app.pipeline.conversation_policy import Observation, TopicUpdate
    c=controller();c.current_question_index=c._last_question_index=c.ids['vision']
    answer='I do not have a hoped-for outcome for AI. Nothing comes to mind.'
    c.note_answer(answer,8)
    c.apply_observation(Observation(updates=[
        TopicUpdate(topic_id='vision',coverage='adequate',quote=answer,meaning='No desired outcome',exploration='settled',correction=False),
        TopicUpdate(topic_id='vision_progress',coverage='not_applicable',quote=answer,meaning='No outcome to assess progress toward',exploration='settled',correction=False),
        TopicUpdate(topic_id='company_trust',coverage='unknown',quote=answer,meaning='Unsupported unrelated absence',exploration='settled',correction=False)],
        opportunities=[],ai_use='unknown',ai_use_quote='',chatbot_use='unclear',chatbot_use_quote='',chatbot_use_sequence=None,boundaries=[]))
    assert c.states[c.ids['vision_progress']].coverage=='not_applicable'
    assert c.states[c.ids['company_trust']].coverage=='missing'
