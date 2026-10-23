"""Semantic routing boundaries; model behavior is evaluated in the live eval."""
from unittest.mock import AsyncMock
import asyncio
import pytest
from app.pipeline import turn_router as router


def route(kind='answer', quote='A research answer', reply='', fields=None):
    return router.TurnRoute(kind=kind, answer_quote=quote, reply=reply, information_fields=fields or [])


async def test_stop_requires_second_semantic_decision(monkeypatch):
    call=AsyncMock(side_effect=[(route('stop',''),{'prompt_tokens':1,'completion_tokens':1}),
                               (route(),{'prompt_tokens':1,'completion_tokens':1})])
    monkeypatch.setattr(router,'_structured_call',call)
    result,usage=await router.route_participant_turn(api_key='test',api_base=None,model='small',fallback_model='large',
        payload={'latest_turn':'A research answer','participant_information':{}})
    assert result.kind=='answer' and call.await_count==2
    assert usage['prompt_tokens']==2


async def test_router_outage_holds_question_without_keyword_fallback(monkeypatch):
    monkeypatch.setattr(router,'_structured_call',AsyncMock(side_effect=RuntimeError('unavailable')))
    result,usage=await router.route_participant_turn(api_key='test',api_base=None,model='small',fallback_model='large',
        payload={'latest_turn':'stop','participant_information':{}})
    assert result.kind=='uncertain' and not result.answer_quote
    assert usage['failed'] and 'Finish interview' in result.reply


async def test_router_timeout_is_bounded(monkeypatch):
    async def stuck(**kwargs):await asyncio.sleep(2)
    monkeypatch.setattr(router,'_structured_call',stuck)
    result,usage=await router.route_participant_turn(api_key='test',api_base=None,model='small',fallback_model='large',
        timeout_seconds=.001,payload={'latest_turn':'anything','participant_information':{}})
    assert result.kind=='uncertain' and usage['elapsed_seconds']<1


async def test_ungrounded_extraction_escalates(monkeypatch):
    monkeypatch.setattr(router,'_structured_call',AsyncMock(side_effect=[
        (route(quote='invented'),{'prompt_tokens':1,'completion_tokens':1}),
        (route(),{'prompt_tokens':1,'completion_tokens':1})]))
    result,usage=await router.route_participant_turn(api_key='test',api_base=None,model='small',fallback_model='large',
        payload={'latest_turn':'A research answer','participant_information':{}})
    assert result.answer_quote=='A research answer'
    assert usage['attempts'][0]['error']=='ValueError'


def test_privacy_renderer_cannot_deliver_generated_facts():
    info={'data_retention_period':'Data is retained for one year.','estimated_minutes':15}
    r=route('procedural','',reply='Your employer will see everything.',fields=['data_retention_period','estimated_minutes','unavailable'])
    text=router.repair_text(r,info)
    assert 'Data is retained for one year.' in text and 'about 15 minutes' in text
    assert 'employer' not in text and "don't have that detail" in text


def test_unknown_information_fields_are_rejected():
    with pytest.raises(ValueError,match='Unknown'):
        router.validate_route(route('procedural','',fields=['made_up']), 'hello', {})


def test_mixed_answer_requires_exact_quote():
    r=route('procedural','AI might replace my job.',fields=['unavailable'])
    router.validate_route(r,'AI might replace my job. Who sees this?',{})
    with pytest.raises(ValueError,match='Ungrounded'):
        router.validate_route(r,'My work is fine. Who sees this?',{})


async def test_mixed_answer_and_privacy_resume_preserve_evidence(
    monkeypatch, db_session_factory, make_study, make_agent
):
    from app.api import text_chat as api
    from app.pipeline import conversation_policy as policy, interview_guide
    from app.models.agent import InterviewMode
    from tests.test_text_chat import _SyntheticParticipantSocket
    study=make_study(title='Synthetic semantic routing')
    agent=make_agent(study_id=study.id,welcome_message='Ready to begin?')
    agent.interview_mode=InterviewMode.STRUCTURED
    agent.interview_guide={'conversation_policy':'contextual_v1','orchestration_profile':'exploratory',
        'questions':[{'id':'everyday_use','text':'What do you use AI for?','phase':'warm_up','always_ask':True,'max_follow_ups':0}],
        'closing_message':'Thank you. This completes the interview.'}
    async with db_session_factory() as db:
        db.add_all([study,agent]);await db.commit()
    socket=_SyntheticParticipantSocket([
        {'type':'message','text':'Yes'},
        {'type':'message','text':'I use it for work. Who reads these answers?'},
        {'type':'message','text':'Okay, thanks.'},
        {'type':'message','text':'I use it daily.'},
    ])
    mock_routes=[route(quote='Yes'),
        route('procedural','I use it for work.',fields=['unavailable']),
        route('resume','',reply='What do you use AI for?'),route(quote='I use it daily.')]
    monkeypatch.setattr(router,'route_participant_turn',AsyncMock(side_effect=[(r,{'prompt_tokens':1,'completion_tokens':1}) for r in mock_routes]))
    # Any accidental re-entry into the legacy lexical classifier fails this test.
    monkeypatch.setattr(interview_guide,'classify_participant_turn',lambda *a,**k: (_ for _ in ()).throw(AssertionError('legacy classifier called')))
    llm=AsyncMock(side_effect=[
        {'content':'[[TURN {"action":"main","question":1,"intent":"everyday_use:main"}]] What do you use AI for?'},
        {'content':'[[TURN {"action":"close","question":null,"intent":null}]] Thank you.'}])
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
    monkeypatch.setattr(policy,'observe_conversation',observer)
    monkeypatch.setattr(policy,'check_question',AsyncMock(return_value=(policy.QuestionCheck(acceptable=True,issue='none',correction=''),{'prompt_tokens':0,'completion_tokens':0})))
    await api.text_chat_ws(socket,agent.widget_key,pid=None,external_study_id=None,external_session_id=None,
        consent=True,response_mode='text',mic_check_status=None)
    assert not socket.incoming
    assert [c.kwargs['payload']['latest_answer'] for c in observer.call_args_list]==['I use it for work.','I use it daily.']
    assert all(c.kwargs['payload']['active_topic']=='everyday_use' for c in observer.call_args_list)
    assert any(m.get('type')=='finalized' and m.get('status')=='completed' for m in socket.sent)
