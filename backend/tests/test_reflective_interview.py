"""Production-boundary regressions for the opt-in draft/critique interviewer."""
import json
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select

from app.models.session import Session, SessionStatus, TranscriptEntry
from app.models.agent import InterviewMode
from app.schemas.agent import InterviewGuide


GUIDE = {"conversation_policy": "reflective_v1", "orchestration_profile": "exploratory",
         "conversation_brief": "Understand concrete UI failures, inputs, and desired results.",
         "questions": [{"id": "task", "text": "Describe a UI task."}],
         "closing_message": "Thank you. Return to Prolific."}


def responses(reply, closed=False):
    draft = {"candidates": [{"id": i, "move": "follow", "reply": reply, "closed": closed}
                             for i in "ABC"]}
    selected = {"selected_id": "A", "reply": reply, "closed": closed}
    return [{"content": json.dumps(x), "prompt_tokens": 10, "completion_tokens": 5}
            for x in (draft, selected)]


class Socket:
    def __init__(self, incoming):
        self.incoming = list(incoming)
        self.sent = []
    async def accept(self): pass
    async def close(self, **kwargs): pass
    async def receive_text(self): return json.dumps(self.incoming.pop(0))
    async def send_json(self, data): self.sent.append(data)


@pytest.mark.asyncio
@pytest.mark.parametrize('answer,reply,closed', [
    ("Is it okay to describe design rather than code?", "Yes, UI design counts. What happened?", False),
    ("Why are we finishing early? Isn't this 25 minutes?", "The time is an estimate. What else would you like to add?", False),
    ("Yes, I have another example.", "What was that other task?", False),
    ("No, I have nothing further.", "Thank you. Return to Prolific.", True),
    ("Please stop the interview now.", "Thank you. Your responses are saved.", True),
    ("The menu breaks at 768px; my email is synthetic.person@example.com.", "What happens to the menu at 768px?", False),
])
async def test_reflective_socket_delivers_selected_reply_and_preserves_controls(
    monkeypatch, db_session_factory, make_study, make_agent, answer, reply, closed
):
    import app.api.text_chat as api
    study = make_study(title="Synthetic reflective integration")
    study.research_spec = {"version": "test-reflective", "privacy_profile": "work_high_risk",
                           "participant_information": {"consent_required": True, "estimated_minutes": 25}}
    agent = make_agent(study_id=study.id, welcome_message="What would you like to discuss?")
    agent.interview_mode = InterviewMode.STRUCTURED
    agent.interview_guide = GUIDE
    async with db_session_factory() as db:
        db.add_all([study, agent])
        await db.commit()
    socket = Socket([{"type": "message", "text": answer}, {"type": "finish"}])
    llm = AsyncMock(side_effect=responses(reply, closed))
    old_judge = AsyncMock(side_effect=AssertionError("Legacy judge must not rewrite C"))
    monkeypatch.setattr(api, 'async_session_factory', db_session_factory)
    for name in ('register_session','unregister_session','publish_transcript_event'):
        monkeypatch.setattr(api, name, AsyncMock())
    monkeypatch.setattr(api, 'validate_agent_pipeline_config', AsyncMock(return_value=[]))
    monkeypatch.setattr(api, '_call_llm', llm)
    monkeypatch.setattr(api, '_judge_structured_turn', old_judge)
    await api.text_chat_ws(socket, agent.widget_key, pid=None, external_study_id=None,
                          external_session_id=None, consent=True, response_mode='text', mic_check_status=None)
    assert llm.await_count == 2
    old_judge.assert_not_awaited()
    assert any(m.get('type')=='message' and m.get('text')==reply for m in socket.sent)
    assert any(m.get('type')=='finalized' and m.get('status')=='completed' for m in socket.sent)
    assert any(m.get('type')=='ended' and m.get('reason')==
               ('interview_complete' if closed else 'participant_finished') for m in socket.sent)
    assert not any(m.get('type') in {'progress','privacy_warning'} for m in socket.sent)
    assert 'synthetic.person@example.com' not in str(llm.await_args_list)
    async with db_session_factory() as db:
        session = await db.scalar(select(Session).where(Session.agent_id==agent.id))
        assert session.status == SessionStatus.COMPLETED
        protocol=session.completion_metadata['protocol']
        assert protocol['conversation_policy']=='reflective_v1'
        assert protocol['model_calls']==2
        assert protocol['format_retries']==0
        entries=(await db.scalars(select(TranscriptEntry).where(TranscriptEntry.session_id==session.id))).all()
        assert 'synthetic.person@example.com' not in str([e.content for e in entries])
        assert 'candidates' not in str([e.content for e in entries])


@pytest.mark.asyncio
@pytest.mark.parametrize('remaining,expected', [(45,'one minute'),(-1,'time has elapsed'),(120,'unresolved detail')])
async def test_timing_is_server_context_not_participant_speech(monkeypatch,remaining,expected):
    import app.api.text_chat as api
    llm=AsyncMock(side_effect=responses('What happened to the spacing?'))
    monkeypatch.setattr(api,'_call_llm',llm)
    result=await api._reflective_agent_turn(
        [{'role':'system','content':'legacy scripted agenda'}, {'role':'user','content':'The spacing changed.'}],
        model='test',study_id=None,guide=GUIDE,research_spec={},remaining_seconds=remaining)
    assert result['closed'] is False
    assert expected in llm.await_args_list[0].args[0][0]['content']
    assert llm.await_args_list[0].args[0][-1]=={'role':'user','content':'The spacing changed.'}
    assert 'legacy scripted agenda' not in str(llm.await_args_list)
    assert '"consent_handled_by_interface": true' in llm.await_args_list[0].args[0][0]['content']
    for call in llm.await_args_list:
        assert '"discussion_guide"' in call.args[0][0]['content']
        assert GUIDE['questions'][0]['text'] in call.args[0][0]['content']
    assert result['prompt_tokens']==20 and result['completion_tokens']==10


@pytest.mark.asyncio
async def test_malformed_output_reuses_answer_with_one_bounded_retry(monkeypatch):
    import app.api.text_chat as api
    llm=AsyncMock(side_effect=[{'content':'bad JSON','prompt_tokens':3,'completion_tokens':2},
                             *responses('What was overlapping?')])
    monkeypatch.setattr(api,'_call_llm',llm)
    result=await api._reflective_agent_turn([{'role':'user','content':'The cards overlapped.'}],
        model='test',study_id=None,guide=GUIDE,research_spec={},remaining_seconds=600)
    assert result['content']=='What was overlapping?'
    assert result['telemetry']['format_retries']==1
    assert result['telemetry']['model_calls']==3
    assert result['prompt_tokens']==23
    assert llm.await_args_list[0].args[0]==llm.await_args_list[1].args[0]


def test_reflective_guide_requires_explicit_brief():
    assert InterviewGuide.model_validate(GUIDE).conversation_policy=='reflective_v1'
    with pytest.raises(ValueError,match='conversation_brief'):
        InterviewGuide.model_validate({**GUIDE,'conversation_brief':None})
