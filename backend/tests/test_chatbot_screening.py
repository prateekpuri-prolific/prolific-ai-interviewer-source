import json
import re
import time

import pytest
from sqlalchemy import select

from app.api.chatbot_screening import signed_token
from app.models.agent import Agent, AgentStatus
from app.models.study import Study, StudyStatus
from app.models.session import Session


@pytest.fixture
async def screening(db_session):
    config = {'external_study_id': 'a'*24,
              'eligible_completion_url': 'https://app.prolific.com/submissions/complete?cc=ELIGIBLE',
              'other_completion_url': 'https://app.prolific.com/submissions/complete?cc=OTHER'}
    study = Study(title='Paid screener', status=StudyStatus.ACTIVE, research_spec={
        'version': 'chatbot-screen-v1', 'chatbot_screening': config, 'participant_information': {}})
    db_session.add(study); await db_session.flush()
    agent = Agent(study_id=study.id, name='Screener', status=AgentStatus.ACTIVE)
    db_session.add(agent); await db_session.commit()
    return agent.widget_key, config


def payload(widget, **changes):
    fields = {'participant_id': 'b'*24, 'external_study_id': 'a'*24, 'external_session_id': 'c'*24,
              'started': int(time.time()), 'consent': True, 'claude': 'no', 'others': ['chatgpt']}
    fields.update(changes)
    fields['token'] = signed_token(widget, fields['participant_id'], fields['external_study_id'], fields['external_session_id'], fields['started'])
    return fields


@pytest.mark.parametrize('claude,others,eligible', [('no',['chatgpt'],True), ('yes',['chatgpt'],False),
    ('unsure',['gemini'],False), ('no',['none'],False), ('no',['unsure'],False)])
async def test_outcomes_are_saved_and_all_return_paid_codes(client, db_session, screening, claude, others, eligible):
    widget, config = screening
    r = await client.post('/api/screeners/'+widget, json=payload(widget, claude=claude, others=others))
    assert r.status_code == 200
    assert r.json()['completion_url'] == config['eligible_completion_url' if eligible else 'other_completion_url']
    saved = (await db_session.execute(select(Session))).scalar_one()
    assert saved.completion_metadata['screening']['eligible'] is eligible
    assert saved.participant_consent_confirmed
    assert saved.protocol_snapshot['research_spec']['version'] == 'chatbot-screen-v1'


async def test_retry_is_idempotent_and_cannot_change_eligibility(client, db_session, screening):
    widget, config = screening
    first = await client.post('/api/screeners/'+widget, json=payload(widget, claude='yes'))
    second = await client.post('/api/screeners/'+widget, json=payload(widget, claude='no'))
    assert first.json() == second.json() == {'completion_url': config['other_completion_url']}
    assert len((await db_session.execute(select(Session))).scalars().all()) == 1


@pytest.mark.parametrize('changes,code', [({'consent':False},422), ({'others':['none','chatgpt']},422),
    ({'external_study_id':'d'*24},403), ({'started':int(time.time())-8000},403)])
async def test_invalid_submission_never_saves(client, db_session, screening, changes, code):
    widget,_ = screening
    r = await client.post('/api/screeners/'+widget,json=payload(widget,**changes))
    assert r.status_code == code
    assert not (await db_session.execute(select(Session))).scalars().all()


async def test_tampering_with_signed_identity_is_rejected(client, screening):
    widget,_ = screening
    p = payload(widget); p['participant_id'] = 'e'*24
    assert (await client.post('/api/screeners/'+widget,json=p)).status_code == 403


async def test_form_binds_prolific_ids_without_exposing_completion_codes(client, screening):
    widget,_ = screening
    r=await client.get('/api/screeners/'+widget,params={'PROLIFIC_PID':'b'*24,'STUDY_ID':'a'*24,'SESSION_ID':'c'*24})
    assert r.status_code == 200
    assert 'ELIGIBLE' not in r.text and 'cc=OTHER' not in r.text
    assert r.headers['cache-control'] == 'no-store'
    context=json.loads(re.search(r'const context=(.*?);',r.text).group(1))
    assert context['participant_id'] == 'b'*24
    assert (await client.get('/api/screeners/'+widget)).status_code == 400


async def test_older_cohort_does_not_require_other_chatbot_use(client, db_session, screening):
    widget, config = screening
    study = (await db_session.execute(select(Study))).scalar_one()
    spec = dict(study.research_spec)
    spec['chatbot_screening'] = {**config, 'require_other_chatbot': False}
    study.research_spec = spec
    await db_session.commit()
    r = await client.post('/api/screeners/'+widget,json=payload(widget, others=['none']))
    assert r.status_code == 200
    assert r.json()['completion_url'] == config['eligible_completion_url']
