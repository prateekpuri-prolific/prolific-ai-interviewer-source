"""Short, paid, deterministic chatbot eligibility questionnaire. No model calls.

Uses the existing consent, session export/deletion and retention infrastructure.
The two completion paths both pay; eligibility only determines a future invite.
"""
import hashlib
import hmac
import json
import time
from datetime import datetime, timezone
from html import escape
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, model_validator
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.cohort_screening import verification_questions, assess_cohort, CONTEXTS
from app.config import settings
from app.database import get_db
from app.models.agent import Agent, AgentStatus
from app.models.study import Study, StudyStatus
from app.models.session import Session, SessionStatus, TranscriptEntry, SpeakerRole
from app.models.privacy_lifecycle import PrivacyDeletionReceipt

router = APIRouter(prefix="/screeners", tags=["screeners"])
OPTIONS = {"chatgpt": "ChatGPT", "gemini": "Gemini / Bard", "copilot": "Microsoft Copilot / Bing Chat",
           "meta": "Meta AI", "deepseek": "DeepSeek", "perplexity": "Perplexity",
           "other": "Another conversational AI chatbot", "none": "None", "unsure": "Not sure / prefer not to say"}


class ScreeningAnswer(BaseModel):
    participant_id: str = Field(pattern=r"^[a-f0-9]{24}$")
    external_study_id: str = Field(pattern=r"^[a-f0-9]{24}$")
    external_session_id: str = Field(pattern=r"^[a-f0-9]{24}$")
    started: int
    token: str = Field(min_length=64, max_length=64)
    consent: Literal[True]
    claude: Literal["yes", "no", "unsure"]
    others: list[Literal["chatgpt", "gemini", "copilot", "meta", "deepseek", "perplexity", "other", "none", "unsure"]] = Field(min_length=1, max_length=9)

    verification: dict[str, str | list[str]] = Field(default_factory=dict, max_length=12)

    @model_validator(mode="after")
    def exclusive_answers(self):
        if len(set(self.others)) != len(self.others) or (set(self.others) & {"none", "unsure"} and len(self.others) != 1):
            raise ValueError("Select either the chatbots used, None, or Not sure.")
        return self

    @property
    def eligible(self):
        return self.claude == "no" and not (set(self.others) & {"none", "unsure"})


def signed_token(widget, pid, study_id, session_id, started):
    payload = json.dumps(["chatbot-screen-v1", widget, pid, study_id, session_id, started], separators=(",", ":"))
    return hmac.new(settings.secret_key.encode(), payload.encode(), hashlib.sha256).hexdigest()


async def context(widget, db):
    agent = (await db.execute(select(Agent).where(Agent.widget_key == widget))).scalar_one_or_none()
    if not agent or agent.status != AgentStatus.ACTIVE:
        raise HTTPException(404, "This questionnaire is not available.")
    study = await db.get(Study, agent.study_id)
    config = (study.research_spec or {}).get("chatbot_screening")
    if study.status != StudyStatus.ACTIVE or not config:
        raise HTTPException(404, "This questionnaire is not available.")
    return agent, study, config


@router.get("/{widget}", response_class=HTMLResponse)
async def form(widget: str, PROLIFIC_PID: str = "", STUDY_ID: str = "", SESSION_ID: str = "", db: AsyncSession = Depends(get_db)):
    import re
    agent, study, config = await context(widget, db)
    if STUDY_ID != config['external_study_id'] or not all(re.fullmatch(r"[a-f0-9]{24}", x) for x in [PROLIFIC_PID, STUDY_ID, SESSION_ID]):
        raise HTTPException(400, "Please open this questionnaire from your Prolific study.")
    started = int(time.time())
    info = study.research_spec['participant_information']
    fields = {"participant_id": PROLIFIC_PID, "external_study_id": STUDY_ID, "external_session_id": SESSION_ID,
              "started": started, "token": signed_token(widget, PROLIFIC_PID, STUDY_ID, SESSION_ID, started)}
    paragraphs = ''.join(f'<p>{escape(info.get(k, ""))}</p>' for k in [
        'purpose_summary', 'participation_summary', 'compensation_summary', 'data_handling_summary',
        'data_recipients_summary', 'data_retention_period', 'withdrawal_instructions', 'researcher_contact'])
    options = ''.join(f'<label><input type="checkbox" name="others" value="{key}"> {escape(label)}</label>' for key, label in OPTIONS.items())
    verification_html = ""
    if config.get("cohort"):
        for key, title, values in verification_questions(config['cohort']):
            radios = ''.join(f'<label><input type="radio" name="verify_{key}" value="{v}" required> {escape(label)}</label>' for v, label in values.items())
            verification_html += f'<fieldset><legend>{escape(title)}</legend>{radios}</fieldset>'
        choices = ''.join(f'<label><input type="checkbox" name="contexts" value="{v}"> {escape(label)}</label>' for v, label in CONTEXTS.items())
        verification_html += '<fieldset><legend>In which contexts have you intentionally used a chatbot? Select all that apply.</legend>' + choices + '</fieldset>'
    # IDs contain only hex and the signed payload has no free-form participant data.
    html = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Chatbot experience questionnaire</title>
<style>body{font:17px/1.6 system-ui,sans-serif;color:#182b2a;background:#f5f7f6;margin:0}main{max-width:680px;margin:32px auto;padding:28px;background:white;border-radius:18px}h1{font-size:28px;line-height:1.2}h2,legend{font-size:19px;font-weight:650}label{display:block;padding:9px 0}input{width:18px;height:18px;vertical-align:middle}fieldset{border:0;border-top:1px solid #ddd;margin:24px 0;padding:22px 0}button,.return{display:inline-block;background:#17635b;color:white;border:0;border-radius:9px;padding:13px 20px;font:inherit;cursor:pointer}button:disabled{opacity:.5}#error{color:#a02020}details{font-size:14px}a{color:#17635b}</style></head><body><main>
<h1>A short questionnaire about chatbot experience</h1><p>Thank you for helping us include a wider range of experiences in our research.</p>
<details open><summary>Study information</summary>''' + paragraphs + '''<p>Prolific Academic Ltd is the data controller. <a href="https://www.prolific.com/privacy-and-legal" target="_blank" rel="noopener">Privacy and legal information</a>.</p></details>
<form id="screen"><label><input type="checkbox" id="consent" required> I have read the information above and agree to take part.</label>
<fieldset><legend>1. Have you ever personally used Claude, the AI chatbot made by Anthropic?</legend><p>Include Claude's website or app, Claude Code, or Claude through another service. Include use at work, in education, or in paid research and AI evaluation, even if only once. Hearing about Claude or watching someone else use it does not count.</p>
<label><input type="radio" name="claude" value="yes" required> Yes</label><label><input type="radio" name="claude" value="no"> No</label><label><input type="radio" name="claude" value="unsure"> Not sure / prefer not to say</label></fieldset>
<fieldset><legend>2. Which other AI chatbots have you personally exchanged messages with?</legend><p>Select all that apply, even if only once, including at work or in paid research. Count Claude use in the question above. Do not count ordinary web searches, automatically displayed search summaries, watching someone else, or this questionnaire.</p>''' + options + '''</fieldset>''' + verification_html + '''
<p>Your payment is the same regardless of your answers. A follow-up interview invitation is not guaranteed.</p><p id="error" role="alert"></p><button id="submit">Submit questionnaire</button></form>
<section id="done" hidden><h2>Thank you—your answers have been saved.</h2><p>Please return to Prolific to submit this paid questionnaire.</p><a id="return" class="return">Return to Prolific</a></section>
<p><a href="https://app.prolific.com/studies">Leave without submitting and return to Prolific</a></p>'''
    if settings.source_code_url:
        html += '<p><a href="' + escape(settings.source_code_url, quote=True) + '">Source code</a></p>'
    html += '<script>const context=' + json.dumps(fields) + ''';
document.getElementById('screen').addEventListener('submit',async e=>{e.preventDefault();const b=document.getElementById('submit');const error=document.getElementById('error');const others=[...document.querySelectorAll('input[name="others"]:checked')].map(x=>x.value);error.textContent='';if(!others.length||others.length>1&&others.some(x=>['none','unsure'].includes(x))){error.textContent='Please select the chatbots used, or select only None or Not sure.';return;}const verification={};document.querySelectorAll('input[name^="verify_"]:checked').forEach(x=>verification[x.name.slice(7)]=x.value);if(document.querySelector('input[name="contexts"]'))verification.contexts=[...document.querySelectorAll('input[name="contexts"]:checked')].map(x=>x.value);b.disabled=true;try{const r=await fetch(location.pathname,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...context,consent:document.getElementById('consent').checked,claude:document.querySelector('input[name="claude"]:checked').value,others,verification})});const data=await r.json();if(!r.ok)throw new Error(typeof data.detail==='string'?data.detail:'Please check your answers and try again, or message the researcher on Prolific.');document.getElementById('return').href=data.completion_url;document.getElementById('screen').hidden=true;document.getElementById('done').hidden=false;}catch(e){error.textContent=e.message;}finally{b.disabled=false;}});
</script></main></body></html>'''
    return HTMLResponse(html, headers={"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"})


@router.post("/{widget}")
async def submit(widget: str, answer: ScreeningAnswer, db: AsyncSession = Depends(get_db)):
    agent, study, config = await context(widget, db)
    expected = signed_token(widget, answer.participant_id, answer.external_study_id, answer.external_session_id, answer.started)
    elapsed = int(time.time()) - answer.started
    if not hmac.compare_digest(answer.token, expected) or elapsed < 0 or elapsed > 7200 or answer.external_study_id != config['external_study_id']:
        raise HTTPException(403, "This questionnaire link has expired. Please reopen it from Prolific.")
    stmt = select(Session).where(Session.agent_id == agent.id, Session.external_session_id == answer.external_session_id)
    existing = (await db.execute(stmt)).scalar_one_or_none()
    if existing:
        if existing.participant_id != answer.participant_id:
            raise HTTPException(409, "This submission has already been used.")
        deleted = (await db.execute(select(PrivacyDeletionReceipt).where(PrivacyDeletionReceipt.session_id == existing.id))).scalar_one_or_none()
        if deleted:
            raise HTTPException(410, "This submission has been deleted.")
        return {"completion_url": existing.completion_metadata['screening']['completion_url']}
    now = datetime.now(timezone.utc)
    eligible = answer.claude == 'no' and (answer.eligible or not config.get('require_other_chatbot', True))
    assessment = {}
    if config.get('cohort'):
        try:
            assessment = assess_cohort(config, answer.claude, answer.others, answer.verification)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        eligible = assessment['eligible']
    version = 'cohort-screen-v2' if config.get('cohort') else 'chatbot-screen-v1'
    completion = config['eligible_completion_url'] if eligible else config['other_completion_url']
    session = Session(agent_id=agent.id, participant_id=answer.participant_id, recruitment_source='prolific',
        external_study_id=answer.external_study_id, external_session_id=answer.external_session_id,
        status=SessionStatus.COMPLETED, duration_seconds=float(elapsed), ended_at=now,
        participant_consent_confirmed=True, participant_consent_at=now,
        research_protocol_version=version, research_stage='pilot',
        protocol_snapshot={'research_spec': study.research_spec, 'screening_version': version},
        completion_metadata={'reason': 'screening_complete', 'screening': {
            **assessment, 'eligible': eligible, 'claude_ever': answer.claude, 'other_chatbots': answer.others,
            'verification': answer.verification,
            'completion_url': completion, 'payment_independent_of_eligibility': True}})
    db.add(session)
    try:
        await db.flush()
    except IntegrityError:
        await db.rollback()
        existing = (await db.execute(stmt)).scalar_one_or_none()
        if existing and existing.participant_id == answer.participant_id:
            return {'completion_url': existing.completion_metadata['screening']['completion_url']}
        raise HTTPException(409, 'This submission has already been used.')
    for seq, role, content in [
        (1, SpeakerRole.AGENT, 'Have you ever personally used Claude, the AI chatbot made by Anthropic?'),
        (2, SpeakerRole.USER, answer.claude),
        (3, SpeakerRole.AGENT, 'Which other AI chatbots have you personally exchanged messages with?'),
        (4, SpeakerRole.USER, ', '.join(OPTIONS[x] for x in answer.others)),
    ]:
        db.add(TranscriptEntry(session_id=session.id, sequence=seq, role=role, content=content))
    if config.get('cohort'):
        for seq, (key, question, options) in enumerate(verification_questions(config['cohort']), 3):
            db.add(TranscriptEntry(session_id=session.id, sequence=seq*2-1, role=SpeakerRole.AGENT, content=question))
            db.add(TranscriptEntry(session_id=session.id, sequence=seq*2, role=SpeakerRole.USER, content=options[answer.verification[key]]))
        seq = len(verification_questions(config['cohort'])) * 2 + 5
        db.add(TranscriptEntry(session_id=session.id, sequence=seq, role=SpeakerRole.AGENT, content='In which contexts have you intentionally used a chatbot?'))
        db.add(TranscriptEntry(session_id=session.id, sequence=seq+1, role=SpeakerRole.USER, content=', '.join(CONTEXTS[x] for x in answer.verification['contexts'])))
    await db.commit()
    return {'completion_url': completion}
