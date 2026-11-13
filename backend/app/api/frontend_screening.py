"""Paid, deterministic frontend eligibility screener for Prolific."""

import hashlib
import hmac
import json
import re
import time
from datetime import datetime, timezone
from html import escape
from typing import Literal
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models.agent import Agent, AgentStatus
from app.models.study import Study, StudyStatus
from app.models.session import Session, SessionStatus, TranscriptEntry, SpeakerRole
from app.models.privacy_lifecycle import PrivacyDeletionReceipt

router = APIRouter(prefix="/frontend-screen", tags=["screeners"])

PROFESSIONS = {
    "frontend_developer": "Frontend developer or engineer",
    "ui_designer": "UX/UI or product designer",
    "fullstack_developer": "Full-stack developer or engineer",
    "web_designer": "Web designer",
    "design_engineer": "Design engineer or creative developer",
    "backend_developer": "Backend developer or engineer",
    "graphic_designer": "Visual or graphic designer",
    "ux_researcher": "UX researcher",
    "product_manager": "Product manager",
    "data_role": "Data analyst, scientist, or engineer",
    "other": "Another profession",
}
TOOLS = {
    "chatgpt": "ChatGPT",
    "claude": "Claude or Claude Code",
    "cursor": "Cursor",
    "copilot": "GitHub Copilot",
    "codex": "Codex",
    "gemini": "Gemini or Gemini Code Assist",
    "windsurf": "Windsurf",
    "vibe_builder": "v0, Lovable, Bolt, or a similar UI builder",
    "figma_ai": "Figma AI",
    "other": "Another AI tool",
    "none": "None of these",
}
QUALIFYING_PROFESSIONS = {
    "frontend_developer", "ui_designer", "fullstack_developer", "web_designer", "design_engineer"
}


class FrontendScreeningAnswer(BaseModel):
    participant_id: str = Field(pattern=r"^[a-f0-9]{24}$")
    external_study_id: str = Field(pattern=r"^[a-f0-9]{24}$")
    external_session_id: str = Field(pattern=r"^[a-f0-9]{24}$")
    started: int
    token: str = Field(min_length=64, max_length=64)
    consent: Literal[True]
    profession: Literal[tuple(PROFESSIONS)]
    ui_experience: Literal["under_one", "one_two", "three_five", "six_plus"]
    hands_on_code: Literal["yes", "no"]
    ai_for_frontend: Literal["yes", "no"]
    ai_tools: list[Literal[tuple(TOOLS)]] = Field(min_length=1, max_length=len(TOOLS))
    visual_judgment: Literal["often", "sometimes", "rarely_never"]
    challenge_ready: Literal["yes", "no"]


def signed_token(widget: str, pid: str, study_id: str, session_id: str, started: int) -> str:
    payload = json.dumps(
        ["frontend-screen-v1", widget, pid, study_id, session_id, started],
        separators=(",", ":"),
    )
    return hmac.new(settings.secret_key.encode(), payload.encode(), hashlib.sha256).hexdigest()


def assess(answer: FrontendScreeningAnswer) -> list[str]:
    reasons = []
    if answer.profession not in QUALIFYING_PROFESSIONS:
        reasons.append("profession_outside_frontend_ui")
    if answer.ui_experience in {"under_one", "one_two"}:
        reasons.append("under_three_years_ui_experience")
    if answer.hands_on_code != "yes":
        reasons.append("no_recent_personal_ui_code")
    if answer.ai_for_frontend != "yes":
        reasons.append("no_recent_ai_frontend_use")
    if not (set(answer.ai_tools) - {"none"}):
        reasons.append("no_work_ai_tool")
    if answer.visual_judgment == "rarely_never":
        reasons.append("no_visual_interaction_judgment")
    if answer.challenge_ready != "yes":
        reasons.append("not_ready_to_discuss_recent_challenge")
    return reasons


async def context(widget: str, db: AsyncSession):
    agent = (await db.execute(select(Agent).where(Agent.widget_key == widget))).scalar_one_or_none()
    if not agent or agent.status != AgentStatus.ACTIVE:
        raise HTTPException(404, "This questionnaire is unavailable.")
    study = await db.get(Study, agent.study_id)
    config = (study.research_spec or {}).get("frontend_screening") if study else None
    if study.status != StudyStatus.ACTIVE or not config:
        raise HTTPException(404, "This questionnaire is unavailable.")
    return agent, study, config


def _next_url(config: dict, answer: FrontendScreeningAnswer, eligible: bool) -> str:
    if not eligible:
        return config["screened_out_completion_url"]
    params = urlencode({
        "PROLIFIC_PID": answer.participant_id,
        "STUDY_ID": answer.external_study_id,
        "SESSION_ID": answer.external_session_id,
    })
    return f"/interview/{config['interview_widget_key']}?{params}"


@router.get("/{widget}", response_class=HTMLResponse)
async def form(widget: str, PROLIFIC_PID: str = "", STUDY_ID: str = "", SESSION_ID: str = "", db: AsyncSession = Depends(get_db)):
    _, study, config = await context(widget, db)
    if STUDY_ID != config["external_study_id"] or not all(
        re.fullmatch(r"[a-f0-9]{24}", value)
        for value in (PROLIFIC_PID, STUDY_ID, SESSION_ID)
    ):
        raise HTTPException(400, "Please open this questionnaire from Prolific.")
    started = int(time.time())
    fields = {
        "participant_id": PROLIFIC_PID,
        "external_study_id": STUDY_ID,
        "external_session_id": SESSION_ID,
        "started": started,
        "token": signed_token(widget, PROLIFIC_PID, STUDY_ID, SESSION_ID, started),
    }
    info = study.research_spec.get("participant_information") or {}
    paragraphs = "".join(f"<p>{escape(str(info.get(key) or ''))}</p>" for key in (
        "purpose_summary", "participation_summary", "compensation_summary",
        "data_handling_summary", "data_retention_period", "withdrawal_instructions",
        "researcher_contact",
    ))
    professions = "".join(
        f'<label><input type="radio" name="profession" value="{key}" required> {escape(label)}</label>'
        for key, label in PROFESSIONS.items()
    )
    tools = "".join(
        f'<label><input type="checkbox" name="ai_tools" value="{key}"> {escape(label)}</label>'
        for key, label in TOOLS.items()
    )
    html = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Frontend work questionnaire</title>
<style>body{font:16px/1.5 system-ui,sans-serif;background:#f5f3ef;color:#192234;margin:0}main{max-width:720px;margin:30px auto;padding:30px;background:white;border-radius:16px}h1{font-size:28px;line-height:1.2}h2{font-size:18px}fieldset{border:0;border-top:1px solid #ddd;padding:20px 0;margin:24px 0}legend{font-weight:700;padding-right:12px}label{display:block;padding:7px 0}input{margin-right:8px}button,.return{display:inline-block;background:#235b67;color:white;border:0;border-radius:8px;padding:12px 18px;font:inherit;cursor:pointer;text-decoration:none}button:disabled{opacity:.5}#error{color:#ad2424}details{font-size:14px}a{color:#235b67}@media(max-width:740px){main{margin:0;border-radius:0;padding:22px}}</style></head><body><main>
<h1>About your UI and AI work</h1><p>This short questionnaire determines whether this interview matches your recent work. Please answer accurately. If it is not a match, you will receive $0.50 for this questionnaire. If it is a match, you can continue to the full interview for the listed $12.50 reward.</p>
<details><summary>Study information and privacy</summary>''' + paragraphs + '''<p>Prolific Academic Ltd is the data controller. <a href="https://www.prolific.com/privacy-and-legal" target="_blank" rel="noopener">Privacy and legal information</a>.</p></details>
<form id="screen"><label><input type="checkbox" id="consent" required> I have read the study information and agree to take part.</label>
<fieldset><legend>1. Which best describes your main professional work?</legend>''' + professions + '''</fieldset>
<fieldset><legend>2. For how many years have you personally built or implemented frontend user interfaces as part of your work?</legend><label><input type="radio" name="ui_experience" value="under_one" required> Less than 1 year</label><label><input type="radio" name="ui_experience" value="one_two"> 1–2 years</label><label><input type="radio" name="ui_experience" value="three_five"> 3–5 years</label><label><input type="radio" name="ui_experience" value="six_plus"> 6 or more years</label></fieldset>
<fieldset><legend>3. In the last three months, have you personally written or edited frontend code for a shipped or release-ready user interface?</legend><label><input type="radio" name="hands_on_code" value="yes" required> Yes</label><label><input type="radio" name="hands_on_code" value="no"> No</label></fieldset>
<fieldset><legend>4. In the last three months, have you used AI while designing or implementing frontend UI work that you personally did?</legend><label><input type="radio" name="ai_for_frontend" value="yes" required> Yes</label><label><input type="radio" name="ai_for_frontend" value="no"> No</label></fieldset>
<fieldset><legend>5. Which AI tools have you used for that work? Select all that apply.</legend>''' + tools + '''</fieldset>
<fieldset><legend>6. How often does your frontend work involve judging visual or interaction quality, such as layout, typography, motion, brand fit, or overall feel?</legend><label><input type="radio" name="visual_judgment" value="often" required> Often</label><label><input type="radio" name="visual_judgment" value="sometimes"> Sometimes</label><label><input type="radio" name="visual_judgment" value="rarely_never"> Rarely or never</label></fieldset>
<fieldset><legend>7. Are you ready to discuss a specific recent challenge where AI missed the visual, interaction, or aesthetic intent of that frontend work?</legend><label><input type="radio" name="challenge_ready" value="yes" required> Yes</label><label><input type="radio" name="challenge_ready" value="no"> No</label></fieldset>
<p id="error" role="alert"></p><button id="submit">Submit answers</button></form>
<section id="done" hidden><h2 id="outcome"></h2><p id="next-text"></p><a id="next" class="return"></a></section>
<p><a href="https://app.prolific.com/studies">Leave without submitting and return to Prolific</a></p>
<script>const context=''' + json.dumps(fields) + ''';
document.getElementById('screen').addEventListener('submit',async e=>{e.preventDefault();const error=document.getElementById('error');const button=document.getElementById('submit');const ai_tools=[...document.querySelectorAll('input[name="ai_tools"]:checked')].map(x=>x.value);error.textContent='';if(!ai_tools.length||ai_tools.includes('none')&&ai_tools.length>1){error.textContent='Choose the tools used, or choose only None of these.';return;}button.disabled=true;try{const body={...context,consent:document.getElementById('consent').checked,profession:document.querySelector('input[name="profession"]:checked').value,ui_experience:document.querySelector('input[name="ui_experience"]:checked').value,hands_on_code:document.querySelector('input[name="hands_on_code"]:checked').value,ai_for_frontend:document.querySelector('input[name="ai_for_frontend"]:checked').value,ai_tools,visual_judgment:document.querySelector('input[name="visual_judgment"]:checked').value,challenge_ready:document.querySelector('input[name="challenge_ready"]:checked').value};const response=await fetch(location.pathname,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const data=await response.json();if(!response.ok)throw new Error(typeof data.detail==='string'?data.detail:'Please check your answers or message the researcher on Prolific.');document.getElementById('screen').hidden=true;document.getElementById('done').hidden=false;document.getElementById('next').href=data.next_url;if(data.eligible){document.getElementById('outcome').textContent='You can continue to the interview';document.getElementById('next-text').textContent='Your answers have been saved. Continue to the full typed interview.';document.getElementById('next').textContent='Continue to interview';}else{document.getElementById('outcome').textContent='Thank you for your time';document.getElementById('next-text').textContent='This interview is not a match for your recent work. Return to Prolific to receive the $0.50 screening payment.';document.getElementById('next').textContent='Return to Prolific';}}catch(err){error.textContent=err.message;}finally{button.disabled=false;}});
</script></main></body></html>'''
    return HTMLResponse(html, headers={"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"})


@router.post("/{widget}")
async def submit(widget: str, answer: FrontendScreeningAnswer, db: AsyncSession = Depends(get_db)):
    agent, study, config = await context(widget, db)
    expected = signed_token(widget, answer.participant_id, answer.external_study_id,
                            answer.external_session_id, answer.started)
    elapsed = int(time.time()) - answer.started
    if not hmac.compare_digest(answer.token, expected) or not 0 <= elapsed <= 7200 or answer.external_study_id != config["external_study_id"]:
        raise HTTPException(403, "This questionnaire link has expired. Please reopen it from Prolific.")
    if len(answer.ai_tools) != len(set(answer.ai_tools)) or "none" in answer.ai_tools and len(answer.ai_tools) > 1:
        raise HTTPException(422, "Choose the tools used, or choose only None of these.")
    stmt = select(Session).where(
        Session.agent_id == agent.id,
        Session.external_session_id == answer.external_session_id,
    )
    existing = (await db.execute(stmt)).scalar_one_or_none()
    if existing:
        if existing.participant_id != answer.participant_id:
            raise HTTPException(409, "This submission has already been used.")
        deleted = (await db.execute(select(PrivacyDeletionReceipt).where(
            PrivacyDeletionReceipt.session_id == existing.id))).scalar_one_or_none()
        if deleted:
            raise HTTPException(410, "This submission has been deleted.")
        saved = existing.completion_metadata["screening"]
        return {"eligible": saved["eligible"],
                "next_url": _next_url(config, answer, saved["eligible"])}
    reasons = assess(answer)
    eligible = not reasons
    next_url = _next_url(config, answer, eligible)
    now = datetime.now(timezone.utc)
    session = Session(
        agent_id=agent.id, participant_id=answer.participant_id,
        recruitment_source="prolific", external_study_id=answer.external_study_id,
        external_session_id=answer.external_session_id, status=SessionStatus.COMPLETED,
        duration_seconds=float(elapsed), ended_at=now,
        participant_consent_confirmed=True, participant_consent_at=now,
        research_protocol_version="frontend-screen-v1", research_stage="field",
        protocol_snapshot={"research_spec": study.research_spec},
        completion_metadata={"reason": "screening_complete", "screening": {
            "eligible": eligible, "reasons": reasons, "profession": answer.profession,
            "ui_experience": answer.ui_experience,
            "hands_on_code": answer.hands_on_code,
            "ai_for_frontend": answer.ai_for_frontend,
            "ai_tools": answer.ai_tools, "visual_judgment": answer.visual_judgment,
            "challenge_ready": answer.challenge_ready,
        }},
    )
    db.add(session)
    try:
        await db.flush()
    except IntegrityError:
        await db.rollback()
        existing = (await db.execute(stmt)).scalar_one_or_none()
        if existing and existing.participant_id == answer.participant_id:
            saved = existing.completion_metadata["screening"]
            return {"eligible": saved["eligible"],
                    "next_url": _next_url(config, answer, saved["eligible"])}
        raise HTTPException(409, "This submission has already been used.")
    transcript = [
        ("Which best describes your main professional work?", PROFESSIONS[answer.profession]),
        ("For how many years have you personally built frontend user interfaces?", answer.ui_experience),
        ("Have you personally written or edited frontend UI code in the last three months?", answer.hands_on_code),
        ("Have you used AI for your frontend UI design or implementation work in the last three months?", answer.ai_for_frontend),
        ("Which AI tools have you used for that work?", ", ".join(TOOLS[x] for x in answer.ai_tools)),
        ("How often do you judge visual or interaction quality in that work?", answer.visual_judgment),
        ("Are you ready to discuss a specific recent AI challenge in that work?", answer.challenge_ready),
    ]
    for index, (question, response) in enumerate(transcript):
        db.add(TranscriptEntry(session_id=session.id, sequence=index * 2 + 1,
                               role=SpeakerRole.AGENT, content=question))
        db.add(TranscriptEntry(session_id=session.id, sequence=index * 2 + 2,
                               role=SpeakerRole.USER, content=response))
    await db.commit()
    return {"eligible": eligible, "next_url": next_url}
