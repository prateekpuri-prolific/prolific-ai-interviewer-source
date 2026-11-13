import time
import json
import uuid

import pytest
from sqlalchemy import select

from app.api.frontend_screening import signed_token
from app.models.agent import Agent, AgentStatus
from app.models.study import Study, StudyStatus
from app.models.session import Session, TranscriptEntry
from app.schemas.study import StudyUpdate


def test_interview_gate_id_is_json_serializable():
    payload = StudyUpdate.model_validate({"research_spec": {
        "required_screener_agent_id": str(uuid.uuid4()),
    }})
    json.dumps(payload.model_dump(exclude_unset=True))


@pytest.fixture
async def screener(db_session):
    config = {
        "external_study_id": "a" * 24,
        "interview_widget_key": "interview_widget_123",
        "screened_out_completion_url": "https://app.prolific.com/submissions/complete?cc=SCREENOUT",
    }
    study = Study(title="Frontend screener", status=StudyStatus.ACTIVE, research_spec={
        "version": "frontend-screen-v1", "frontend_screening": config,
        "participant_information": {"compensation_summary": "Screen-outs receive $0.50."},
    })
    db_session.add(study)
    await db_session.flush()
    agent = Agent(study_id=study.id, name="Screener", status=AgentStatus.ACTIVE)
    db_session.add(agent)
    await db_session.commit()
    return agent.widget_key, config


def payload(widget, **changes):
    result = {
        "participant_id": "b" * 24,
        "external_study_id": "a" * 24,
        "external_session_id": "c" * 24,
        "started": int(time.time()),
        "consent": True,
        "profession": "ui_designer",
        "hands_on_code": "yes",
        "ai_for_frontend": "yes",
        "ai_tools": ["cursor", "chatgpt"],
        "challenge_ready": "yes",
    }
    result.update(changes)
    result["token"] = signed_token(
        widget, result["participant_id"], result["external_study_id"],
        result["external_session_id"], result["started"])
    return result


@pytest.mark.parametrize("changes,eligible", [
    ({}, True),
    ({"profession": "frontend_developer"}, True),
    ({"profession": "graphic_designer"}, False),
    ({"hands_on_code": "no"}, False),
    ({"ai_for_frontend": "no"}, False),
    ({"ai_tools": ["none"]}, False),
    ({"challenge_ready": "no"}, False),
])
async def test_screener_routes_and_records_decision(client, db_session, screener, changes, eligible):
    widget, config = screener
    response = await client.post("/api/frontend-screen/" + widget, json=payload(widget, **changes))
    assert response.status_code == 200
    assert response.json()["eligible"] is eligible
    if eligible:
        assert response.json()["next_url"].startswith("/interview/interview_widget_123?")
    else:
        assert response.json()["next_url"] == config["screened_out_completion_url"]
    saved = (await db_session.execute(select(Session))).scalar_one()
    assert saved.completion_metadata["screening"]["eligible"] is eligible
    assert saved.participant_consent_confirmed
    assert len((await db_session.execute(select(TranscriptEntry))).scalars().all()) == 10


async def test_form_contains_all_questions_and_hides_screenout_code(client, screener):
    widget, _ = screener
    response = await client.get("/api/frontend-screen/" + widget, params={
        "PROLIFIC_PID": "b" * 24, "STUDY_ID": "a" * 24, "SESSION_ID": "c" * 24})
    assert response.status_code == 200
    for text in ("Which best describes", "personally written or edited frontend code",
                 "Which AI tools", "specific recent challenge", "$0.50"):
        assert text in response.text
    assert "SCREENOUT" not in response.text
    assert response.headers["cache-control"] == "no-store"
    assert (await client.get("/api/frontend-screen/" + widget)).status_code == 400


async def test_retry_cannot_change_outcome(client, db_session, screener):
    widget, config = screener
    first = await client.post("/api/frontend-screen/" + widget,
                              json=payload(widget, profession="backend_developer"))
    second = await client.post("/api/frontend-screen/" + widget, json=payload(widget))
    assert first.json() == second.json() == {
        "eligible": False, "next_url": config["screened_out_completion_url"]}
    assert len((await db_session.execute(select(Session))).scalars().all()) == 1


@pytest.mark.parametrize("changes,code", [
    ({"consent": False}, 422),
    ({"external_study_id": "d" * 24}, 403),
    ({"started": int(time.time()) - 8000}, 403),
    ({"ai_tools": ["none", "cursor"]}, 422),
])
async def test_invalid_answers_do_not_save(client, db_session, screener, changes, code):
    widget, _ = screener
    response = await client.post("/api/frontend-screen/" + widget,
                                 json=payload(widget, **changes))
    assert response.status_code == code
    assert not (await db_session.execute(select(Session))).scalars().all()
