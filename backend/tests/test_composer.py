"""Study composer API contract and non-mutating draft behavior."""

from types import SimpleNamespace

from httpx import AsyncClient

from app.api.composer import (
    AuditedComposerDraft,
    ComposerAuditReport,
    ComposerDraft,
    ComposerStudyFoundation,
    QuestionComposerDraft,
    _AUDIT_INSTRUCTIONS,
    _COMPOSER_INSTRUCTIONS,
    _FOUNDATION_INSTRUCTIONS,
    _QUESTION_COMPOSER_INSTRUCTIONS,
    _STAGED_DRAFT_INSTRUCTIONS,
    _compose_with_openai,
    _edited_question_context,
    _normalize_question_metadata,
)


def test_composer_prompts_define_schema_enum_contract():
    assert 'priority is only\n  "required" or "optional"' in _COMPOSER_INSTRUCTIONS
    assert 'sensitivity is only "standard" or "elevated"' in _COMPOSER_INSTRUCTIONS
    assert 'priority is only\n  "required" or "optional"' in _QUESTION_COMPOSER_INSTRUCTIONS
    assert "structured clarification_policy" in _COMPOSER_INSTRUCTIONS
    assert "in-scope and out-of-scope" in _FOUNDATION_INSTRUCTIONS
    assert "scope_clarity" in _AUDIT_INSTRUCTIONS


def test_composer_normalizes_semantic_question_metadata_labels():
    payload = {
        "interview_guide": {
            "probe_selection": "adaptive",
            "questions": [
                {"priority": "high", "sensitivity": "moderate"},
                {"priority": "low", "sensitivity": "low"},
            ],
        }
    }

    normalized = _normalize_question_metadata(payload)

    assert normalized["interview_guide"]["probe_selection"] == "relevance"
    assert normalized["interview_guide"]["questions"] == [
        {"priority": "required", "sensitivity": "elevated", "id": "question_1", "depends_on": []},
        {"priority": "optional", "sensitivity": "standard", "id": "question_2", "depends_on": []},
    ]


async def test_full_composer_runs_foundation_draft_and_audit(monkeypatch):
    foundation = ComposerStudyFoundation.model_validate({
        "decision_to_inform": "Choose what to build.",
        "primary_research_question": "What do participants need?",
        "secondary_research_questions": ["What changes by context?"],
        "participant_population": "Adults with recent relevant experience",
        "inclusion_signal": "One relevant experience in the past six months",
        "knowledge_relationship": "Participants report their own experience.",
        "participant_scope_definition": "The study includes the qualifying feature, not every use of the service that contains it.",
        "in_scope_categories": ["Use of the qualifying feature"],
        "out_of_scope_categories": ["Unrelated use of the surrounding service"],
        "ambiguous_scope_cases": ["Services containing both qualifying and unrelated features"],
        "clarification_principles": ["Answer only the boundary asked about"],
        "evidence_targets": [{
            "id": "episode",
            "name": "Concrete episode",
            "why_it_matters": "Grounds the findings.",
            "evidence_needed": ["Trigger", "Action", "Outcome"],
            "participant_can_know": ["What they experienced"],
            "do_not_infer": ["Prevalence"],
        }],
        "privacy_and_safety_constraints": ["Avoid identifying details"],
        "wording_and_sequence_principles": ["Use one concrete episode"],
        "material_ambiguities": [],
        "source_notes": [],
    })
    candidate = ComposerDraft.model_validate({
        "title": "Candidate study",
        "summary": "Drafted the study.",
        "estimated_minutes": 15,
        "research_spec": {
            "decision_to_inform": "Choose what to build.",
            "research_questions": ["What do participants need?"],
            "targets": [{"id": "episode", "name": "Concrete episode"}],
            "clarification_policy": {
                "participant_definition": "The study includes the qualifying feature rather than every use of its host service.",
                "in_scope": ["Use of the qualifying feature"],
                "out_of_scope": ["Unrelated use of the host service"],
                "ambiguous_cases": ["Unclear mixed-feature use"],
            },
        },
        "system_prompt": "Ask one question at a time.",
        "welcome_message": "Thanks for taking part.",
        "interview_guide": {
            "questions": [{
                "id": "episode",
                "text": "Tell me about the most recent time that happened?",
                "target_ids": ["episode"],
            }],
        },
        "warnings": ["Candidate warning"],
    })
    final = candidate.model_copy(deep=True)
    final.title = "Audited study"
    audit = ComposerAuditReport.model_validate({
        name: {"score": 4, "finding": "Strong with a small revision."}
        for name in (
            "goal_fit",
            "answerability",
            "realistic_evidence_capture",
            "evidence_quality",
            "neutrality",
            "cohesion",
            "privacy_and_safety",
            "scope_clarity",
            "timing_and_efficiency",
        )
    } | {
        "changes_made": ["Tightened the sequence."],
        "remaining_warnings": ["Audit warning"],
    })
    audited = AuditedComposerDraft(audit=audit, draft=final)
    outputs = [
        foundation.model_dump_json(),
        candidate.model_dump_json(),
        audited.model_dump_json(),
    ]
    calls = []

    class FakeResponses:
        async def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(output_text=outputs.pop(0))

    class FakeClient:
        def __init__(self, **kwargs):
            self.responses = FakeResponses()

    async def fake_key(_field):
        return "sk-test"

    async def fake_base():
        return None

    monkeypatch.setattr("app.api.composer.AsyncOpenAI", FakeClient)
    monkeypatch.setattr("app.api.composer._get_key", fake_key)
    monkeypatch.setattr("app.api.composer._openai_api_base", fake_base)

    result = await _compose_with_openai(
        model="gpt-5.6-sol",
        instruction="Create a 15-minute guide",
        context={"study": {"title": "Test"}},
    )

    assert result.title == "Audited study"
    assert result.warnings == ["Candidate warning", "Audit warning"]
    assert len(calls) == 3
    assert all(call["model"] == "gpt-5.6-sol" for call in calls)
    assert all(call["reasoning"] == {"effort": "medium"} for call in calls)
    assert calls[0]["instructions"] == _FOUNDATION_INSTRUCTIONS
    assert calls[0]["tools"] == [{"type": "web_search"}]
    assert calls[1]["instructions"] == _STAGED_DRAFT_INSTRUCTIONS
    assert "METHODOLOGICAL FOUNDATION" in calls[1]["input"]
    assert "tools" not in calls[1]
    assert calls[2]["instructions"] == _AUDIT_INSTRUCTIONS
    assert "CANDIDATE DRAFT TO AUDIT" in calls[2]["input"]
    assert "tools" not in calls[2]


def test_composer_normalizes_nested_audit_output():
    payload = {
        "audit": {"goal_fit": {"score": 9.0, "finding": "Strong"}},
        "draft": {
            "interview_guide": {
                "probe_selection": "adaptive",
                "questions": [{"priority": "high", "sensitivity": "moderate"}],
            }
        },
    }

    normalized = _normalize_question_metadata(payload)

    assert normalized["audit"]["goal_fit"]["score"] == 4.5
    question = normalized["draft"]["interview_guide"]["questions"][0]
    assert question == {
        "priority": "required",
        "sensitivity": "elevated",
        "id": "question_1",
        "depends_on": [],
    }
    assert normalized["draft"]["interview_guide"]["probe_selection"] == "relevance"


def test_composer_replaces_zero_fielding_placeholders_with_warned_defaults():
    payload = {
        "warnings": [],
        "research_spec": {
            "fielding": {
                "response_limit": 0,
                "soft_launch_limit": 0,
                "participant_id_parameter": "",
            }
        },
    }

    normalized = _normalize_question_metadata(payload)

    assert normalized["research_spec"]["fielding"] == {
        "response_limit": 50,
        "soft_launch_limit": 10,
        "participant_id_parameter": "PROLIFIC_PID",
    }
    assert normalized["warnings"] == [
        "Fielding counts or participant-ID settings were not specified; "
        "provisional defaults were applied and must be confirmed before launch."
    ]


async def test_composer_returns_valid_draft_without_mutating(monkeypatch, client: AsyncClient):
    study = (await client.post(
        "/api/studies",
        json={"title": "Original title", "description": "Parents and AI"},
    )).json()
    agent = (await client.post(
        f"/api/studies/{study['id']}/agents",
        json={"name": "Pilot", "status": "draft"},
    )).json()

    expected = ComposerDraft.model_validate({
        "title": "Parents and generative AI",
        "summary": "Built a neutral critical-incident guide.",
        "estimated_minutes": 12,
        "research_spec": {
            "decision_to_inform": "Choose parent safety supports.",
            "research_questions": ["What do parents directly observe?"],
            "population": {
                "participant_role": "Parents of children under 18",
                "subject": "Their child's use of generative AI",
            },
            "targets": [{"id": "observed_episode", "name": "Observed episode"}],
        },
        "system_prompt": "Ask one neutral question at a time.",
        "welcome_message": "Thanks for joining. You may skip any question.",
        "interview_guide": {
            "questions": [{
                "id": "recent_episode",
                "text": "Tell me about the most recent time you noticed your child using generative AI.",
                "purpose": "Elicit a concrete episode.",
                "evidence_goal": ["What was directly observed"],
                "do_not_assume": ["The experience was harmful"],
                "probes": ["What did you notice directly?"],
                "role_variants": {"educator": "Tell me about the most recent time you noticed a student using generative AI."},
            }],
        },
        "warnings": [],
    })

    async def fake_compose(**kwargs):
        assert kwargs["model"] == "gpt-5.6-sol"
        assert kwargs["instruction"] == "Create a neutral guide"
        assert kwargs["context"]["study"]["title"] == "Original title"
        return expected

    monkeypatch.setattr("app.api.composer._compose_with_openai", fake_compose)
    response = await client.post(
        f"/api/studies/{study['id']}/composer",
        json={"instruction": "Create a neutral guide", "agent_id": agent["id"]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["title"] == "Parents and generative AI"
    assert payload["interview_guide"]["questions"][0]["role_variants"]["educator"]
    assert payload["research_spec"]["fielding"]["recruitment_source"] == "prolific"

    # The composer proposes; the researcher-controlled save endpoint mutates.
    saved_study = (await client.get(f"/api/studies/{study['id']}")).json()
    saved_agent = (await client.get(
        f"/api/studies/{study['id']}/agents/{agent['id']}"
    )).json()
    assert saved_study["title"] == "Original title"
    assert saved_agent["interview_guide"] is None


async def test_composer_rejects_agent_from_another_study(client: AsyncClient):
    first = (await client.post("/api/studies", json={"title": "One"})).json()
    second = (await client.post("/api/studies", json={"title": "Two"})).json()
    agent = (await client.post(
        f"/api/studies/{second['id']}/agents", json={"name": "Wrong study"}
    )).json()

    response = await client.post(
        f"/api/studies/{first['id']}/composer",
        json={"instruction": "Revise it", "agent_id": agent["id"]},
    )
    assert response.status_code == 404


async def test_composer_keeps_thread_focus_and_explains_local_question_revision(
    monkeypatch, client: AsyncClient
):
    study = (await client.post(
        "/api/studies",
        json={"title": "Parents and AI", "description": "Parent interview pilot"},
    )).json()
    agent = (await client.post(
        f"/api/studies/{study['id']}/agents",
        json={"name": "Pilot", "status": "draft"},
    )).json()
    before_guide = {
        "questions": [
            {
                "id": "opening",
                "text": "Tell me about the child you have in mind.",
                "phase": "warm_up",
                "visit_scope": "first_time",
            },
            {"id": "context", "text": "What role does technology play at home?"},
        ]
    }
    revised = ComposerDraft.model_validate({
        "title": "Parents and AI",
        "summary": "A qualitative interview about parents and AI.",
        "estimated_minutes": 12,
        "research_spec": {
            "decision_to_inform": "Improve the parent interview.",
            "research_questions": ["What are parents noticing?"],
            "population": {
                "participant_role": "Parents of children under 18",
                "subject": "Their child's use of generative AI",
            },
        },
        "system_prompt": "Ask one neutral question at a time.",
        "welcome_message": "Thanks for taking part.",
        "interview_guide": {
            "questions": [
                {
                    "id": "opening",
                    "text": "Before we get into AI, could you tell me a little about your family and your role as a parent?",
                    "phase": "warm_up",
                    "visit_scope": "first_time",
                },
                {"id": "context", "text": "What role does technology play at home?"},
            ]
        },
        "warnings": [],
    })

    async def fake_compose(**kwargs):
        assert kwargs["instruction"].startswith("I think the first question")
        composer_context = kwargs["context"]["composer"]
        assert composer_context["conversation_history"] == [
            {"role": "composer", "text": "I created the first draft."}
        ]
        assert composer_context["selected_question_index"] == 0
        assert composer_context["selected_question_number"] == 1
        assert composer_context["selected_question_id"] == "opening"
        assert composer_context["manually_edited_questions"][0]["id"] == "opening"
        assert composer_context["edit_history"] == [{
            "id": "edit-1",
            "timestamp": "2026-08-26T12:00:00.000Z",
            "source": "manual",
            "summary": "Reworded the opening.",
            "changes": [{
                "path": "guide.questions[0].text",
                "before": "Tell me about the child you have in mind.",
                "after": "Tell me a little about your family.",
            }],
        }]
        return revised

    monkeypatch.setattr("app.api.composer._compose_with_openai", fake_compose)
    response = await client.post(
        f"/api/studies/{study['id']}/composer",
        json={
            "instruction": (
                "I think the first question sounds weird. "
                "'Tell me about the child you have in mind' is an odd way to start."
            ),
            "agent_id": agent["id"],
            "current_interview_guide": before_guide,
            "conversation_history": [
                {"role": "composer", "text": "I created the first draft."}
            ],
            "selected_question_index": 0,
            "selected_question_id": "opening",
            "edited_question_ids": ["opening"],
            "edit_history": [{
                "id": "edit-1",
                "timestamp": "2026-08-26T12:00:00.000Z",
                "source": "manual",
                "summary": "Reworded the opening.",
                "changes": [{
                    "path": "guide.questions[0].text",
                    "before": "Tell me about the child you have in mind.",
                    "after": "Tell me a little about your family.",
                }],
            }],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"] == (
        "I revised Question 1 to: “Before we get into AI, could you tell me a "
        "little about your family and your role as a parent?” The other question "
        "wording is unchanged."
    )
    assert payload["interview_guide"]["questions"][1]["text"] == before_guide["questions"][1]["text"]


def test_composer_detects_edited_question_text_against_saved_guide():
    changed = _edited_question_context(
        saved={
            "questions": [
                {"id": "q1", "text": "Original opening"},
                {"id": "q2", "text": "Original concern question"},
            ]
        },
        current={
            "questions": [
                {"id": "q1", "text": "Original opening"},
                {"id": "q2", "text": "What kinds of AI use concern you?"},
            ]
        },
        explicit_ids=[],
    )

    assert changed == [{
        "index": 1,
        "number": 2,
        "id": "q2",
        "saved_text": "Original concern question",
        "current_text": "What kinds of AI use concern you?",
    }]


async def test_question_composer_is_scoped_to_one_card(
    monkeypatch, client: AsyncClient
):
    study = (await client.post(
        "/api/studies", json={"title": "Card editor test"}
    )).json()
    original_guide = {
        "questions": [
            {"id": "opening", "text": "How does AI come up in your family?"},
            {
                "id": "concerns",
                "text": "What kinds of AI use concern you?",
                "purpose": "Old purpose",
                "evidence_goal": ["Old evidence"],
            },
        ]
    }
    agent = (await client.post(
        f"/api/studies/{study['id']}/agents",
        json={
            "name": "Pilot",
            "status": "draft",
            "interview_mode": "structured",
            "interview_guide": original_guide,
        },
    )).json()
    revised = QuestionComposerDraft.model_validate({
        "summary": "I aligned the supporting details with the concern question.",
        "question": {
            "id": "concerns",
            "module": "Concerning uses",
            "text": "What kinds of AI use concern you?",
            "purpose": "Understand which uses concern the parent and why.",
            "evidence_goal": ["The concerning use", "Why it matters"],
            "probes": ["What about that use concerns you?"],
            "do_not_assume": ["That the use caused harm"],
        },
        "warnings": [],
    })

    async def fake_question_compose(**kwargs):
        assert kwargs["model"] == "gpt-5.6-sol"
        assert kwargs["instruction"] == "Make the subdetails fit this question"
        context = kwargs["context"]
        assert context["target"]["question_number"] == 2
        assert context["target"]["question"]["id"] == "concerns"
        assert context["read_only_flow_context"]["previous_question"] == original_guide["questions"][0]["text"]
        assert context["conversation_history"] == [
            {"role": "composer", "text": "What would you like to change?"}
        ]
        assert context["edit_history"][0]["source"] == "manual"
        assert context["edit_history"][0]["changes"][0]["path"] == (
            "guide.questions[1].purpose"
        )
        return revised

    monkeypatch.setattr(
        "app.api.composer._compose_question_with_openai",
        fake_question_compose,
    )
    response = await client.post(
        f"/api/studies/{study['id']}/composer/question",
        json={
            "instruction": "Make the subdetails fit this question",
            "agent_id": agent["id"],
            "question_index": 1,
            "current_interview_guide": original_guide,
            "conversation_history": [
                {"role": "composer", "text": "What would you like to change?"}
            ],
            "edit_history": [{
                "id": "edit-2",
                "timestamp": "2026-08-26T12:01:00.000Z",
                "source": "manual",
                "changes": [{
                    "path": "guide.questions[1].purpose",
                    "before": "Old purpose",
                    "after": "Understand the parent's concerns.",
                }],
            }],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["question"]["module"] == "Concerning uses"
    assert "interview_guide" not in payload
    assert "title" not in payload
    saved_agent = (await client.get(
        f"/api/studies/{study['id']}/agents/{agent['id']}"
    )).json()
    saved_questions = saved_agent["interview_guide"]["questions"]
    assert [question["text"] for question in saved_questions] == [
        question["text"] for question in original_guide["questions"]
    ]
    assert saved_questions[1]["purpose"] == "Old purpose"
    assert saved_questions[1]["evidence_goal"] == ["Old evidence"]
