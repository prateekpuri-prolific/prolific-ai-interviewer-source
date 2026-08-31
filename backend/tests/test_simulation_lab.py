from __future__ import annotations

import pytest
from httpx import AsyncClient

from app.simulation_lab.conditioning import build_prompt
from app.simulation_lab.fixtures import CASES
from app.simulation_lab.scoring import score_response


def test_conditioning_prompt_does_not_leak_held_out_response():
    case = CASES[0]
    instructions, input_text = build_prompt(case, "state_aware", 6)
    prompt = f"{instructions}\n{input_text}".lower()
    assert case.held_out.text.lower() not in prompt
    assert case.transcript[-1].text.lower() in prompt
    assert case.transcript[-7].text.lower() not in prompt


def test_scores_are_bounded_and_identical_response_scores_highest():
    reference = CASES[1].held_out.text
    exact = score_response(reference, reference)
    unrelated = score_response("I went to the store yesterday.", reference)
    assert exact.composite == 1.0
    assert exact.composite > unrelated.composite
    assert all(0 <= value <= 1 for value in exact.model_dump().values())
    assert all(0 <= value <= 1 for value in unrelated.model_dump().values())


@pytest.mark.asyncio
async def test_overview_exposes_provenance_and_cases(client: AsyncClient):
    response = await client.get("/api/simulation-lab/overview")
    assert response.status_code == 200
    body = response.json()
    assert body["dataset"]["case_count"] == 3
    assert "not real" in body["dataset"]["provenance"].lower()
    assert len(body["techniques"]) == 5


@pytest.mark.asyncio
async def test_empirical_results_use_participant_level_test_split(client: AsyncClient):
    response = await client.get("/api/simulation-lab/empirical")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "complete"
    assert body["dataset"]["split_counts"]["test"] == 19
    assert body["design"]["test_samples"] == 2
    assert "paired_differences_vs_context" in body
    selected = body["development_selected_arm"]
    assert selected == "survey_card"
    interval = body["paired_differences_vs_context"]["test"][selected]["semantic_similarity"]
    assert interval["low"] <= interval["mean"] <= interval["high"]


@pytest.mark.asyncio
async def test_empirical_studies_expose_annomi_and_mibot(client: AsyncClient):
    response = await client.get("/api/simulation-lab/empirical-studies")
    assert response.status_code == 200
    studies = {item["study_id"]: item for item in response.json()}
    assert studies["annomi-heldout-v1"]["test_cases"] == 29
    assert studies["mibot-heldout-v1"]["test_cases"] == 19


@pytest.mark.asyncio
async def test_annomi_results_use_source_grouped_conversation_split(client: AsyncClient):
    response = await client.get(
        "/api/simulation-lab/empirical", params={"study_id": "annomi-heldout-v1"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["dataset"]["name"] == "AnnoMI"
    assert body["dataset"]["split_counts"] == {"train": 76, "development": 24, "test": 29}
    assert "source-video" in body["design"]["split_policy"]
    assert body["development_selected_arm"] == "behavioral_profile"
    assert body["design"]["oracle_arm"] == "retrospective_world_model"
    assert "target-redacted" in body["design"]["oracle_definition"]
    assert len(body["test_rows"]) == 29 * len(body["design"]["arms"]) * 2
    assert all("talk_type_match" in row["scores"] for row in body["test_rows"])


@pytest.mark.asyncio
async def test_judge_validation_exposes_failed_controls_and_population_metrics(client: AsyncClient):
    response = await client.get("/api/simulation-lab/judge-validation")
    assert response.status_code == 200
    body = response.json()
    assert body["readiness"] == "not_ready_for_optimization"
    assert body["checks_passed"] < body["checks_total"]
    assert body["reliability"]["order_invariance"]["pooled_items"]["n"] > 0
    assert body["candidate_summaries"]["context_only"]["mean_item_pass_rate"] == 1.0
    assert body["pairwise_summaries"]["real_vs_context"]["right_win_rate"] > 0.5
    assert body["population_metrics"]["mibot"]["context_only"]["mean_words"] > body["population_metrics"]["mibot"]["real_observed"]["mean_words"]


@pytest.mark.asyncio
async def test_demo_run_returns_each_case_by_technique(client: AsyncClient):
    response = await client.post(
        "/api/simulation-lab/run",
        json={
            "case_ids": ["sleep-pressure", "social-withdrawal"],
            "technique_ids": ["context_only", "state_aware"],
            "context_messages": 6,
            "mode": "demo",
            "model": "gpt-4.1-mini",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["candidates"]) == 4
    assert len(body["summaries"]) == 2
    assert all("held-out reply" not in row["prompt_preview"].lower() for row in body["candidates"])


@pytest.mark.asyncio
async def test_run_rejects_unknown_technique(client: AsyncClient):
    response = await client.post(
        "/api/simulation-lab/run",
        json={"technique_ids": ["magic_persona"], "mode": "demo"},
    )
    assert response.status_code == 400
