from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from openai import AsyncOpenAI

from app.api.text_chat import _get_key, _openai_api_base
from app.simulation_lab.conditioning import build_prompt, technique_by_id
from app.simulation_lab.fixtures import CASES, DATASET, DEMO_RESPONSES, TECHNIQUES
from app.simulation_lab.models import (
    CandidateResult, OverviewResponse, RunRequest, RunResponse, TechniqueSummary,
)
from app.simulation_lab.scoring import score_response

router = APIRouter(prefix="/simulation-lab", tags=["simulation-lab"])
_EMPIRICAL_RESULTS = Path(__file__).resolve().parents[1] / "simulation_lab" / "empirical_results.json"
_ANNOMI_RESULTS = Path(__file__).resolve().parents[1] / "simulation_lab" / "annomi_results.json"
_JUDGE_VALIDATION_RESULTS = Path(__file__).resolve().parents[1] / "simulation_lab" / "judge_validation_results.json"
_EMPIRICAL_STUDIES = {
    "mibot-heldout-v1": _EMPIRICAL_RESULTS,
    "annomi-heldout-v1": _ANNOMI_RESULTS,
}


@router.get("/overview", response_model=OverviewResponse)
async def overview() -> OverviewResponse:
    return OverviewResponse(
        dataset=DATASET,
        techniques=TECHNIQUES,
        cases=CASES,
        recommended_design=[
            "Split by participant, never by turn, before prompt development.",
            "Tune on a development set and touch the held-out test set once.",
            "Report each metric separately; do not treat lexical similarity as realism.",
            "Add blinded human ratings after the pipeline and failure modes stabilize.",
        ],
    )


@router.get("/empirical-studies")
async def empirical_studies():
    studies = []
    for study_id, path in _EMPIRICAL_STUDIES.items():
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        studies.append({
            "study_id": study_id,
            "name": payload["dataset"]["name"],
            "population_note": payload["dataset"]["population_note"],
            "created_at": payload["created_at"],
            "status": payload["status"],
            "test_cases": payload["dataset"]["split_counts"]["test"],
        })
    return sorted(studies, key=lambda item: (item["study_id"] != "annomi-heldout-v1", item["name"]))


@router.get("/empirical")
async def empirical_results(study_id: str = "mibot-heldout-v1"):
    path = _EMPIRICAL_STUDIES.get(study_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Unknown empirical study.")
    if not path.exists():
        raise HTTPException(status_code=404, detail="No completed empirical study is available yet.")
    return json.loads(path.read_text(encoding="utf-8"))


@router.get("/judge-validation")
async def judge_validation_results():
    if not _JUDGE_VALIDATION_RESULTS.exists():
        raise HTTPException(status_code=404, detail="No completed judge-validation study is available yet.")
    return json.loads(_JUDGE_VALIDATION_RESULTS.read_text(encoding="utf-8"))


async def _live_response(case, technique_id: str, context_messages: int, model: str, client: AsyncOpenAI) -> str:
    instructions, input_text = build_prompt(case, technique_id, context_messages)
    response = await client.responses.create(
        model=model,
        instructions=instructions,
        input=input_text,
        max_output_tokens=180,
        temperature=0.7,
        store=False,
    )
    if not response.output_text:
        raise ValueError("The model returned an empty continuation.")
    return response.output_text.strip()


@router.post("/run", response_model=RunResponse)
async def run(payload: RunRequest) -> RunResponse:
    selected_cases = [case for case in CASES if not payload.case_ids or case.id in payload.case_ids]
    technique_ids = payload.technique_ids or [item.id for item in TECHNIQUES]
    unknown_cases = set(payload.case_ids) - {case.id for case in CASES}
    unknown_techniques = set(technique_ids) - {item.id for item in TECHNIQUES}
    if unknown_cases or unknown_techniques:
        raise HTTPException(status_code=400, detail="Unknown case or conditioning technique.")
    if not selected_cases:
        raise HTTPException(status_code=400, detail="Select at least one case.")

    client = None
    if payload.mode == "live":
        key = await _get_key("openai_api_key")
        if not key:
            raise HTTPException(status_code=503, detail="Add an OpenAI API key in Settings before a live run.")
        client = AsyncOpenAI(api_key=key, base_url=await _openai_api_base())

    async def make_candidate(case, technique_id: str) -> CandidateResult:
        instructions, user_input = build_prompt(case, technique_id, payload.context_messages)
        if payload.mode == "demo":
            response = DEMO_RESPONSES[case.id][technique_id]
        else:
            assert client is not None
            response = await _live_response(case, technique_id, payload.context_messages, payload.model, client)
        preview = f"{instructions}\n\n{user_input}"
        return CandidateResult(
            case_id=case.id,
            technique_id=technique_id,
            response=response,
            scores=score_response(response, case.held_out.text),
            prompt_preview=preview,
        )

    try:
        candidates = await asyncio.gather(*[
            make_candidate(case, technique_id)
            for case in selected_cases for technique_id in technique_ids
        ])
    except Exception as exc:
        if isinstance(exc, HTTPException):
            raise
        raise HTTPException(status_code=502, detail=f"Generation failed: {exc}") from exc

    summaries = []
    for technique_id in technique_ids:
        rows = [row for row in candidates if row.technique_id == technique_id]
        summaries.append(TechniqueSummary(
            technique_id=technique_id,
            mean_composite=round(sum(row.scores.composite for row in rows) / len(rows), 3),
            mean_behavior_match=round(sum(row.scores.behavior_match for row in rows) / len(rows), 3),
            response_count=len(rows),
        ))
    return RunResponse(
        run_id=str(uuid4()), mode=payload.mode, model=payload.model,
        context_messages=payload.context_messages,
        created_at=datetime.now(timezone.utc).isoformat(),
        candidates=candidates, summaries=summaries,
        interpretation_note="Scores are diagnostic proxies, not evidence that a response is clinically or developmentally realistic. Compare patterns across a participant-held-out test set and confirm them with blinded human review.",
    )
