from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TranscriptTurn(BaseModel):
    role: Literal["counselor", "client"]
    text: str


class Persona(BaseModel):
    age: int
    setting: str
    presenting_context: str
    background: list[str]
    communication_style: list[str]
    current_state: list[str]


class BenchmarkCase(BaseModel):
    id: str
    title: str
    topic: str
    persona: Persona
    transcript: list[TranscriptTurn]
    held_out: TranscriptTurn


class Technique(BaseModel):
    id: str
    name: str
    short_name: str
    description: str
    adds: list[str]
    color: str


class DatasetSummary(BaseModel):
    id: str
    name: str
    provenance: str
    population: str
    case_count: int
    warning: str


class OverviewResponse(BaseModel):
    dataset: DatasetSummary
    techniques: list[Technique]
    cases: list[BenchmarkCase]
    recommended_design: list[str]


class RunRequest(BaseModel):
    case_ids: list[str] = Field(default_factory=list)
    technique_ids: list[str] = Field(default_factory=list)
    context_messages: int = Field(default=6, ge=2, le=20)
    mode: Literal["demo", "live"] = "demo"
    model: str = Field(default="gpt-4.1-mini", min_length=1, max_length=100)


class SimilarityScores(BaseModel):
    composite: float
    token_overlap: float
    phrase_similarity: float
    length_match: float
    style_match: float
    behavior_match: float


class CandidateResult(BaseModel):
    case_id: str
    technique_id: str
    response: str
    scores: SimilarityScores
    prompt_preview: str


class TechniqueSummary(BaseModel):
    technique_id: str
    mean_composite: float
    mean_behavior_match: float
    response_count: int


class RunResponse(BaseModel):
    run_id: str
    mode: Literal["demo", "live"]
    model: str
    context_messages: int
    created_at: str
    candidates: list[CandidateResult]
    summaries: list[TechniqueSummary]
    interpretation_note: str
