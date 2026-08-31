import { clearAuthToken, formatApiError, getAuthToken } from "./api";

export type Role = "counselor" | "client";

export interface TranscriptTurn { role: Role; text: string }
export interface Persona {
  age: number;
  setting: string;
  presenting_context: string;
  background: string[];
  communication_style: string[];
  current_state: string[];
}
export interface BenchmarkCase {
  id: string;
  title: string;
  topic: string;
  persona: Persona;
  transcript: TranscriptTurn[];
  held_out: TranscriptTurn;
}
export interface Technique {
  id: string;
  name: string;
  short_name: string;
  description: string;
  adds: string[];
  color: string;
}
export interface Overview {
  dataset: { id: string; name: string; provenance: string; population: string; case_count: number; warning: string };
  techniques: Technique[];
  cases: BenchmarkCase[];
  recommended_design: string[];
}
export interface Scores {
  composite: number;
  token_overlap: number;
  phrase_similarity: number;
  length_match: number;
  style_match: number;
  behavior_match: number;
}
export interface Candidate {
  case_id: string;
  technique_id: string;
  response: string;
  scores: Scores;
  prompt_preview: string;
}
export interface RunResult {
  run_id: string;
  mode: "demo" | "live";
  model: string;
  context_messages: number;
  created_at: string;
  candidates: Candidate[];
  summaries: { technique_id: string; mean_composite: number; mean_behavior_match: number; response_count: number }[];
  interpretation_note: string;
}

export interface Estimate { mean: number; low: number; high: number }
export interface EmpiricalRow {
  case_id: string;
  split: "test";
  arm: string;
  sample: number;
  response: string;
  reference: string;
  reference_talk_type: string;
  scores: Record<string, number>;
  visible_prefix: TranscriptTurn[];
  survey_card: Record<string, string>;
  persona: { stable_narrative: string[]; communication_policy: string[]; state_at_turn: string[] };
  grounded_world_model?: Record<string, string[] | string>;
  retrospective_world_model?: Record<string, string[]>;
  case_metadata?: { video_title?: string; video_url?: string; topic?: string; mi_quality?: string };
}
export interface EmpiricalStudySummary {
  study_id: string;
  name: string;
  population_note: string;
  created_at: string;
  status: "complete";
  test_cases: number;
}
export interface EmpiricalResults {
  study_id: string;
  created_at: string;
  status: "complete";
  dataset: {
    name: string; license: string; source: string; population_note: string;
    eligible_participants: number; split_counts: Record<string, number>; evaluation_unit?: string;
  };
  design: {
    target: string; context_turns: number; arms: string[]; development_samples: number;
    test_samples: number; generation_model: string; judge_model: string;
    primary_metric: string; bootstrap_iterations: number; split_policy?: string;
    predictive_arms?: string[]; oracle_arm?: string; oracle_definition?: string;
    literature_followup_arms?: string[]; followup_definition?: string;
  };
  development_selected_arm: string;
  test_descriptive_best_arm: string;
  literature_followup_development_best_arm?: string;
  literature_followup_test_descriptive_best_arm?: string;
  summaries: Record<string, Record<string, Record<string, Estimate>>>;
  paired_differences_vs_context: Record<string, Record<string, Record<string, Estimate>>>;
  test_rows: EmpiricalRow[];
  limitations: string[];
}

export interface ReliabilityEstimate { agreement: number; kappa: number; n: number }
export interface JudgeValidationResults {
  study_id: string;
  created_at: string;
  status: "complete";
  claim_boundary: string;
  readiness: "not_ready_for_optimization" | "ready_for_human_calibration";
  checks_passed: number;
  checks_total: number;
  design: {
    datasets: string[]; cases_per_dataset: number; total_cases: number;
    conditions: { id: string; model: string; prompt: string; order: string }[];
    evaluated_arms: string[]; rubric_items: string[]; rating_scale: string[];
    controls: string[]; pairwise_comparisons: string[];
  };
  control_checks: { name: string; observed: number; threshold: number; passed: boolean }[];
  red_flags: { id: string; severity: "critical" | "warning"; finding: string; implication: string }[];
  reliability: Record<string, {
    pooled_items: ReliabilityEstimate;
    by_item: Record<string, ReliabilityEstimate>;
    pairwise_winners: ReliabilityEstimate;
  }>;
  candidate_summaries: Record<string, Record<string, number | Record<string, number>>>;
  pairwise_summaries: Record<string, {
    left: string; right: string; n: number; left_win_rate: number;
    right_win_rate: number; tie_rate: number; high_confidence_rate: number;
  }>;
  population_metrics: Record<string, Record<string, Record<string, number | boolean | string>>>;
  trajectory_evaluation_available: boolean;
  limitations: string[];
}

async function labRequest<T>(path: string, options?: RequestInit): Promise<T> {
  const token = getAuthToken();
  const response = await fetch(`/api/simulation-lab${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...((options?.headers as Record<string, string>) || {}),
    },
  });
  if (response.status === 401) {
    clearAuthToken();
    window.location.href = "/login";
    throw new Error("Authentication required");
  }
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(formatApiError(body.detail, `API error: ${response.status}`));
  }
  return response.json();
}

export const simulationLab = {
  overview: () => labRequest<Overview>("/overview"),
  empiricalStudies: () => labRequest<EmpiricalStudySummary[]>("/empirical-studies"),
  empirical: (studyId = "mibot-heldout-v1") => labRequest<EmpiricalResults>(`/empirical?study_id=${encodeURIComponent(studyId)}`),
  judgeValidation: () => labRequest<JudgeValidationResults>("/judge-validation"),
  run: (payload: { case_ids: string[]; technique_ids: string[]; context_messages: number; mode: "demo" | "live"; model: string }) =>
    labRequest<RunResult>("/run", { method: "POST", body: JSON.stringify(payload) }),
};
