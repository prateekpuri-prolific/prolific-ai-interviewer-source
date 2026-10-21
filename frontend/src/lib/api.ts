/**
 * OASIS — API client utility.
 *
 * Thin wrapper around fetch for communicating with the FastAPI backend.
 */

import { formatApiError } from "./apiErrors";

export { formatApiError, localDateEndIso, localDateStartIso, isValidWidgetHexColor, parseWidgetHexColor } from "./apiErrors";

const BASE = "/api";

// ── Auth token management ─────────────────────────────────────
const TOKEN_KEY = "oasis_auth_token";

export function getAuthToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

export function setAuthToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token);
}

export function clearAuthToken(): void {
  localStorage.removeItem(TOKEN_KEY);
}

async function request<T>(
  path: string,
  options?: RequestInit
): Promise<T> {
  const token = getAuthToken();
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...((options?.headers as Record<string, string>) || {}),
  };

  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const res = await fetch(`${BASE}${path}`, {
    ...options,
    headers,
  });

  if (res.status === 204) return undefined as T;

  if (res.status === 401) {
    // Token expired or invalid — clear it and redirect to login
    clearAuthToken();
    if (window.location.pathname !== "/login") {
      window.location.href = "/login";
    }
    throw new Error("Authentication required");
  }

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(formatApiError(body.detail, `API error: ${res.status}`));
  }

  return res.json();
}

// ── Auth API ──────────────────────────────────────────────────

export interface AuthStatusResponse {
  auth_enabled: boolean;
  authenticated: boolean;
  username: string | null;
}

export interface LoginResponse {
  token: string;
  username: string;
  expires_in: number;
}

export const auth = {
  login: (username: string, password: string) =>
    request<LoginResponse>("/auth/login", {
      method: "POST",
      body: JSON.stringify({ username, password }),
    }),

  status: () => request<AuthStatusResponse>("/auth/status"),
};

// ── Types ────────────────────────────────────────────────────

export interface ResearchPopulation {
  participant_role: string;
  subject: string;
  knowledge_relationship: string;
  inclusion_notes: string;
  knowledge_limits: string[];
  terminology_notes: string;
  accessibility_notes: string;
  sensitivity_notes: string;
}

export interface ResearchTarget {
  id: string;
  name: string;
  description: string;
  evidence_needed: string[];
  do_not_infer: string[];
  required: boolean;
}

export interface ScopeClarificationPolicy {
  participant_definition: string;
  in_scope: string[];
  out_of_scope: string[];
  ambiguous_cases: string[];
  ambiguity_handling: string;
}

export interface ParticipantInformation {
  purpose_summary: string;
  participation_summary: string;
  estimated_minutes: number | null;
  ai_disclosure: string;
  data_handling_summary: string;
  data_controller: string;
  data_recipients_summary: string;
  risks_summary: string;
  benefits_summary: string;
  compensation_summary: string;
  privacy_reminder: string;
  rights: string[];
  consent_required: boolean;
  consent_label: string;
  researcher_identity: string;
  researcher_contact: string;
  complaints_contact: string;
  ethics_review_reference: string;
  data_retention_period: string;
  withdrawal_instructions: string;
  privacy_notice_url: string;
}

export interface ResearchFielding {
  recruitment_source: "prolific" | "direct_link" | "mixed";
  response_limit: number;
  soft_launch_limit: number | null;
  participant_id_parameter: string;
  require_profile: boolean;
  completion_url: string;
  no_consent_url: string;
  inactivity_url: string;
}

export interface ResearchSpec {
  version: string;
  stage: "pilot" | "field";
  method: "exploration" | "journey" | "needs" | "concept_test" | "evaluation" | "standardized_comparison" | "other";
  decision_to_inform: string;
  research_questions: string[];
  population: ResearchPopulation;
  targets: ResearchTarget[];
  clarification_policy: ScopeClarificationPolicy;
  interviewer_stance: {
    organization_label: string;
    disclosure: string;
    boundaries: string[];
  };
  participant_information: ParticipantInformation;
  fielding: ResearchFielding;
  analysis_notes: string;
}

export interface Study {
  id: string;
  title: string;
  description: string | null;
  status: "draft" | "active" | "paused" | "completed";
  research_spec: ResearchSpec | null;
  created_at: string;
  updated_at: string;
}

export interface StudyListItem {
  id: string;
  title: string;
  status: "draft" | "active" | "paused" | "completed";
  created_at: string;
}

export interface InterviewQuestionConfig {
  text: string;
  probes: string[];
  max_follow_ups: number;
  transition: string | null;
  id?: string | null;
  module?: string | null;
  purpose?: string | null;
  target_ids?: string[];
  evidence_goal?: string[];
  exploration_focus?: string[];
  do_not_assume?: string[];
  answer_requirement?: "any" | "problem_episode";
  requirement_miss_action?: "continue" | "end_interview";
  priority?: "required" | "optional";
  sensitivity?: "standard" | "elevated";
  phase?: "warm_up" | "core" | "closing";
  visit_scope?: "all" | "first_time" | "returning";
  skip_remaining_module_if_not_applicable?: boolean;
  always_ask?: boolean;
  depends_on?: string[];
  role_variants?: Record<string, string | { text?: string; probes?: string[] }>;
  returning_role_variants?: Record<string, string | { text?: string; probes?: string[] }>;
}

export interface InterviewGuideConfig {
  conversation_policy?: "contextual_v1" | null;
  conversation_follow_up_budget?: number;
  questions: InterviewQuestionConfig[];
  orchestration_profile?: "fixed" | "repair_only" | "semi_structured" | "exploratory" | null;
  closing_message: string | null;
  allow_follow_ups?: boolean;
  skip_answered_questions?: boolean | null;
  probe_selection?: "ordered" | "relevance";
  allow_emergent_follow_ups?: boolean;
  max_quality_repairs?: number;
}

export interface ParticipantExperienceFeedback {
  token: string;
  naturalness_rating: number;
  repeated_questions: boolean;
  technical_issues: boolean;
  comments?: string | null;
}

export interface Agent {
  id: string;
  study_id: string;
  name: string;
  modality: "voice" | "text";
  avatar: string | null;
  status: "draft" | "active" | "paused";
  system_prompt: string;
  welcome_message: string | null;
  pipeline_type: "modular" | "voice_to_voice";
  llm_model: string;
  stt_provider: string;
  stt_model: string | null;
  tts_provider: string;
  tts_model: string | null;
  tts_voice: string | null;
  turn_detection: "local" | "remote";
  language: string;
  max_duration_seconds: number | null;
  participant_id_mode: "random" | "predefined" | "input" | "external";
  widget_key: string;
  widget_title: string | null;
  widget_description: string | null;
  widget_primary_color: string | null;
  widget_listening_message: string | null;
  widget_show_progress: boolean;
  interview_mode: "free_form" | "structured";
  interview_guide: InterviewGuideConfig | null;
  silence_timeout_seconds: number | null;
  silence_prompt: string | null;
  twilio_phone_number: string | null;
  store_audio: boolean;
  track_engagement: boolean;
  engagement_config: EngagementConfig | null;
  adaptive_enabled: boolean;
  adaptive_policy: AdaptivePolicy | null;
  created_at: string;
  updated_at: string;
}

export interface EngagementWeights {
  length: number;
  latency: number;
  rate: number;
  fillers: number;
  energy: number;
}

export interface EngagementConfig {
  window_size: number;
  low_threshold: number;
  high_threshold: number;
  long_latency_ms: number;
  short_answer_words: number;
  weights: EngagementWeights;
}

export const DEFAULT_ENGAGEMENT_CONFIG: EngagementConfig = {
  window_size: 3,
  low_threshold: 0.34,
  high_threshold: 0.67,
  long_latency_ms: 4000,
  short_answer_words: 3,
  weights: { length: 0.35, latency: 0.25, rate: 0.15, fillers: 0.15, energy: 0.1 },
};

// Text interviews: latency is reading + typing time, so thresholds are much
// wider, and only length / latency / lexical hedging contribute to the score.
export const DEFAULT_ENGAGEMENT_CONFIG_TEXT: EngagementConfig = {
  window_size: 3,
  low_threshold: 0.34,
  high_threshold: 0.67,
  long_latency_ms: 45000,
  short_answer_words: 3,
  weights: { length: 0.5, latency: 0.2, rate: 0, fillers: 0.3, energy: 0 },
};

// ── Adaptive behavior ─────────────────────────────────────────

export type AdaptiveTrigger =
  | "sustained_disengagement"
  | "positive_engagement_streak"
  | "recovery_after_dip"
  | "long_latency"
  | "very_short_answer"
  | "high_filler";

export type AdaptiveActionId =
  | "offer_break"
  | "soften_next_probe"
  | "encourage_elaboration"
  | "acknowledge_effort"
  | "privacy_check"
  | "match_style"
  | "slow_down"
  | "reset_pace";

export interface AdaptiveRule {
  on: AdaptiveTrigger;
  action: AdaptiveActionId;
  custom_instruction: string | null;
  cooldown_seconds: number;
  params: Record<string, number>;
}

export interface AdaptivePolicy {
  mode: "shadow" | "live";
  rules: AdaptiveRule[];
}

export const ADAPTIVE_TRIGGER_LABELS: Record<AdaptiveTrigger, string> = {
  sustained_disengagement: "Sustained disengagement",
  positive_engagement_streak: "Positive engagement streak",
  recovery_after_dip: "Recovery after a dip",
  long_latency: "Long response delay (turn)",
  very_short_answer: "Very short answer (turn)",
  high_filler: "High filler / hedging (turn)",
};

// Labels and default instruction texts mirror the backend's ACTION_CATALOG
// (backend/app/engagement/adaptive.py). The instruction shown in the form is
// exactly what gets injected — researchers can edit it per rule.
export const ADAPTIVE_ACTION_META: Record<
  AdaptiveActionId,
  { label: string; type: "prompt" | "tts_speed"; defaultInstruction?: string }
> = {
  offer_break: {
    label: "Offer a break",
    type: "prompt",
    defaultInstruction:
      "The participant has shown signs of fatigue or disengagement over the " +
      "last few turns. Gently offer a short break or to move to a lighter " +
      "topic. Keep it brief and warm. Do not mention that this was detected " +
      "automatically.",
  },
  soften_next_probe: {
    label: "Soften the next question",
    type: "prompt",
    defaultInstruction:
      "Make your next question gentler and less probing. Lead with warmth and " +
      "give the participant room to answer at their own pace.",
  },
  encourage_elaboration: {
    label: "Encourage elaboration",
    type: "prompt",
    defaultInstruction:
      "The participant's recent answers have been brief. Warmly invite them to " +
      "say more with a single open follow-up question.",
  },
  acknowledge_effort: {
    label: "Acknowledge engagement",
    type: "prompt",
    defaultInstruction:
      "Briefly acknowledge the participant's effort and engagement before " +
      "continuing with the interview.",
  },
  privacy_check: {
    label: "Check in on comfort",
    type: "prompt",
    defaultInstruction:
      "Check in about the participant's comfort and privacy in one short, warm " +
      "sentence before continuing.",
  },
  match_style: {
    label: "Mirror the participant's style",
    type: "prompt",
    defaultInstruction:
      "From now on, subtly mirror the participant's communication style: " +
      "match their level of formality, their sentence length, and their " +
      "energy. If they are brief and casual, be brief and casual; if they " +
      "are detailed and reflective, give them room and depth. Keep the " +
      "mirroring subtle — never imitate their exact words back at them, and " +
      "never mention that you are adapting.",
  },
  slow_down: { label: "Slow speaking pace", type: "tts_speed" },
  reset_pace: { label: "Reset speaking pace", type: "tts_speed" },
};

// Starter rules seeded into the form when adaptive behavior is first enabled.
// They are ordinary, fully editable rules — adjust or remove them like any
// other (the instruction text is pre-filled from the catalog defaults).
export function defaultAdaptiveRules(): AdaptiveRule[] {
  const rule = (
    on: AdaptiveTrigger,
    action: AdaptiveActionId,
    cooldown_seconds: number
  ): AdaptiveRule => ({
    on,
    action,
    custom_instruction: ADAPTIVE_ACTION_META[action].defaultInstruction ?? null,
    cooldown_seconds,
    params: {},
  });
  return [
    rule("sustained_disengagement", "offer_break", 180),
    rule("very_short_answer", "encourage_elaboration", 90),
    rule("high_filler", "soften_next_probe", 120),
  ];
}

export const DEFAULT_ADAPTIVE_POLICY: AdaptivePolicy = {
  mode: "shadow",
  rules: [],
};

export interface AgentListItem {
  id: string;
  study_id: string;
  name: string;
  modality: "voice" | "text";
  avatar: string | null;
  status: "draft" | "active" | "paused";
  pipeline_type: "modular" | "voice_to_voice";
  llm_model: string;
  stt_provider: string;
  stt_model: string | null;
  tts_provider: string;
  tts_model: string | null;
  tts_voice: string | null;
  language: string;
  widget_key: string;
  participant_id_mode: "random" | "predefined" | "input" | "external";
  interview_mode: "free_form" | "structured";
  created_at: string;
}

export interface WidgetConfig {
  widget_key: string;
  modality: "voice" | "text";
  avatar: string;
  widget_title: string | null;
  widget_description: string | null;
  widget_primary_color: string;
  widget_listening_message: string | null;
  widget_show_progress: boolean;
  participant_id_mode: "random" | "predefined" | "input" | "external";
  interview_mode: "free_form" | "structured";
  question_count: number;
  audio_available: boolean;
  audio_default_on: boolean;
  voice_response_available: boolean;
  response_mode_default: "voice" | "text";
  welcome_message: string | null;
  language: string;
  participant_information: ParticipantInformation | null;
  fielding: ResearchFielding;
  research_protocol_version: string | null;
  research_stage: "pilot" | "field" | null;
  source_code_url?: string | null;
}

export interface ParticipantIdentifier {
  id: string;
  agent_id: string;
  identifier: string;
  label: string | null;
  profile: Record<string, unknown> | null;
  used: boolean;
  session_id: string | null;
  created_at: string;
  updated_at: string;
}

// ── Session Types ─────────────────────────────────────────────

export interface SessionItem {
  id: string;
  agent_id: string;
  status: "active" | "completed" | "timed_out" | "error";
  duration_seconds: number | null;
  total_tokens: number | null;
  participant_id: string | null;
  recruitment_source?: string | null;
  external_study_id?: string | null;
  external_session_id?: string | null;
  research_protocol_version?: string | null;
  research_stage?: string | null;
  completion_metadata?: Record<string, unknown> | null;
  post_interview_feedback?: Record<string, unknown> | null;
  post_interview_feedback_submitted_at?: string | null;
  participant_consent_confirmed?: boolean;
  participant_consent_at?: string | null;
  ended_at: string | null;
  audio_recording_enabled: boolean;
  audio_storage_uri: string | null;
  audio_recording_status: string;
  adaptive_active?: boolean;
  created_at: string;
  updated_at: string;
}

export interface AudioTurn {
  sequence: number;
  role: string;
  filename: string;
  duration_ms?: number | null;
  content_preview?: string | null;
}

export interface SessionAudioManifest {
  session_id: string;
  storage_uri: string | null;
  recording_status: string;
  turns: AudioTurn[];
}

export interface EngagementTurn {
  transcript_sequence: number;
  response_latency_ms: number | null;
  voiced_ms: number | null;
  word_count: number | null;
  char_count: number | null;
  speech_rate_wpm: number | null;
  filler_count: number | null;
  rms_energy: number | null;
  score: number | null;
  label: string | null;
  flags: string[];
}

export interface EngagementEvent {
  transcript_sequence: number | null;
  event_type: string;
  score_at_event: number | null;
}

export interface AdaptiveActionRecord {
  transcript_sequence: number | null;
  trigger: string;
  action: string;
  mode: string;
  detail: Record<string, unknown> | null;
  created_at: string;
}

export interface EngagementSummary {
  session_id: string;
  turn_count: number;
  average_score: number | null;
  label: string | null;
  average_latency_ms: number | null;
  average_words: number | null;
  low_engagement_turns: number;
  turns: EngagementTurn[];
  events: EngagementEvent[];
  adaptive_active: boolean;
  adaptive_actions: AdaptiveActionRecord[];
}

export interface TranscriptEntry {
  id: string;
  session_id: string;
  role: "user" | "agent" | "system";
  content: string;
  sequence: number;
  prompt_tokens: number | null;
  completion_tokens: number | null;
  spoken_at: string;
  created_at: string;
}

export interface SessionDetail extends SessionItem {
  protocol_snapshot?: Record<string, unknown> | null;
  entries: TranscriptEntry[];
}

// ── Analytics Types ──────────────────────────────────────────

export interface AgentStats {
  agent_id: string;
  agent_name: string;
  total_sessions: number;
  completed_sessions: number;
  error_sessions: number;
  timed_out_sessions: number;
  active_sessions: number;
  avg_duration_seconds: number | null;
  total_utterances: number;
  completion_rate: number;
}

export interface StudyAnalytics {
  study_id: string;
  total_sessions: number;
  completed_sessions: number;
  error_sessions: number;
  timed_out_sessions: number;
  active_sessions: number;
  avg_duration_seconds: number | null;
  total_utterances: number;
  completion_rate: number;
  agents: AgentStats[];
}

export interface SessionStats {
  total_sessions: number;
  completed_sessions: number;
  error_sessions: number;
  timed_out_sessions: number;
  active_sessions: number;
  avg_duration_seconds: number | null;
  total_utterances: number;
  completion_rate: number;
}

// ── Studies ──────────────────────────────────────────────────

export const studies = {
  list: () => request<StudyListItem[]>("/studies"),

  get: (id: string) => request<Study>(`/studies/${id}`),

  create: (data: { title: string; description?: string }) =>
    request<Study>("/studies", {
      method: "POST",
      body: JSON.stringify(data),
    }),

  update: (id: string, data: Partial<Pick<Study, "title" | "description" | "status" | "research_spec">>) =>
    request<Study>(`/studies/${id}`, {
      method: "PATCH",
      body: JSON.stringify(data),
    }),

  delete: (id: string) =>
    request<void>(`/studies/${id}`, { method: "DELETE" }),

  analytics: (id: string) =>
    request<StudyAnalytics>(`/studies/${id}/analytics`),
};

export interface ComposerDraft {
  title: string;
  summary: string;
  estimated_minutes: number;
  research_spec: ResearchSpec;
  system_prompt: string;
  welcome_message: string;
  interview_guide: InterviewGuideConfig;
  warnings: string[];
}

export interface QuestionComposerDraft {
  summary: string;
  question: InterviewQuestionConfig;
  warnings: string[];
}

export interface ComposerEditChange {
  path: string;
  before: unknown;
  after: unknown;
}

export interface ComposerEditEvent {
  id: string;
  timestamp: string;
  source: "manual" | "study_composer" | "question_composer" | "undo" | "redo";
  summary?: string;
  changes: ComposerEditChange[];
}

export const composer = {
  compose: (
    studyId: string,
    data: {
      instruction: string;
      agent_id?: string;
      current_research_spec?: ResearchSpec | null;
      current_interview_guide?: InterviewGuideConfig | null;
      current_system_prompt?: string;
      current_welcome_message?: string | null;
      conversation_history?: Array<{ role: "researcher" | "composer"; text: string }>;
      selected_question_index?: number;
      selected_question_id?: string;
      edited_question_ids?: string[];
      edit_history?: ComposerEditEvent[];
    },
  ) =>
    request<ComposerDraft>(`/studies/${studyId}/composer`, {
      method: "POST",
      body: JSON.stringify(data),
    }),
  composeQuestion: (
    studyId: string,
    data: {
      instruction: string;
      agent_id: string;
      question_index: number;
      current_research_spec?: ResearchSpec | null;
      current_interview_guide: InterviewGuideConfig;
      conversation_history?: Array<{ role: "researcher" | "composer"; text: string }>;
      edit_history?: ComposerEditEvent[];
    },
  ) =>
    request<QuestionComposerDraft>(`/studies/${studyId}/composer/question`, {
      method: "POST",
      body: JSON.stringify(data),
    }),
};

// ── Agents ───────────────────────────────────────────────────

export type AgentCreatePayload = Omit<
  Agent,
  "id" | "study_id" | "widget_key" | "created_at" | "updated_at"
>;

export type AgentUpdatePayload = Partial<AgentCreatePayload>;

export const agents = {
  list: (studyId: string) =>
    request<AgentListItem[]>(`/studies/${studyId}/agents`),

  get: (studyId: string, agentId: string) =>
    request<Agent>(`/studies/${studyId}/agents/${agentId}`),

  create: (studyId: string, data: Partial<AgentCreatePayload> & { name: string }) =>
    request<Agent>(`/studies/${studyId}/agents`, {
      method: "POST",
      body: JSON.stringify(data),
    }),

  update: (studyId: string, agentId: string, data: AgentUpdatePayload) =>
    request<Agent>(`/studies/${studyId}/agents/${agentId}`, {
      method: "PATCH",
      body: JSON.stringify(data),
    }),

  delete: (studyId: string, agentId: string) =>
    request<void>(`/studies/${studyId}/agents/${agentId}`, {
      method: "DELETE",
    }),

  /** Instantiate an Agent in this study from a built-in template. */
  createFromTemplate: (
    studyId: string,
    templateId: string,
    body?: { name?: string },
  ) =>
    request<Agent>(`/studies/${studyId}/agents/from-template/${templateId}`, {
      method: "POST",
      body: JSON.stringify(body ?? {}),
    }),
};

// ── Templates (public listing, auth-required to instantiate) ────

export interface AgentTemplate {
  id: string;
  name: string;
  description: string;
  tags: string[];
  modality: "voice" | "text";
  pipeline_type: "modular" | "voice_to_voice";
  llm_model: string;
  interview_mode: "free_form" | "structured";
}

export const templates = {
  list: () => request<AgentTemplate[]>("/templates"),
};

// ── Widget Config (public) ───────────────────────────────────

export const widget = {
  config: (widgetKey: string) =>
    request<WidgetConfig>(`/widget/${widgetKey}`),
  submitFeedback: (
    widgetKey: string,
    sessionId: string,
    feedback: ParticipantExperienceFeedback,
  ) =>
    request<{ submitted: boolean; already_submitted: boolean }>(
      `/widget/${widgetKey}/sessions/${sessionId}/feedback`,
      { method: "POST", body: JSON.stringify(feedback) },
    ),
};

// ── Participant Identifiers ──────────────────────────────────

export const participants = {
  list: (studyId: string, agentId: string) =>
    request<ParticipantIdentifier[]>(
      `/studies/${studyId}/agents/${agentId}/participants`
    ),

  create: (
    studyId: string,
    agentId: string,
    data: {
      identifier: string;
      label?: string | null;
      profile?: Record<string, unknown> | null;
    },
  ) =>
    request<ParticipantIdentifier>(
      `/studies/${studyId}/agents/${agentId}/participants`,
      { method: "POST", body: JSON.stringify(data) }
    ),

  bulkCreate: (studyId: string, agentId: string, identifiers: string[]) =>
    request<ParticipantIdentifier[]>(
      `/studies/${studyId}/agents/${agentId}/participants/bulk`,
      { method: "POST", body: JSON.stringify({ identifiers }) }
    ),

  bulkCreateProfiles: (
    studyId: string,
    agentId: string,
    rows: Array<{
      identifier: string;
      label?: string | null;
      profile?: Record<string, unknown> | null;
    }>,
  ) =>
    request<ParticipantIdentifier[]>(
      `/studies/${studyId}/agents/${agentId}/participants/bulk`,
      { method: "POST", body: JSON.stringify({ participants: rows }) }
    ),

  delete: (studyId: string, agentId: string, participantId: string) =>
    request<void>(
      `/studies/${studyId}/agents/${agentId}/participants/${participantId}`,
      { method: "DELETE" }
    ),

  /** Re-mark a predefined participant identifier as available. */
  release: (studyId: string, agentId: string, participantId: string) =>
    request<ParticipantIdentifier>(
      `/studies/${studyId}/agents/${agentId}/participants/${participantId}/release`,
      { method: "POST" }
    ),
};

// ── Knowledge Base ──────────────────────────────────────────────

export interface KnowledgeDocument {
  id: string;
  study_id: string;
  title: string;
  source_type: string;
  content_length: number;
  chunk_count: number;
  created_at: string;
}

export interface KnowledgeSearchResult {
  content: string;
  title: string;
  similarity: number;
}

export const knowledge = {
  list: (studyId: string) =>
    request<KnowledgeDocument[]>(`/studies/${studyId}/knowledge`),

  uploadText: (studyId: string, data: { title: string; content: string }) =>
    request<KnowledgeDocument>(`/studies/${studyId}/knowledge/text`, {
      method: "POST",
      body: JSON.stringify(data),
    }),

  uploadFile: (studyId: string, file: File, title?: string) => {
    const formData = new FormData();
    formData.append("file", file);
    if (title) formData.append("title", title);
    const headers: Record<string, string> = {};
    const token = getAuthToken();
    if (token) headers["Authorization"] = `Bearer ${token}`;
    return fetch(`${BASE}/studies/${studyId}/knowledge/file`, {
      method: "POST",
      body: formData,
      headers,
    }).then(async (res) => {
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(formatApiError(body.detail, `API error: ${res.status}`));
      }
      return res.json() as Promise<KnowledgeDocument>;
    });
  },

  delete: (studyId: string, documentId: string) =>
    request<void>(`/studies/${studyId}/knowledge/${documentId}`, {
      method: "DELETE",
    }),

  search: (studyId: string, query: string, topK?: number) =>
    request<KnowledgeSearchResult[]>(`/studies/${studyId}/knowledge/search`, {
      method: "POST",
      body: JSON.stringify({ query, top_k: topK || 5 }),
    }),
};

// ── Sessions ────────────────────────────────────────────────────

export interface SessionListParams {
  status?: string;
  date_from?: string;
  date_to?: string;
  sort_by?: string;
  sort_order?: string;
  session_ids?: string;
}

export const sessions = {
  list: (studyId: string, agentId: string, params?: SessionListParams) => {
    const searchParams = new URLSearchParams();
    if (params?.status) searchParams.set("status", params.status);
    if (params?.date_from) searchParams.set("date_from", params.date_from);
    if (params?.date_to) searchParams.set("date_to", params.date_to);
    if (params?.sort_by) searchParams.set("sort_by", params.sort_by);
    if (params?.sort_order) searchParams.set("sort_order", params.sort_order);
    const qs = searchParams.toString();
    return request<SessionItem[]>(
      `/studies/${studyId}/agents/${agentId}/sessions${qs ? `?${qs}` : ""}`
    );
  },

  get: (studyId: string, agentId: string, sessionId: string) =>
    request<SessionDetail>(
      `/studies/${studyId}/agents/${agentId}/sessions/${sessionId}`
    ),

  stats: (studyId: string, agentId: string) =>
    request<SessionStats>(
      `/studies/${studyId}/agents/${agentId}/sessions/stats/summary`
    ),

  terminate: (studyId: string, agentId: string, sessionId: string) =>
    request<void>(
      `/studies/${studyId}/agents/${agentId}/sessions/${sessionId}/terminate`,
      { method: "POST" }
    ),

  getAudioManifest: (studyId: string, agentId: string, sessionId: string) =>
    request<SessionAudioManifest>(
      `/studies/${studyId}/agents/${agentId}/sessions/${sessionId}/audio`
    ),

  getEngagement: (studyId: string, agentId: string, sessionId: string) =>
    request<EngagementSummary>(
      `/studies/${studyId}/agents/${agentId}/sessions/${sessionId}/engagement`
    ),

  audioTurnUrl: (
    studyId: string,
    agentId: string,
    sessionId: string,
    filename: string
  ) =>
    `${BASE}/studies/${studyId}/agents/${agentId}/sessions/${sessionId}/audio/${filename}`,

  exportCsvUrl: (studyId: string, agentId: string, params?: SessionListParams) => {
    const searchParams = new URLSearchParams();
    if (params?.status) searchParams.set("status", params.status);
    if (params?.date_from) searchParams.set("date_from", params.date_from);
    if (params?.date_to) searchParams.set("date_to", params.date_to);
    if (params?.session_ids) searchParams.set("session_ids", params.session_ids);
    const qs = searchParams.toString();
    return `${BASE}/studies/${studyId}/agents/${agentId}/sessions/export/csv${qs ? `?${qs}` : ""}`;
  },

  exportJsonUrl: (studyId: string, agentId: string, params?: SessionListParams) => {
    const searchParams = new URLSearchParams();
    if (params?.status) searchParams.set("status", params.status);
    if (params?.date_from) searchParams.set("date_from", params.date_from);
    if (params?.date_to) searchParams.set("date_to", params.date_to);
    if (params?.session_ids) searchParams.set("session_ids", params.session_ids);
    const qs = searchParams.toString();
    return `${BASE}/studies/${studyId}/agents/${agentId}/sessions/export/json${qs ? `?${qs}` : ""}`;
  },

  /**
   * Trigger an authenticated CSV download. Uses a Bearer-authed fetch so the
   * browser doesn't strip the Authorization header (which it does for plain
   * <a download> clicks). Falls back to the response Content-Disposition
   * filename when present.
   */
  downloadCsv: (studyId: string, agentId: string, params?: SessionListParams) =>
    downloadAuthed(sessions.exportCsvUrl(studyId, agentId, params), "sessions.csv"),

  downloadJson: (studyId: string, agentId: string, params?: SessionListParams) =>
    downloadAuthed(sessions.exportJsonUrl(studyId, agentId, params), "sessions.json"),

  downloadAudioTurn: (
    studyId: string,
    agentId: string,
    sessionId: string,
    filename: string
  ) =>
    downloadAuthed(
      sessions.audioTurnUrl(studyId, agentId, sessionId, filename),
      filename
    ),
};

/** Fetch a URL with the auth token, then save the blob with a sensible filename. */
export async function downloadAuthed(url: string, fallbackName: string): Promise<void> {
  const token = getAuthToken();
  const headers: Record<string, string> = {};
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const res = await fetch(url, { headers });
  if (res.status === 401) {
    clearAuthToken();
    if (window.location.pathname !== "/login") window.location.href = "/login";
    throw new Error("Authentication required");
  }
  if (!res.ok) {
    let detail = `Download failed: ${res.status}`;
    try {
      const body = await res.json();
      if (body?.detail != null) {
        detail = formatApiError(body.detail, detail);
      }
    } catch {
      // not JSON, ignore
    }
    throw new Error(detail);
  }

  // Pull the filename from Content-Disposition when the backend sets it.
  let filename = fallbackName;
  const disp = res.headers.get("Content-Disposition") || "";
  const match = disp.match(/filename\*=UTF-8''([^;]+)|filename="?([^";]+)"?/i);
  if (match) filename = decodeURIComponent(match[1] || match[2] || fallbackName);

  const blob = await res.blob();
  const objectUrl = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = objectUrl;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(objectUrl);
}

// ── Settings (API Keys) ─────────────────────────────────────────

export interface ApiKeyStatus {
  field: string;
  env_var: string;
  is_set: boolean;
  source: "env" | "dashboard" | "none";
  masked_value: string;
}

export interface ApiKeysResponse {
  keys: ApiKeyStatus[];
}

export interface FlagStatus {
  field: string;
  env_var: string;
  enabled: boolean;
  source: "env" | "dashboard" | "default";
}

export interface FlagsResponse {
  flags: FlagStatus[];
}

export interface AudioStorageSettingStatus {
  field: string;
  env_var: string;
  is_set: boolean;
  source: "env" | "dashboard" | "none";
  display_value: string;
  sensitive: boolean;
}

export interface AudioStorageResponse {
  settings: AudioStorageSettingStatus[];
}

export interface CatalogModelOption {
  value: string;
  label: string;
  group: string;
  provider: string;
  api_kind: string;
}

export interface CatalogProviderOption {
  value: string;
  label: string;
  provider: string;
  models: { value: string; label: string }[];
}

export interface CatalogVoiceOption {
  value: string;
  label: string;
}

export interface ProviderCatalog {
  defaults: {
    modular_llm: string;
    text_llm: string;
    v2v_llm: string;
    stt_provider: string;
    stt_model: string;
    tts_provider: string;
    tts_model: string;
    tts_voice: string;
    v2v_voice_openai: string;
    v2v_voice_google: string;
    self_hosted_stt_model: string;
    self_hosted_tts_model: string;
    self_hosted_tts_voice: string;
  };
  llm_modular: CatalogModelOption[];
  llm_text: CatalogModelOption[];
  llm_v2v: CatalogModelOption[];
  stt_providers: CatalogProviderOption[];
  tts_providers: CatalogProviderOption[];
  voices: {
    openai_realtime: CatalogVoiceOption[];
    gemini_live: CatalogVoiceOption[];
    openai_tts: CatalogVoiceOption[];
    elevenlabs: CatalogVoiceOption[];
    cartesia: CatalogVoiceOption[];
  };
  supports_custom_llm: boolean;
}

export interface SmokeProbeResult {
  category: string;
  provider: string;
  model: string;
  endpoint: string;
  ok: boolean;
  latency_ms: number | null;
  error: string | null;
}

export interface SmokeTestResponse {
  live: boolean;
  total: number;
  passed: number;
  failed: number;
  results: SmokeProbeResult[];
}

export const settingsApi = {
  getKeys: () => request<ApiKeysResponse>("/settings/keys"),

  updateKeys: (updates: Record<string, string>) =>
    request<ApiKeysResponse>("/settings/keys", {
      method: "PUT",
      body: JSON.stringify(updates),
    }),

  getFlags: () => request<FlagsResponse>("/settings/flags"),

  updateFlags: (updates: Record<string, boolean>) =>
    request<FlagsResponse>("/settings/flags", {
      method: "PUT",
      body: JSON.stringify(updates),
    }),

  getAudioStorage: () =>
    request<AudioStorageResponse>("/settings/audio-storage"),

  updateAudioStorage: (updates: Record<string, string>) =>
    request<AudioStorageResponse>("/settings/audio-storage", {
      method: "PUT",
      body: JSON.stringify(updates),
    }),

  getAuthConfig: () =>
    request<{ auth_enabled: boolean; username: string }>("/settings/auth"),

  getCatalog: () => request<ProviderCatalog>("/settings/catalog"),

  smokeTest: (live = true) =>
    request<SmokeTestResponse>("/settings/smoke-test", {
      method: "POST",
      body: JSON.stringify({ live }),
    }),
};
