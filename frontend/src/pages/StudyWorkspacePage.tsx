import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import {
  agents,
  composer,
  participants,
  studies,
  type Agent,
  type ComposerEditChange,
  type ComposerEditEvent,
  type ComposerDraft,
  type InterviewGuideConfig,
  type InterviewQuestionConfig,
  type ParticipantIdentifier,
  type ResearchSpec,
  type Study,
  type StudyAnalytics,
} from "../lib/api";
import { defaultResearchSpec } from "../components/StudyProtocolBuilder";

type WorkspaceTab = "guide" | "launch";
type PreviewRole = "parent" | "educator";
type PreviewVisit = "first_time" | "returning";

interface WorkspaceDraft {
  title: string;
  description: string;
  researchSpec: ResearchSpec;
  systemPrompt: string;
  welcomeMessage: string;
  guide: InterviewGuideConfig;
  estimatedMinutes: number;
}

interface ComposerMessage {
  role: "researcher" | "composer";
  text: string;
}

type EditSource = ComposerEditEvent["source"];

interface CommitMeta {
  source?: EditSource;
  summary?: string;
}

const EDIT_HISTORY_LIMIT = 50;
const EDIT_HISTORY_CONTEXT_CHARS = 18_000;
const EDIT_COALESCE_MS = 2_000;
const EDIT_CHANGE_LIMIT = 12;

const emptyQuestion = (index: number): InterviewQuestionConfig => ({
  id: `question_${index + 1}`,
  module: "Core interview",
  text: "",
  probes: [],
  max_follow_ups: 2,
  transition: null,
  purpose: "",
  target_ids: [],
  evidence_goal: [],
  do_not_assume: [],
  priority: "required",
  sensitivity: "standard",
  phase: "core",
  visit_scope: "all",
  skip_remaining_module_if_not_applicable: false,
  always_ask: false,
  depends_on: [],
  role_variants: {},
  returning_role_variants: {},
});

const emptyGuide = (): InterviewGuideConfig => ({
  questions: [emptyQuestion(0)],
  orchestration_profile: "semi_structured",
  closing_message: "Before we finish, is there anything important that I did not ask about?",
  allow_follow_ups: true,
  skip_answered_questions: true,
  probe_selection: "relevance",
  allow_emergent_follow_ups: false,
});

type OrchestrationProfile = NonNullable<InterviewGuideConfig["orchestration_profile"]>;

const orchestrationRules: Record<OrchestrationProfile, Pick<InterviewGuideConfig,
  "allow_follow_ups" | "skip_answered_questions" | "allow_emergent_follow_ups">> = {
  fixed: { allow_follow_ups: false, skip_answered_questions: false, allow_emergent_follow_ups: false },
  repair_only: { allow_follow_ups: false, skip_answered_questions: true, allow_emergent_follow_ups: false },
  semi_structured: { allow_follow_ups: true, skip_answered_questions: true, allow_emergent_follow_ups: false },
  exploratory: { allow_follow_ups: true, skip_answered_questions: true, allow_emergent_follow_ups: true },
};

const resolvedProfile = (guide: InterviewGuideConfig): OrchestrationProfile => {
  if (guide.orchestration_profile) return guide.orchestration_profile;
  const hasCapacity = guide.questions.some((question) =>
    question.max_follow_ups > 0 && ((question.probes || []).length > 0 || guide.allow_emergent_follow_ups));
  if (guide.allow_follow_ups === false || !hasCapacity) {
    return guide.skip_answered_questions === true ? "repair_only" : "fixed";
  }
  return guide.allow_emergent_follow_ups ? "exploratory" : "semi_structured";
};

const withProfile = (guide: InterviewGuideConfig, profile: OrchestrationProfile): InterviewGuideConfig => ({
  ...guide,
  conversation_policy: profile === "exploratory" ? guide.conversation_policy : null,
  orchestration_profile: profile,
  ...orchestrationRules[profile],
});

const uniqueQuestionId = (questions: InterviewQuestionConfig[], source: string): string => {
  const base = (source || "question").replace(/_copy(?:_\d+)?$/, "");
  const used = new Set(questions.map((question) => question.id).filter(Boolean));
  let suffix = 2;
  let candidate = `${base}_${suffix}`;
  while (used.has(candidate)) candidate = `${base}_${++suffix}`;
  return candidate;
};

const normalizeSpec = (value: ResearchSpec | null): ResearchSpec => {
  const base = defaultResearchSpec();
  if (!value) return base;
  return {
    ...base,
    ...value,
    population: { ...base.population, ...value.population },
    clarification_policy: { ...base.clarification_policy, ...value.clarification_policy },
    interviewer_stance: { ...base.interviewer_stance, ...value.interviewer_stance },
    participant_information: { ...base.participant_information, ...value.participant_information },
    fielding: { ...base.fielding, ...value.fielding },
  };
};

const toDraft = (study: Study, agent: Agent): WorkspaceDraft => {
  const spec = normalizeSpec(study.research_spec);
  return {
    title: study.title,
    description: study.description || "",
    researchSpec: spec,
    systemPrompt: agent.system_prompt || "",
    welcomeMessage: agent.welcome_message || "Thanks for joining. I’ll ask one question at a time, and you can skip any question.",
    guide: agent.interview_guide
      ? withProfile(agent.interview_guide, resolvedProfile(agent.interview_guide))
      : emptyGuide(),
    estimatedMinutes: spec.participant_information.estimated_minutes || 15,
  };
};

// Preserve in-progress whitespace and blank lines while the researcher types.
// Values are normalized only when sent to the composer or saved.
const splitLines = (value: string) => value.split("\n");
const joinLines = (value?: string[]) => (value || []).join("\n");
const clone = <T,>(value: T): T => JSON.parse(JSON.stringify(value));
const compactEditValue = (value: unknown): unknown => {
  if (value === undefined) return null;
  if (typeof value === "string") return value.length > 500 ? `${value.slice(0, 497)}…` : value;
  if (value === null || typeof value === "number" || typeof value === "boolean") return value;
  const serialized = JSON.stringify(value);
  return serialized.length > 800 ? `${serialized.slice(0, 797)}…` : clone(value);
};
const sameValue = (left: unknown, right: unknown) => JSON.stringify(left) === JSON.stringify(right);
const draftChanges = (before: WorkspaceDraft, after: WorkspaceDraft): ComposerEditChange[] => {
  const changes: ComposerEditChange[] = [];
  const visit = (left: unknown, right: unknown, path: string) => {
    if (changes.length >= EDIT_CHANGE_LIMIT || sameValue(left, right)) return;
    if (Array.isArray(left) && Array.isArray(right)) {
      if (path === "guide.questions") {
        const leftIds = left.map((item) => (item as InterviewQuestionConfig)?.id || null);
        const rightIds = right.map((item) => (item as InterviewQuestionConfig)?.id || null);
        if (left.length !== right.length || !sameValue(leftIds, rightIds)) {
          changes.push({ path: "guide.question_order", before: compactEditValue(leftIds), after: compactEditValue(rightIds) });
          return;
        }
      }
      const complex = left.some((item) => item !== null && typeof item === "object")
        || right.some((item) => item !== null && typeof item === "object");
      if (!complex || left.length !== right.length) {
        changes.push({ path, before: compactEditValue(left), after: compactEditValue(right) });
        return;
      }
      left.forEach((item, index) => visit(item, right[index], `${path}[${index}]`));
      return;
    }
    if (left !== null && right !== null && typeof left === "object" && typeof right === "object") {
      const keys = new Set([...Object.keys(left as object), ...Object.keys(right as object)]);
      keys.forEach((key) => visit(
        (left as Record<string, unknown>)[key],
        (right as Record<string, unknown>)[key],
        path ? `${path}.${key}` : key,
      ));
      return;
    }
    changes.push({ path, before: compactEditValue(left), after: compactEditValue(right) });
  };
  visit(before, after, "");
  return changes;
};
const cleanLines = (value?: string[]) => (value || []).map((line) => line.trim()).filter(Boolean);
const cleanGuide = (guide: InterviewGuideConfig): InterviewGuideConfig => ({
  ...clone(guide),
  questions: guide.questions.map((question) => ({
    ...clone(question),
    probes: cleanLines(question.probes),
    target_ids: cleanLines(question.target_ids),
    evidence_goal: cleanLines(question.evidence_goal),
    do_not_assume: cleanLines(question.do_not_assume),
  })),
});
export const guideForStudyComposer = (
  guide: InterviewGuideConfig,
): InterviewGuideConfig | null => {
  const cleaned = cleanGuide(guide);
  const questions = cleaned.questions.filter((question) => question.text.trim());
  return questions.length ? { ...cleaned, questions } : null;
};
const cleanResearchSpec = (spec: ResearchSpec): ResearchSpec => {
  const next = clone(spec);
  next.research_questions = cleanLines(next.research_questions);
  next.population.knowledge_limits = cleanLines(next.population.knowledge_limits);
  next.clarification_policy.in_scope = cleanLines(next.clarification_policy.in_scope);
  next.clarification_policy.out_of_scope = cleanLines(next.clarification_policy.out_of_scope);
  next.clarification_policy.ambiguous_cases = cleanLines(next.clarification_policy.ambiguous_cases);
  next.participant_information.rights = cleanLines(next.participant_information.rights);
  return next;
};
const roleText = (question: InterviewQuestionConfig, role: string) => {
  const value = question.role_variants?.[role];
  return typeof value === "string" ? value : value?.text || "";
};
const questionThreadKey = (question: InterviewQuestionConfig, index: number) => question.id || `question_${index}`;
const questionComposerWelcome = (index: number): ComposerMessage => ({
  role: "composer",
  text: `What would you like to change about Question ${index + 1}? I’ll edit only this card.`,
});

function qaChecks(draft: WorkspaceDraft) {
  const checks: Array<{ tone: "good" | "warn"; text: string }> = [];
  const questions = draft.guide.questions;
  const firstTimeQuestions = questions.filter((question) => question.visit_scope !== "returning");
  const firstTimeWarmUp = firstTimeQuestions.filter((question) => question.phase === "warm_up");
  const firstCoreIndex = firstTimeQuestions.findIndex((question) => (question.phase || "core") === "core");
  const lastWarmUpIndex = firstTimeQuestions.reduce((last, question, index) => question.phase === "warm_up" ? index : last, -1);
  if (firstTimeQuestions.length >= 6 && firstTimeQuestions.length <= 10) checks.push({ tone: "good", text: `${firstTimeQuestions.length} first-time questions is a focused guide.` });
  else checks.push({ tone: "warn", text: `${firstTimeQuestions.length} first-time questions; 6–10 including warm-up is a useful starting range.` });
  if (firstTimeWarmUp.length >= 1 && firstTimeWarmUp.length <= 2 && lastWarmUpIndex < firstCoreIndex) checks.push({ tone: "good", text: `${firstTimeWarmUp.length} low-burden warm-up question${firstTimeWarmUp.length === 1 ? "" : "s"} precede the core interview.` });
  else checks.push({ tone: "warn", text: "Add 1–2 first-time warm-up questions before the core interview." });
  if (firstTimeWarmUp.every((question) => question.max_follow_ups <= 1)) checks.push({ tone: "good", text: "Warm-up is bounded and will not crowd out the core interview." });
  else checks.push({ tone: "warn", text: "Cap warm-up follow-ups at one per question." });
  if (draft.researchSpec.population.participant_role.trim()) checks.push({ tone: "good", text: "Target participant role is explicit." });
  else checks.push({ tone: "warn", text: "Define who is speaking so wording can adapt to their role." });
  if (draft.researchSpec.targets.length) checks.push({ tone: "good", text: "Questions can be traced to evidence targets." });
  else checks.push({ tone: "warn", text: "Add evidence targets before fielding." });
  if (draft.researchSpec.participant_information.privacy_reminder.trim()) checks.push({ tone: "good", text: "Privacy guidance is present." });
  else checks.push({ tone: "warn", text: "Add a privacy reminder for third-party and minor data." });
  if (draft.researchSpec.participant_information.data_handling_summary.trim()) checks.push({ tone: "good", text: "Participant-facing data handling and protection are explained." });
  else checks.push({ tone: "warn", text: "Explain how participant data is handled and protected before fielding." });
  questions.forEach((question, index) => {
    const lower = question.text.toLowerCase();
    if (/\b(don't you|wouldn't you|how concerned are you)\b/.test(lower)) checks.push({ tone: "warn", text: `Q${index + 1} may lead the participant.` });
    if ((question.text.match(/\?/g) || []).length > 1) checks.push({ tone: "warn", text: `Q${index + 1} asks more than one question at once.` });
    if (!question.purpose?.trim() || !cleanLines(question.evidence_goal).length) checks.push({ tone: "warn", text: `Q${index + 1} needs a purpose and evidence goal.` });
  });
  return checks;
}

export default function StudyWorkspacePage() {
  const { studyId, agentId } = useParams<{ studyId: string; agentId: string }>();
  const [tab, setTab] = useState<WorkspaceTab>("guide");
  const [study, setStudy] = useState<Study | null>(null);
  const [agent, setAgent] = useState<Agent | null>(null);
  const [draft, setDraft] = useState<WorkspaceDraft | null>(null);
  const [participantList, setParticipantList] = useState<ParticipantIdentifier[]>([]);
  const [analytics, setAnalytics] = useState<StudyAnalytics | null>(null);
  const [selectedQuestion, setSelectedQuestion] = useState(0);
  const [previewRole, setPreviewRole] = useState<PreviewRole>("parent");
  const [previewVisit, setPreviewVisit] = useState<PreviewVisit>("first_time");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState("");
  const [composerInput, setComposerInput] = useState("");
  const [composing, setComposing] = useState(false);
  const [composerMessages, setComposerMessages] = useState<ComposerMessage[]>([
    { role: "composer", text: "Tell me what you want to learn or ask me to revise the current guide. I’ll propose an editable draft—nothing goes live until you save and launch." },
  ]);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [questionComposerIndex, setQuestionComposerIndex] = useState<number | null>(null);
  const [questionComposerInput, setQuestionComposerInput] = useState("");
  const [questionComposerThreads, setQuestionComposerThreads] = useState<Record<string, ComposerMessage[]>>({});
  const [questionComposing, setQuestionComposing] = useState(false);
  const [questionComposerError, setQuestionComposerError] = useState("");
  const [participantRows, setParticipantRows] = useState("");
  const [importing, setImporting] = useState(false);
  const history = useRef<WorkspaceDraft[]>([]);
  const future = useRef<WorkspaceDraft[]>([]);
  const manuallyEditedQuestionIds = useRef<Set<string>>(new Set());
  const editHistory = useRef<ComposerEditEvent[]>([]);

  const editHistoryStorageKey = studyId && agentId
    ? `oasis:composer-edit-history:${studyId}:${agentId}`
    : null;

  useEffect(() => {
    if (!editHistoryStorageKey) return;
    try {
      const stored = JSON.parse(window.localStorage.getItem(editHistoryStorageKey) || "[]");
      editHistory.current = Array.isArray(stored) ? stored.slice(-EDIT_HISTORY_LIMIT) : [];
    } catch {
      editHistory.current = [];
    }
  }, [editHistoryStorageKey]);

  useEffect(() => {
    if (!studyId || !agentId) return;
    Promise.all([
      studies.get(studyId),
      agents.get(studyId, agentId),
      participants.list(studyId, agentId).catch(() => [] as ParticipantIdentifier[]),
      studies.analytics(studyId).catch(() => null),
    ]).then(([loadedStudy, loadedAgent, loadedParticipants, loadedAnalytics]) => {
      setStudy(loadedStudy);
      setAgent(loadedAgent);
      setDraft(toDraft(loadedStudy, loadedAgent));
      setParticipantList(loadedParticipants);
      setAnalytics(loadedAnalytics);
      if (loadedStudy.description) setComposerInput(`Create a strong first guide for this brief: ${loadedStudy.description}`);
    }).catch((err) => setError(err instanceof Error ? err.message : "Could not load study"))
      .finally(() => setLoading(false));
  }, [studyId, agentId]);

  const persistEditHistory = () => {
    if (!editHistoryStorageKey) return;
    try {
      window.localStorage.setItem(editHistoryStorageKey, JSON.stringify(editHistory.current));
    } catch {
      // A storage quota or privacy setting should never block draft editing.
    }
  };

  const recordDraftEdit = (before: WorkspaceDraft, after: WorkspaceDraft, meta: CommitMeta = {}) => {
    const changes = draftChanges(before, after);
    if (!changes.length) return;
    const now = new Date();
    const source = meta.source || "manual";
    const previous = editHistory.current[editHistory.current.length - 1];
    const paths = changes.map((change) => change.path).join("|");
    const previousPaths = previous?.changes.map((change) => change.path).join("|");
    if (
      source === "manual"
      && previous?.source === "manual"
      && paths === previousPaths
      && now.getTime() - Date.parse(previous.timestamp) <= EDIT_COALESCE_MS
    ) {
      previous.timestamp = now.toISOString();
      previous.changes = changes.map((change, index) => ({
        ...change,
        before: previous.changes[index]?.before ?? change.before,
      }));
      previous.summary = meta.summary || previous.summary;
    } else {
      editHistory.current.push({
        id: globalThis.crypto?.randomUUID?.() || `${now.getTime()}-${Math.random().toString(16).slice(2)}`,
        timestamp: now.toISOString(),
        source,
        summary: meta.summary,
        changes,
      });
      editHistory.current = editHistory.current.slice(-EDIT_HISTORY_LIMIT);
    }
    persistEditHistory();
  };

  const editHistoryForComposer = () => {
    const newestThatFit: ComposerEditEvent[] = [];
    let usedCharacters = 0;
    for (const event of [...editHistory.current].reverse()) {
      const size = JSON.stringify(event).length;
      if (newestThatFit.length && usedCharacters + size > EDIT_HISTORY_CONTEXT_CHARS) break;
      if (size > EDIT_HISTORY_CONTEXT_CHARS) continue;
      newestThatFit.push(event);
      usedCharacters += size;
    }
    return newestThatFit.reverse();
  };

  const commit = (next: WorkspaceDraft, meta: CommitMeta = {}) => {
    if (draft) {
      history.current.push(clone(draft));
      recordDraftEdit(draft, next, meta);
    }
    future.current = [];
    setDraft(next);
    setSaved(false);
  };

  const patchDraft = (patch: Partial<WorkspaceDraft>) => {
    if (!draft) return;
    commit({ ...draft, ...patch });
  };

  const patchSpec = (patch: Partial<ResearchSpec>) => {
    if (!draft) return;
    commit({ ...draft, researchSpec: { ...draft.researchSpec, ...patch } });
  };

  const patchQuestion = (index: number, patch: Partial<InterviewQuestionConfig>) => {
    if (!draft) return;
    const questionId = draft.guide.questions[index]?.id;
    if (questionId) manuallyEditedQuestionIds.current.add(questionId);
    const questions = draft.guide.questions.map((question, i) => i === index ? { ...question, ...patch } : question);
    commit({ ...draft, guide: { ...draft.guide, questions } });
  };

  const moveQuestion = (from: number, to: number) => {
    if (!draft || to < 0 || to >= draft.guide.questions.length) return;
    const questions = [...draft.guide.questions];
    const [moved] = questions.splice(from, 1);
    questions.splice(to, 0, moved);
    commit({ ...draft, guide: { ...draft.guide, questions } });
    setSelectedQuestion(to);
  };

  const patchRoleQuestion = (index: number, role: string, text: string) => {
    if (!draft) return;
    const question = draft.guide.questions[index];
    const variants = { ...(question.role_variants || {}) };
    if (text.trim()) variants[role] = text;
    else delete variants[role];
    patchQuestion(index, { role_variants: variants });
  };

  const undo = () => {
    if (!draft || !history.current.length) return;
    future.current.push(clone(draft));
    const next = history.current.pop() || draft;
    recordDraftEdit(draft, next, { source: "undo", summary: "Undid the most recent draft change." });
    setDraft(next);
    setSaved(false);
  };

  const redo = () => {
    if (!draft || !future.current.length) return;
    history.current.push(clone(draft));
    const next = future.current.pop() || draft;
    recordDraftEdit(draft, next, { source: "redo", summary: "Redid the most recent draft change." });
    setDraft(next);
    setSaved(false);
  };

  const applyComposerDraft = (result: ComposerDraft) => {
    if (!draft) return;
    const spec = normalizeSpec(result.research_spec);
    spec.participant_information.estimated_minutes = result.estimated_minutes;
    commit({
      title: result.title,
      description: draft.description,
      researchSpec: spec,
      systemPrompt: result.system_prompt,
      welcomeMessage: result.welcome_message,
      guide: result.interview_guide,
      estimatedMinutes: result.estimated_minutes,
    }, { source: "study_composer", summary: result.summary });
    setWarnings(result.warnings || []);
    setComposerMessages((messages) => [...messages, { role: "composer", text: result.summary }]);
    setSelectedQuestion(Math.min(selectedQuestion, Math.max(0, result.interview_guide.questions.length - 1)));
    manuallyEditedQuestionIds.current.clear();
  };

  const runComposer = async (instruction?: string) => {
    if (!studyId || !agentId || !draft) return;
    const text = (instruction || composerInput).trim();
    if (!text) return;
    setComposing(true); setError("");
    setComposerMessages((messages) => [...messages, { role: "researcher", text }]);
    setComposerInput("");
    try {
      const result = await composer.compose(studyId, {
        instruction: text,
        agent_id: agentId,
        current_research_spec: cleanResearchSpec(draft.researchSpec),
        current_interview_guide: guideForStudyComposer(draft.guide),
        current_system_prompt: draft.systemPrompt,
        current_welcome_message: draft.welcomeMessage,
        conversation_history: composerMessages.slice(1).slice(-12),
        selected_question_index: selectedQuestion,
        selected_question_id: draft.guide.questions[selectedQuestion]?.id || undefined,
        edited_question_ids: [...manuallyEditedQuestionIds.current],
        edit_history: editHistoryForComposer(),
      });
      applyComposerDraft(result);
    } catch (err) {
      const message = err instanceof Error ? err.message : "The composer could not revise the guide";
      setError(message);
      setComposerMessages((messages) => [...messages, { role: "composer", text: `I couldn't apply that revision: ${message}` }]);
    } finally { setComposing(false); }
  };

  const openQuestionComposer = (index: number) => {
    if (!draft) return;
    const question = draft.guide.questions[index];
    if (!question) return;
    const key = questionThreadKey(question, index);
    setQuestionComposerThreads((threads) => threads[key] ? threads : {
      ...threads,
      [key]: [questionComposerWelcome(index)],
    });
    setSelectedQuestion(index);
    setQuestionComposerIndex(index);
    setQuestionComposerInput("");
    setQuestionComposerError("");
  };

  const runQuestionComposer = async (instruction?: string) => {
    if (!studyId || !agentId || !draft || questionComposerIndex === null) return;
    const text = (instruction || questionComposerInput).trim();
    if (!text) return;
    const index = questionComposerIndex;
    const question = draft.guide.questions[index];
    if (!question) return;
    const key = questionThreadKey(question, index);
    const currentThread = questionComposerThreads[key] || [questionComposerWelcome(index)];
    const researcherTurn: ComposerMessage = { role: "researcher", text };

    setQuestionComposing(true);
    setQuestionComposerError("");
    setQuestionComposerInput("");
    setQuestionComposerThreads((threads) => ({
      ...threads,
      [key]: [...currentThread, researcherTurn],
    }));

    try {
      const result = await composer.composeQuestion(studyId, {
        instruction: text,
        agent_id: agentId,
        question_index: index,
        current_research_spec: cleanResearchSpec(draft.researchSpec),
        current_interview_guide: cleanGuide(draft.guide),
        conversation_history: currentThread.slice(1).slice(-12),
        edit_history: editHistoryForComposer(),
      });
      const revisedQuestion = { ...result.question, id: question.id || result.question.id };
      const questions = draft.guide.questions.map((current, questionIndex) => (
        questionIndex === index ? revisedQuestion : current
      ));
      commit(
        { ...draft, guide: { ...draft.guide, questions } },
        { source: "question_composer", summary: result.summary },
      );
      if (question.id) manuallyEditedQuestionIds.current.delete(question.id);
      if (result.warnings.length) setWarnings(result.warnings);
      setQuestionComposerThreads((threads) => ({
        ...threads,
        [key]: [...currentThread, researcherTurn, { role: "composer", text: result.summary }],
      }));
    } catch (err) {
      const message = err instanceof Error ? err.message : "The question editor could not revise this card";
      setQuestionComposerError(message);
      setQuestionComposerThreads((threads) => ({
        ...threads,
        [key]: [...currentThread, researcherTurn, { role: "composer", text: `I couldn't revise this card: ${message}` }],
      }));
    } finally {
      setQuestionComposing(false);
    }
  };

  const save = async () => {
    if (!studyId || !agentId || !draft) return;
    setSaving(true); setError("");
    try {
      const nextSpec = cleanResearchSpec(draft.researchSpec);
      const nextGuide = cleanGuide(draft.guide);
      nextSpec.participant_information.estimated_minutes = draft.estimatedMinutes;
      const [savedStudy, savedAgent] = await Promise.all([
        studies.update(studyId, { title: draft.title, description: draft.description, research_spec: nextSpec }),
        agents.update(studyId, agentId, {
          modality: agent?.modality || "voice",
          pipeline_type: "modular",
          interview_mode: "structured",
          system_prompt: draft.systemPrompt,
          welcome_message: draft.welcomeMessage,
          interview_guide: nextGuide,
          widget_title: draft.title,
          widget_description: nextSpec.participant_information.purpose_summary || draft.description,
          widget_show_progress: agent?.widget_show_progress ?? true,
          // This is the participant-facing target. Runtime orchestration adds
          // a bounded grace period instead of advertising a shorter duration.
          max_duration_seconds: Math.max(300, draft.estimatedMinutes * 60),
          participant_id_mode: nextSpec.fielding.recruitment_source === "prolific" ? "external" : agent?.participant_id_mode || "random",
        }),
      ]);
      setStudy(savedStudy); setAgent(savedAgent); setDraft({ ...draft, researchSpec: nextSpec, guide: nextGuide }); setSaved(true);
      manuallyEditedQuestionIds.current.clear();
      window.setTimeout(() => setSaved(false), 2500);
    } catch (err) { setError(err instanceof Error ? err.message : "Could not save the guide"); }
    finally { setSaving(false); }
  };

  const setLaunchStatus = async (status: "active" | "paused") => {
    if (!studyId || !agentId || !draft) return;
    if (status === "active") {
      const info = draft.researchSpec.participant_information;
      const missingTerms = [
        [info.participation_summary, "what participation involves"],
        [info.data_handling_summary, "data handling"],
        [info.data_controller, "data controller"],
        [info.data_recipients_summary, "who receives the data"],
        [info.risks_summary, "risks or discomforts"],
        [info.compensation_summary, "compensation"],
        [info.researcher_identity, "researcher identity"],
        [info.researcher_contact, "researcher contact"],
        [info.complaints_contact, "complaints contact"],
        [info.data_retention_period, "retention period"],
        [info.withdrawal_instructions, "withdrawal/deletion instructions"],
      ].filter(([value]) => !String(value || "").trim()).map(([, label]) => label);
      if (missingTerms.length) {
        setError(`Complete the participant information before launch: ${missingTerms.join(", ")}.`);
        return;
      }
      if (draft.researchSpec.fielding.recruitment_source === "prolific") {
        const fielding = draft.researchSpec.fielding;
        if (!fielding.completion_url.trim() || !fielding.no_consent_url.trim() || !fielding.inactivity_url.trim()) {
          setError("Add Prolific completion, no-consent, and inactivity return URLs before launch.");
          return;
        }
      }
    }
    setSaving(true); setError("");
    try {
      if (status === "active") await save();
      const savedStudy = await studies.update(studyId, { status });
      const savedAgent = await agents.update(studyId, agentId, { status });
      setStudy(savedStudy); setAgent(savedAgent);
    } catch (err) { setError(err instanceof Error ? err.message : "Could not change launch status"); }
    finally { setSaving(false); }
  };

  const importProfiles = async () => {
    if (!studyId || !agentId) return;
    const rows = participantRows.split("\n").map((row) => row.trim()).filter(Boolean).map((row) => {
      const [identifier, role, label] = row.split(",").map((cell) => cell.trim());
      return { identifier, label: label || null, profile: role ? { role } : {} };
    }).filter((row) => row.identifier);
    if (!rows.length) return;
    setImporting(true); setError("");
    try {
      const created = await participants.bulkCreateProfiles(studyId, agentId, rows);
      setParticipantList((current) => [...created, ...current]); setParticipantRows("");
    } catch (err) { setError(err instanceof Error ? err.message : "Could not import participants"); }
    finally { setImporting(false); }
  };

  const checks = useMemo(() => draft ? qaChecks(draft) : [], [draft]);
  const firstTimeQuestionCount = draft?.guide.questions.filter((question) => question.visit_scope !== "returning").length || 0;
  const returningQuestionCount = draft?.guide.questions.filter((question) => question.visit_scope !== "first_time").length || 0;
  const completed = analytics?.completed_sessions || 0;
  const responseLimit = draft?.researchSpec.fielding.response_limit || 50;
  const responseProgress = Math.min(100, Math.round((completed / responseLimit) * 100));

  if (loading) return <div className="py-20 text-center text-sm text-gray-400">Loading study workspace…</div>;
  if (!study || !agent || !draft || !studyId || !agentId) return <div className="py-20 text-center text-sm text-red-600">{error || "Study workspace not found."}</div>;

  const currentQuestion = draft.guide.questions[selectedQuestion] || draft.guide.questions[0];
  const previewQuestions = draft.guide.questions.filter((question) => (
    question.visit_scope === "all" || !question.visit_scope || question.visit_scope === previewVisit
  ));
  const selectedPreviewIndex = previewQuestions.findIndex((question) => question === currentQuestion);
  const previewQuestion = selectedPreviewIndex >= 0 ? currentQuestion : previewQuestions[0];
  const previewQuestionIndex = Math.max(0, selectedPreviewIndex);
  const previewQuestionText = previewQuestion
    ? roleText(previewQuestion, previewRole) || previewQuestion.text
    : "Your question appears here.";
  const interviewUrl = `${window.location.origin}/interview/${agent.widget_key}`;
  const prolificInterviewUrl = `${interviewUrl}?PROLIFIC_PID={{%PROLIFIC_PID%}}&STUDY_ID={{%STUDY_ID%}}&SESSION_ID={{%SESSION_ID%}}`;
  const activeQuestionComposerCard = questionComposerIndex === null
    ? null
    : draft.guide.questions[questionComposerIndex] || null;
  const activeQuestionComposerKey = activeQuestionComposerCard && questionComposerIndex !== null
    ? questionThreadKey(activeQuestionComposerCard, questionComposerIndex)
    : null;
  const activeQuestionComposerMessages = activeQuestionComposerKey
    ? questionComposerThreads[activeQuestionComposerKey] || [questionComposerWelcome(questionComposerIndex || 0)]
    : [];

  return (
    <div className="-mx-4 -my-8 sm:-mx-6 lg:-mx-8 min-h-[calc(100vh-3.5rem)] bg-[#f7f8fb]">
      <header className="sticky top-14 z-30 border-b border-gray-200 bg-white/95 backdrop-blur">
        <div className="px-5 pt-3">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex min-w-0 items-center gap-3">
              <Link to={`/studies/${studyId}`} className="rounded-lg p-1.5 text-gray-400 hover:bg-gray-100 hover:text-gray-700" aria-label="Back to study">←</Link>
              <input className="min-w-0 max-w-xl border-0 bg-transparent text-base font-semibold text-gray-900 outline-none focus:ring-0" value={draft.title} onChange={(event) => patchDraft({ title: event.target.value })} />
              <span className={`rounded-full px-2 py-1 text-[10px] font-semibold uppercase tracking-wide ${study.status === "active" ? "bg-emerald-50 text-emerald-700" : "bg-gray-100 text-gray-600"}`}>{study.status}</span>
            </div>
            <div className="flex items-center gap-2">
              <button type="button" onClick={undo} disabled={!history.current.length} className="rounded-lg border border-gray-200 px-2.5 py-1.5 text-xs text-gray-600 disabled:opacity-30">↶ Undo</button>
              <button type="button" onClick={redo} disabled={!future.current.length} className="rounded-lg border border-gray-200 px-2.5 py-1.5 text-xs text-gray-600 disabled:opacity-30">Redo ↷</button>
              <a href={interviewUrl} target="_blank" rel="noreferrer" className="rounded-lg border border-gray-200 px-3 py-1.5 text-xs font-medium text-gray-700 hover:bg-gray-50">Preview</a>
              <button type="button" onClick={save} disabled={saving} className="rounded-lg bg-[#2f5bea] px-4 py-1.5 text-xs font-semibold text-white hover:bg-[#244bc8] disabled:opacity-50">{saving ? "Saving…" : saved ? "Saved" : "Save draft"}</button>
            </div>
          </div>
          <nav className="mt-3 flex items-center gap-6 text-sm">
            <button onClick={() => setTab("guide")} className={`border-b-2 pb-2.5 font-medium ${tab === "guide" ? "border-[#2f5bea] text-gray-900" : "border-transparent text-gray-500"}`}>Study guide</button>
            <button onClick={() => setTab("launch")} className={`border-b-2 pb-2.5 font-medium ${tab === "launch" ? "border-[#2f5bea] text-gray-900" : "border-transparent text-gray-500"}`}>Recruit & launch</button>
            <Link to={`/studies/${studyId}/agents/${agentId}/sessions`} className="border-b-2 border-transparent pb-2.5 font-medium text-gray-500 hover:text-gray-900">Responses <span className="ml-1 rounded-full bg-gray-100 px-1.5 py-0.5 text-[10px]">{analytics?.total_sessions || 0}</span></Link>
            <Link to={`/studies/${studyId}/agents/${agentId}`} className="border-b-2 border-transparent pb-2.5 font-medium text-gray-500 hover:text-gray-900">Advanced settings</Link>
          </nav>
        </div>
      </header>

      {error && <div className="mx-5 mt-4 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-xs text-red-700">{error}</div>}

      {tab === "guide" ? (
        <div className="grid min-h-[calc(100vh-8.8rem)] grid-cols-1 xl:grid-cols-[300px_minmax(520px,1fr)_330px]">
          <aside className="border-r border-gray-200 bg-white p-4">
            <div className="mb-3 flex items-center gap-2"><div className="flex h-7 w-7 items-center justify-center rounded-lg bg-[#e9efff] text-sm text-[#2f5bea]">✦</div><div><h2 className="text-sm font-semibold text-gray-900">Study composer</h2><p className="text-[10px] text-gray-400">AI proposes. You decide.</p></div></div>
            <div className="max-h-[38vh] space-y-3 overflow-y-auto pr-1">
              {composerMessages.map((message, index) => <div key={index} className={`rounded-xl px-3 py-2.5 text-xs leading-relaxed ${message.role === "researcher" ? "ml-5 bg-[#2f5bea] text-white" : "mr-2 bg-gray-100 text-gray-700"}`}>{message.text}</div>)}
              {composing && <div className="mr-2 rounded-xl bg-gray-100 px-3 py-2.5 text-xs leading-relaxed text-gray-500">Researching relevant evidence, drafting the guide, and auditing its flow and timing. This can take a few minutes…</div>}
            </div>
            {!composerMessages.some((message) => message.role === "researcher") && <div className="mt-4"><p className="mb-1.5 text-[9px] font-semibold uppercase tracking-[0.14em] text-gray-400">Quick edits</p><div className="flex flex-wrap gap-1.5">
              {["Make every question neutral", "Add educator-specific wording", "Shorten this to 12 minutes", "Find assumptions and compound questions"].map((suggestion) => <button key={suggestion} onClick={() => runComposer(suggestion)} disabled={composing} className="rounded-full border border-gray-200 bg-white px-2.5 py-1 text-[10px] text-gray-600 hover:border-[#9bb0f4] hover:text-[#2f5bea] disabled:opacity-50">{suggestion}</button>)}
            </div></div>}
            <form onSubmit={(event) => { event.preventDefault(); runComposer(); }} className="mt-3">
              <textarea value={composerInput} onChange={(event) => setComposerInput(event.target.value)} rows={5} placeholder="Describe the study, paste a draft guide, or ask for a revision…" className="w-full resize-none rounded-xl border border-gray-200 bg-white p-3 text-xs leading-relaxed outline-none focus:border-[#7290ef] focus:ring-2 focus:ring-[#e7ecff]" />
              <button disabled={composing || !composerInput.trim()} className="mt-2 w-full rounded-lg bg-gray-900 px-3 py-2 text-xs font-semibold text-white disabled:opacity-40">{composing ? "Building reviewed draft…" : "Revise draft"}</button>
            </form>
            <p className="mt-2 text-[9px] leading-relaxed text-gray-400">Composer context includes up to 50 recent manual, AI, undo, and redo edits. The current draft always wins.</p>
            {warnings.length > 0 && <div className="mt-4 rounded-xl border border-amber-200 bg-amber-50 p-3"><p className="text-[10px] font-semibold uppercase tracking-wide text-amber-700">Composer notes</p>{warnings.map((warning) => <p key={warning} className="mt-1.5 text-[11px] leading-relaxed text-amber-800">• {warning}</p>)}</div>}
          </aside>

          <main className="p-5 lg:p-7">
            <div className="mx-auto max-w-3xl space-y-4">
              <section className="rounded-2xl border border-gray-200 bg-white p-5 shadow-sm">
                <div className="mb-4 flex items-center justify-between"><h2 className="text-base font-semibold text-gray-900">Study foundation</h2><span className="text-[10px] text-gray-400">Shared across interviewer versions</span></div>
                <div className="grid gap-4 md:grid-cols-2">
                  <label className="block md:col-span-2"><span className="mb-1 block text-[11px] font-semibold text-gray-600">What decision will this research inform?</span><textarea value={draft.researchSpec.decision_to_inform} onChange={(event) => patchSpec({ decision_to_inform: event.target.value })} rows={2} className="w-full rounded-lg border border-gray-200 px-3 py-2 text-sm outline-none focus:border-[#7290ef]" placeholder="What should someone decide differently after reading the findings?" /></label>
                  <label className="block"><span className="mb-1 block text-[11px] font-semibold text-gray-600">Target participant</span><input value={draft.researchSpec.population.participant_role} onChange={(event) => patchSpec({ population: { ...draft.researchSpec.population, participant_role: event.target.value } })} className="w-full rounded-lg border border-gray-200 px-3 py-2 text-sm outline-none focus:border-[#7290ef]" placeholder="Parents of children aged 10–17" /></label>
                  <label className="block"><span className="mb-1 block text-[11px] font-semibold text-gray-600">What they know about</span><input value={draft.researchSpec.population.subject} onChange={(event) => patchSpec({ population: { ...draft.researchSpec.population, subject: event.target.value } })} className="w-full rounded-lg border border-gray-200 px-3 py-2 text-sm outline-none focus:border-[#7290ef]" placeholder="Their child’s use of generative AI" /></label>
                  <label className="block md:col-span-2"><span className="mb-1 block text-[11px] font-semibold text-gray-600">Research questions <span className="font-normal text-gray-400">— one per line, not asked verbatim</span></span><textarea value={joinLines(draft.researchSpec.research_questions)} onChange={(event) => patchSpec({ research_questions: splitLines(event.target.value) })} rows={3} className="w-full rounded-lg border border-gray-200 px-3 py-2 text-sm outline-none focus:border-[#7290ef]" /></label>
                </div>
                <details className="mt-4 border-t border-gray-100 pt-3"><summary className="cursor-pointer text-xs font-medium text-[#2f5bea]">Population boundaries and participant terms</summary><div className="mt-3 grid gap-3 md:grid-cols-2">
                  <label className="block"><span className="mb-1 block text-[11px] text-gray-500">Relationship to topic</span><textarea value={draft.researchSpec.population.knowledge_relationship} onChange={(event) => patchSpec({ population: { ...draft.researchSpec.population, knowledge_relationship: event.target.value } })} rows={3} className="w-full rounded-lg border border-gray-200 p-2 text-xs" /></label>
                  <label className="block"><span className="mb-1 block text-[11px] text-gray-500">Knowledge limits — one per line</span><textarea value={joinLines(draft.researchSpec.population.knowledge_limits)} onChange={(event) => patchSpec({ population: { ...draft.researchSpec.population, knowledge_limits: splitLines(event.target.value) } })} rows={3} className="w-full rounded-lg border border-gray-200 p-2 text-xs" /></label>
                  <label className="block md:col-span-2"><span className="mb-1 block text-[11px] text-gray-500">Participant-facing scope definition</span><textarea value={draft.researchSpec.clarification_policy.participant_definition} onChange={(event) => patchSpec({ clarification_policy: { ...draft.researchSpec.clarification_policy, participant_definition: event.target.value } })} rows={2} className="w-full rounded-lg border border-gray-200 p-2 text-xs" placeholder="A short neutral definition used only when a participant asks what the study means." /></label>
                  <label className="block"><span className="mb-1 block text-[11px] text-gray-500">In scope — one per line</span><textarea value={joinLines(draft.researchSpec.clarification_policy.in_scope)} onChange={(event) => patchSpec({ clarification_policy: { ...draft.researchSpec.clarification_policy, in_scope: splitLines(event.target.value) } })} rows={3} className="w-full rounded-lg border border-gray-200 p-2 text-xs" /></label>
                  <label className="block"><span className="mb-1 block text-[11px] text-gray-500">Out of scope — one per line</span><textarea value={joinLines(draft.researchSpec.clarification_policy.out_of_scope)} onChange={(event) => patchSpec({ clarification_policy: { ...draft.researchSpec.clarification_policy, out_of_scope: splitLines(event.target.value) } })} rows={3} className="w-full rounded-lg border border-gray-200 p-2 text-xs" /></label>
                  <label className="block"><span className="mb-1 block text-[11px] text-gray-500">Ambiguous cases — one per line</span><textarea value={joinLines(draft.researchSpec.clarification_policy.ambiguous_cases)} onChange={(event) => patchSpec({ clarification_policy: { ...draft.researchSpec.clarification_policy, ambiguous_cases: splitLines(event.target.value) } })} rows={3} className="w-full rounded-lg border border-gray-200 p-2 text-xs" /></label>
                  <label className="block"><span className="mb-1 block text-[11px] text-gray-500">Ambiguity handling</span><textarea value={draft.researchSpec.clarification_policy.ambiguity_handling} onChange={(event) => patchSpec({ clarification_policy: { ...draft.researchSpec.clarification_policy, ambiguity_handling: event.target.value } })} rows={3} className="w-full rounded-lg border border-gray-200 p-2 text-xs" /></label>
                  <label className="block"><span className="mb-1 block text-[11px] text-gray-500">Participant-facing purpose</span><textarea value={draft.researchSpec.participant_information.purpose_summary} onChange={(event) => patchSpec({ participant_information: { ...draft.researchSpec.participant_information, purpose_summary: event.target.value } })} rows={3} className="w-full rounded-lg border border-gray-200 p-2 text-xs" /></label>
                  <label className="block"><span className="mb-1 block text-[11px] text-gray-500">What participation involves</span><textarea value={draft.researchSpec.participant_information.participation_summary} onChange={(event) => patchSpec({ participant_information: { ...draft.researchSpec.participant_information, participation_summary: event.target.value } })} rows={3} className="w-full rounded-lg border border-gray-200 p-2 text-xs" placeholder="What they will do, response options, and approximate format." /></label>
                  <label className="block"><span className="mb-1 block text-[11px] text-gray-500">Privacy reminder</span><textarea value={draft.researchSpec.participant_information.privacy_reminder} onChange={(event) => patchSpec({ participant_information: { ...draft.researchSpec.participant_information, privacy_reminder: event.target.value } })} rows={3} className="w-full rounded-lg border border-gray-200 p-2 text-xs" /></label>
                  <label className="block md:col-span-2"><span className="mb-1 block text-[11px] text-gray-500">Data handling and protection <span className="font-normal text-gray-400">— shown to participants and available to the interviewer</span></span><textarea value={draft.researchSpec.participant_information.data_handling_summary} onChange={(event) => patchSpec({ participant_information: { ...draft.researchSpec.participant_information, data_handling_summary: event.target.value } })} rows={3} className="w-full rounded-lg border border-gray-200 p-2 text-xs" placeholder="Explain transcription, what is stored, who reviews it, and any protections or retention limits you can truthfully promise." /></label>
                  <label className="block"><span className="mb-1 block text-[11px] text-gray-500">Data controller</span><input value={draft.researchSpec.participant_information.data_controller} onChange={(event) => patchSpec({ participant_information: { ...draft.researchSpec.participant_information, data_controller: event.target.value } })} className="w-full rounded-lg border border-gray-200 p-2 text-xs" placeholder="Legal entity responsible for the study data" /></label>
                  <label className="block"><span className="mb-1 block text-[11px] text-gray-500">Who receives the data</span><textarea value={draft.researchSpec.participant_information.data_recipients_summary} onChange={(event) => patchSpec({ participant_information: { ...draft.researchSpec.participant_information, data_recipients_summary: event.target.value } })} rows={2} className="w-full rounded-lg border border-gray-200 p-2 text-xs" placeholder="Research team and service providers, including transcription processing." /></label>
                  <label className="block"><span className="mb-1 block text-[11px] text-gray-500">Risks or discomforts</span><textarea value={draft.researchSpec.participant_information.risks_summary} onChange={(event) => patchSpec({ participant_information: { ...draft.researchSpec.participant_information, risks_summary: event.target.value } })} rows={2} className="w-full rounded-lg border border-gray-200 p-2 text-xs" /></label>
                  <label className="block"><span className="mb-1 block text-[11px] text-gray-500">Benefits — optional</span><textarea value={draft.researchSpec.participant_information.benefits_summary} onChange={(event) => patchSpec({ participant_information: { ...draft.researchSpec.participant_information, benefits_summary: event.target.value } })} rows={2} className="w-full rounded-lg border border-gray-200 p-2 text-xs" /></label>
                  <label className="block md:col-span-2"><span className="mb-1 block text-[11px] text-gray-500">Compensation</span><textarea value={draft.researchSpec.participant_information.compensation_summary} onChange={(event) => patchSpec({ participant_information: { ...draft.researchSpec.participant_information, compensation_summary: event.target.value } })} rows={2} className="w-full rounded-lg border border-gray-200 p-2 text-xs" placeholder="Describe payment and how partial or early-finished interviews are handled." /></label>
                  <label className="block"><span className="mb-1 block text-[11px] text-gray-500">Researcher identity</span><input value={draft.researchSpec.participant_information.researcher_identity} onChange={(event) => patchSpec({ participant_information: { ...draft.researchSpec.participant_information, researcher_identity: event.target.value } })} className="w-full rounded-lg border border-gray-200 p-2 text-xs" placeholder="Prolific research team" /></label>
                  <label className="block"><span className="mb-1 block text-[11px] text-gray-500">Researcher contact</span><input value={draft.researchSpec.participant_information.researcher_contact} onChange={(event) => patchSpec({ participant_information: { ...draft.researchSpec.participant_information, researcher_contact: event.target.value } })} className="w-full rounded-lg border border-gray-200 p-2 text-xs" placeholder="Email or Prolific messaging instructions" /></label>
                  <label className="block"><span className="mb-1 block text-[11px] text-gray-500">Complaints or independent concerns</span><input value={draft.researchSpec.participant_information.complaints_contact} onChange={(event) => patchSpec({ participant_information: { ...draft.researchSpec.participant_information, complaints_contact: event.target.value } })} className="w-full rounded-lg border border-gray-200 p-2 text-xs" placeholder="A complaints route, distinct from study questions where possible" /></label>
                  <label className="block"><span className="mb-1 block text-[11px] text-gray-500">Ethics review reference — optional</span><input value={draft.researchSpec.participant_information.ethics_review_reference} onChange={(event) => patchSpec({ participant_information: { ...draft.researchSpec.participant_information, ethics_review_reference: event.target.value } })} className="w-full rounded-lg border border-gray-200 p-2 text-xs" placeholder="Only if approval or exemption is confirmed" /></label>
                  <label className="block"><span className="mb-1 block text-[11px] text-gray-500">Data retention period</span><input value={draft.researchSpec.participant_information.data_retention_period} onChange={(event) => patchSpec({ participant_information: { ...draft.researchSpec.participant_information, data_retention_period: event.target.value } })} className="w-full rounded-lg border border-gray-200 p-2 text-xs" placeholder="How long identifiable and de-identified data are kept" /></label>
                  <label className="block"><span className="mb-1 block text-[11px] text-gray-500">Privacy notice URL — optional</span><input type="url" value={draft.researchSpec.participant_information.privacy_notice_url} onChange={(event) => patchSpec({ participant_information: { ...draft.researchSpec.participant_information, privacy_notice_url: event.target.value } })} className="w-full rounded-lg border border-gray-200 p-2 text-xs" placeholder="https://…" /></label>
                  <label className="block md:col-span-2"><span className="mb-1 block text-[11px] text-gray-500">Withdrawal and deletion instructions</span><textarea value={draft.researchSpec.participant_information.withdrawal_instructions} onChange={(event) => patchSpec({ participant_information: { ...draft.researchSpec.participant_information, withdrawal_instructions: event.target.value } })} rows={2} className="w-full rounded-lg border border-gray-200 p-2 text-xs" placeholder="How and until when participants can request withdrawal or deletion." /></label>
                </div></details>
              </section>

              <section className="rounded-2xl border border-gray-200 bg-white p-5 shadow-sm">
                <div className="mb-4 flex items-center justify-between"><div><h2 className="text-base font-semibold text-gray-900">Interview guide</h2><p className="mt-1 text-xs text-gray-400">Choose one coherent orchestration policy; card-level anchors and dependencies remain server-enforced.</p></div><button onClick={() => { const questions = [...draft.guide.questions, emptyQuestion(draft.guide.questions.length)]; commit({ ...draft, guide: { ...draft.guide, questions } }); setSelectedQuestion(questions.length - 1); }} className="rounded-lg border border-gray-200 px-3 py-1.5 text-xs font-medium text-gray-700 hover:bg-gray-50">+ Question</button></div>
                <label className="mb-4 block rounded-xl border border-blue-100 bg-blue-50/60 px-4 py-3"><span className="block text-xs font-semibold text-gray-800">Interview behavior</span><select aria-label="Interview behavior" value={resolvedProfile(draft.guide)} onChange={(event) => commit({ ...draft, guide: withProfile(draft.guide, event.target.value as OrchestrationProfile) })} className="mt-2 w-full rounded-lg border border-blue-200 bg-white px-3 py-2 text-sm text-gray-800"><option value="fixed">Fixed — exact questions, no skipping</option><option value="repair_only">Quality control — skip covered cards, one narrow repair</option><option value="semi_structured">Semi-structured — authored probes only</option><option value="exploratory">Exploratory — authored and bounded emergent probes</option></select><span className="mt-2 block text-[11px] leading-relaxed text-gray-500">Quality control is the safest standardized option. Semi-structured is the default for qualitative interviews; exploratory should be chosen deliberately.</span></label>
                <div className="space-y-3">
                  {draft.guide.questions.map((question, index) => <article key={question.id || index} onClick={() => { setSelectedQuestion(index); if (question.visit_scope === "first_time" || question.visit_scope === "returning") setPreviewVisit(question.visit_scope); }} className={`rounded-xl border bg-white p-4 transition ${selectedQuestion === index ? "border-[#7290ef] ring-2 ring-[#eef1ff]" : "border-gray-200 hover:border-gray-300"}`}>
                    <div className="mb-2 flex items-center gap-2"><div className="flex flex-col text-[9px] leading-none text-gray-300"><button type="button" disabled={index === 0} onClick={(event) => { event.stopPropagation(); moveQuestion(index, index - 1); }} className="disabled:opacity-20" title="Move up">▲</button><button type="button" disabled={index === draft.guide.questions.length - 1} onClick={(event) => { event.stopPropagation(); moveQuestion(index, index + 1); }} className="disabled:opacity-20" title="Move down">▼</button></div><span className="flex h-6 w-6 items-center justify-center rounded-full bg-gray-100 text-[10px] font-semibold text-gray-600">{index + 1}</span><input value={question.module || ""} onChange={(event) => patchQuestion(index, { module: event.target.value })} className="min-w-0 flex-1 border-0 bg-transparent text-[10px] font-semibold uppercase tracking-wide text-gray-400 outline-none" placeholder="MODULE" /><button type="button" onClick={(event) => { event.stopPropagation(); openQuestionComposer(index); }} className="flex shrink-0 items-center gap-1 rounded-md border border-[#cbd5ff] bg-[#f5f7ff] px-2 py-1 text-[10px] font-semibold text-[#2f5bea] hover:border-[#9bb0f4] hover:bg-[#eef1ff]" aria-label={`Edit Question ${index + 1} with AI`}><span aria-hidden="true">✦</span> AI edit</button>{draft.guide.allow_follow_ups !== false && <select value={question.max_follow_ups} onChange={(event) => patchQuestion(index, { max_follow_ups: Number(event.target.value) })} className="rounded-md border border-gray-200 bg-white px-2 py-1 text-[10px] text-gray-600"><option value={0}>No follow-up</option><option value={1}>Short follow-up</option><option value={2}>Up to 2 probes</option><option value={3}>Up to 3 probes</option></select>}</div>
                    <div className="mb-2 flex flex-wrap items-center gap-2 pl-14"><select value={question.phase || "core"} onChange={(event) => patchQuestion(index, { phase: event.target.value as InterviewQuestionConfig["phase"] })} className={`rounded-full border-0 px-2 py-1 text-[10px] font-semibold ${question.phase === "warm_up" ? "bg-amber-50 text-amber-700" : "bg-gray-100 text-gray-600"}`}><option value="warm_up">Warm-up</option><option value="core">Core</option><option value="closing">Closing</option></select><select value={question.visit_scope || "all"} onChange={(event) => patchQuestion(index, { visit_scope: event.target.value as InterviewQuestionConfig["visit_scope"] })} className="rounded-full border-0 bg-blue-50 px-2 py-1 text-[10px] font-semibold text-blue-700"><option value="all">Every interview</option><option value="first_time">First time only</option><option value="returning">Returning only</option></select>{question.phase === "warm_up" && <span className="text-[10px] text-gray-400">Easy, descriptive, non-identifying</span>}</div>
                    <label className="mb-3 ml-14 flex items-start gap-2 text-[10px] text-gray-500"><input type="checkbox" className="mt-0.5" checked={question.skip_remaining_module_if_not_applicable === true} onChange={(event) => patchQuestion(index, { skip_remaining_module_if_not_applicable: event.target.checked })} /><span><span className="font-semibold text-gray-600">Let the interviewer skip the rest of this module when no answer applies</span><span className="block text-gray-400">The interview model interprets the response; the server limits the skip to dependent questions in this module.</span></span></label>
                    {draft.guide.skip_answered_questions === true && <label className="mb-3 ml-14 flex items-start gap-2 text-[10px] text-gray-500"><input type="checkbox" className="mt-0.5" checked={question.always_ask === true} onChange={(event) => patchQuestion(index, { always_ask: event.target.checked })} /><span><span className="font-semibold text-gray-600">Always ask this question</span><span className="block text-gray-400">Protect a task or module anchor even if the participant supplied its evidence earlier.</span></span></label>}
                    <textarea value={question.text} onChange={(event) => patchQuestion(index, { text: event.target.value })} rows={2} className="w-full resize-none border-0 p-0 text-[15px] font-medium leading-relaxed text-gray-900 outline-none focus:ring-0" placeholder="Ask one neutral, open question…" />
                    <div className="mt-3 grid gap-3 md:grid-cols-2"><label><span className="mb-1 block text-[10px] font-semibold text-gray-400">PURPOSE</span><textarea value={question.purpose || ""} onChange={(event) => patchQuestion(index, { purpose: event.target.value })} rows={2} className="w-full rounded-lg border border-gray-100 bg-gray-50 p-2 text-xs outline-none focus:border-[#9bb0f4]" /></label><label><span className="mb-1 block text-[10px] font-semibold text-gray-400">EVIDENCE TO LISTEN FOR — ONE PER LINE</span><textarea value={joinLines(question.evidence_goal)} onChange={(event) => patchQuestion(index, { evidence_goal: splitLines(event.target.value) })} rows={2} className="w-full rounded-lg border border-gray-100 bg-gray-50 p-2 text-xs outline-none focus:border-[#9bb0f4]" /></label></div>
                    <details className="mt-3"><summary className="cursor-pointer text-[11px] font-medium text-[#2f5bea]">Probes, guardrails, and role-specific wording</summary><div className="mt-2 grid gap-3 md:grid-cols-2"><label><span className="mb-1 block text-[10px] text-gray-400">OPTIONAL PROBES — ONE PER LINE</span><textarea value={joinLines(question.probes)} onChange={(event) => patchQuestion(index, { probes: splitLines(event.target.value) })} rows={3} className="w-full rounded-lg border border-gray-200 p-2 text-xs" /></label><label><span className="mb-1 block text-[10px] text-gray-400">DO NOT ASSUME — ONE PER LINE</span><textarea value={joinLines(question.do_not_assume)} onChange={(event) => patchQuestion(index, { do_not_assume: splitLines(event.target.value) })} rows={3} className="w-full rounded-lg border border-gray-200 p-2 text-xs" /></label><label><span className="mb-1 block text-[10px] text-gray-400">DEPENDS ON QUESTION IDS — ONE PER LINE</span><textarea value={joinLines(question.depends_on)} onChange={(event) => patchQuestion(index, { depends_on: splitLines(event.target.value).map((value) => value.trim()).filter(Boolean) })} rows={2} className="w-full rounded-lg border border-gray-200 p-2 font-mono text-xs" placeholder="consequential_task" /></label><label><span className="mb-1 block text-[10px] text-gray-400">PARENT WORDING — OPTIONAL</span><textarea value={roleText(question, "parent")} onChange={(event) => patchRoleQuestion(index, "parent", event.target.value)} rows={2} className="w-full rounded-lg border border-gray-200 p-2 text-xs" placeholder="Uses ‘your child’ where appropriate" /></label><label><span className="mb-1 block text-[10px] text-gray-400">EDUCATOR WORDING — OPTIONAL</span><textarea value={roleText(question, "educator")} onChange={(event) => patchRoleQuestion(index, "educator", event.target.value)} rows={2} className="w-full rounded-lg border border-gray-200 p-2 text-xs" placeholder="Uses ‘your students’ where appropriate" /></label></div></details>
                    <div className="mt-3 flex items-center justify-end gap-3 border-t border-gray-100 pt-2"><button onClick={(event) => { event.stopPropagation(); const copy = clone(question); copy.id = uniqueQuestionId(draft.guide.questions, question.id || "question"); const questions = [...draft.guide.questions]; questions.splice(index + 1, 0, copy); commit({ ...draft, guide: { ...draft.guide, questions } }); }} className="text-[11px] text-gray-500 hover:text-gray-900">Duplicate</button><button disabled={draft.guide.questions.length === 1} onClick={(event) => { event.stopPropagation(); const questions = draft.guide.questions.filter((_, i) => i !== index); commit({ ...draft, guide: { ...draft.guide, questions } }); setSelectedQuestion(Math.max(0, index - 1)); }} className="text-[11px] text-red-500 disabled:opacity-30">Delete</button></div>
                  </article>)}
                </div>
                <label className="mt-4 block"><span className="mb-1 block text-[11px] font-semibold text-gray-600">Closing prompt</span><textarea value={draft.guide.closing_message || ""} onChange={(event) => commit({ ...draft, guide: { ...draft.guide, closing_message: event.target.value } })} rows={2} className="w-full rounded-lg border border-gray-200 p-3 text-sm" /></label>
              </section>
            </div>
          </main>

          <aside className="border-l border-gray-200 bg-white p-4">
            <div className="mb-3 flex items-center justify-between"><h2 className="text-sm font-semibold text-gray-900">Participant preview</h2><span className="rounded-full bg-gray-100 px-2 py-1 text-[10px] text-gray-500">Text response</span></div>
            <div className="mb-3 grid grid-cols-2 gap-2"><select aria-label="Preview participant role" value={previewRole} onChange={(event) => setPreviewRole(event.target.value as PreviewRole)} className="rounded-lg border border-gray-200 bg-white px-2 py-1.5 text-[10px] font-medium text-gray-600"><option value="parent">Parent</option><option value="educator">Educator</option></select><select aria-label="Preview interview visit" value={previewVisit} onChange={(event) => setPreviewVisit(event.target.value as PreviewVisit)} className="rounded-lg border border-gray-200 bg-white px-2 py-1.5 text-[10px] font-medium text-gray-600"><option value="first_time">First interview</option><option value="returning">Returning panelist</option></select></div>
            <div className="overflow-hidden rounded-2xl border border-gray-200 bg-[#fcfcfd] shadow-sm">
              <div className="h-1 bg-gray-100"><div className="h-full bg-[#2f5bea] transition-all" style={{ width: `${((previewQuestionIndex + 1) / Math.max(1, previewQuestions.length)) * 100}%` }} /></div>
              <div className="flex items-center justify-between px-4 py-3 text-[10px] text-gray-500"><span>English⌄</span><span>🔇 Audio off</span></div>
              <div className="flex min-h-[285px] flex-col items-center justify-center px-6 text-center"><p className="mb-3 text-[10px] font-semibold uppercase tracking-[0.18em] text-gray-400">Question {previewQuestionIndex + 1} of {previewQuestions.length}</p><p className="text-xl font-medium leading-snug text-gray-900">{previewQuestionText}</p></div>
              <div className="mx-3 mb-3 flex items-center justify-between rounded-xl border border-gray-200 bg-white px-3 py-2 text-[10px] text-gray-400 shadow-sm"><span>Write your response…</span><span className="flex h-7 w-7 items-center justify-center rounded-lg bg-[#2f5bea] text-sm text-white">↗</span></div>
              <div className="border-t border-gray-100 p-3"><a href={interviewUrl} target="_blank" rel="noreferrer" className="block rounded-lg bg-gray-900 px-3 py-2 text-center text-xs font-semibold text-white">Run test interview</a></div>
            </div>
            <div className="mt-4 rounded-2xl border border-gray-200 p-4"><div className="flex items-center justify-between"><h3 className="text-xs font-semibold text-gray-800">Field estimate</h3><span className="text-xs font-semibold text-[#2f5bea]">~{draft.estimatedMinutes} min</span></div><input type="range" min={5} max={60} value={draft.estimatedMinutes} onChange={(event) => patchDraft({ estimatedMinutes: Number(event.target.value) })} className="mt-3 w-full accent-[#2f5bea]" /><div className="mt-2 text-[10px] text-gray-400"><span>{firstTimeQuestionCount} first-time questions{firstTimeQuestionCount !== returningQuestionCount ? ` · ${returningQuestionCount} returning` : ""}</span><span className="mt-1 block truncate">{draft.researchSpec.population.participant_role || "Audience not set"}</span></div></div>
            <div className="mt-4 rounded-2xl border border-gray-200 p-4"><h3 className="text-xs font-semibold text-gray-800">Guide review</h3><div className="mt-3 space-y-2">{checks.slice(0, 8).map((check, index) => <div key={`${check.text}-${index}`} className="flex gap-2 text-[11px] leading-relaxed"><span className={check.tone === "good" ? "text-emerald-500" : "text-amber-500"}>{check.tone === "good" ? "✓" : "!"}</span><span className="text-gray-600">{check.text}</span></div>)}</div></div>
          </aside>
        </div>
      ) : (
        <main className="mx-auto max-w-5xl p-6 lg:p-10">
          <div className="mb-7"><h2 className="text-2xl font-semibold text-gray-900">Recruit and launch</h2><p className="mt-1 text-sm text-gray-500">Connect the guide to your Prolific sample, soft-launch, and monitor completes.</p></div>
          <div className="grid gap-5 lg:grid-cols-[1fr_340px]">
            <div className="space-y-5">
              <section className="rounded-2xl border border-gray-200 bg-white p-5 shadow-sm"><div className="mb-4 flex items-center justify-between"><div><h3 className="text-sm font-semibold text-gray-900">Recruitment source</h3><p className="mt-1 text-xs text-gray-500">OASIS accepts Prolific IDs and participant metadata while Prolific handles sampling and rewards.</p></div><span className="rounded-full bg-[#e9efff] px-2.5 py-1 text-[10px] font-semibold text-[#2f5bea]">Prolific-ready</span></div><div className="grid gap-3 sm:grid-cols-3">{(["prolific", "direct_link", "mixed"] as const).map((source) => <button key={source} onClick={() => patchSpec({ fielding: { ...draft.researchSpec.fielding, recruitment_source: source } })} className={`rounded-xl border p-3 text-left ${draft.researchSpec.fielding.recruitment_source === source ? "border-[#7290ef] bg-[#f5f7ff]" : "border-gray-200"}`}><span className="block text-xs font-semibold text-gray-800">{source === "prolific" ? "Prolific" : source === "direct_link" ? "Direct link" : "Mixed sources"}</span><span className="mt-1 block text-[10px] leading-relaxed text-gray-500">{source === "prolific" ? "Bring your existing audience and metadata." : source === "direct_link" ? "Anyone with the link can join." : "Combine panel and direct recruits."}</span></button>)}</div></section>
              <section className="rounded-2xl border border-gray-200 bg-white p-5 shadow-sm"><h3 className="text-sm font-semibold text-gray-900">Fielding plan</h3><div className="mt-4 grid gap-4 sm:grid-cols-2"><label><span className="mb-1 block text-[11px] font-medium text-gray-600">Target completes</span><input type="number" min={1} value={draft.researchSpec.fielding.response_limit} onChange={(event) => patchSpec({ fielding: { ...draft.researchSpec.fielding, response_limit: Number(event.target.value) } })} className="w-full rounded-lg border border-gray-200 px-3 py-2 text-sm" /></label><label><span className="mb-1 block text-[11px] font-medium text-gray-600">Soft-launch review after</span><input type="number" min={1} value={draft.researchSpec.fielding.soft_launch_limit || ""} onChange={(event) => patchSpec({ fielding: { ...draft.researchSpec.fielding, soft_launch_limit: event.target.value ? Number(event.target.value) : null } })} className="w-full rounded-lg border border-gray-200 px-3 py-2 text-sm" /></label><label><span className="mb-1 block text-[11px] font-medium text-gray-600">External participant parameter</span><input value={draft.researchSpec.fielding.participant_id_parameter} onChange={(event) => patchSpec({ fielding: { ...draft.researchSpec.fielding, participant_id_parameter: event.target.value.replace(/[^a-zA-Z0-9_-]/g, "") } })} className="w-full rounded-lg border border-gray-200 px-3 py-2 font-mono text-xs" /></label><label className="flex items-end gap-2 pb-2 text-xs text-gray-600"><input type="checkbox" checked={draft.researchSpec.fielding.require_profile} onChange={(event) => patchSpec({ fielding: { ...draft.researchSpec.fielding, require_profile: event.target.checked } })} /> Require participant role/profile metadata</label></div><p className="mt-3 text-[10px] text-gray-400">OASIS flags this milestone in recruitment progress; pause the study from this page after reviewing pilot transcripts.</p></section>
              {draft.researchSpec.fielding.recruitment_source === "prolific" && <section className="rounded-2xl border border-gray-200 bg-white p-5 shadow-sm"><h3 className="text-sm font-semibold text-gray-900">Prolific completion paths</h3><p className="mt-1 text-xs leading-relaxed text-gray-500">Create these paths in the Prolific study draft, then paste the generated redirect URLs here. Redirects occur only after OASIS confirms the session was saved.</p><div className="mt-4 space-y-3"><label className="block"><span className="mb-1 block text-[11px] font-medium text-gray-600">Completed interview URL</span><input type="url" value={draft.researchSpec.fielding.completion_url} onChange={(event) => patchSpec({ fielding: { ...draft.researchSpec.fielding, completion_url: event.target.value } })} className="w-full rounded-lg border border-gray-200 px-3 py-2 font-mono text-xs" placeholder="https://app.prolific.com/submissions/complete?cc=…" /></label><label className="block"><span className="mb-1 block text-[11px] font-medium text-gray-600">No-consent URL</span><input type="url" value={draft.researchSpec.fielding.no_consent_url} onChange={(event) => patchSpec({ fielding: { ...draft.researchSpec.fielding, no_consent_url: event.target.value } })} className="w-full rounded-lg border border-gray-200 px-3 py-2 font-mono text-xs" placeholder="Request-return completion path" /></label><label className="block"><span className="mb-1 block text-[11px] font-medium text-gray-600">Inactive or disconnected URL</span><input type="url" value={draft.researchSpec.fielding.inactivity_url} onChange={(event) => patchSpec({ fielding: { ...draft.researchSpec.fielding, inactivity_url: event.target.value } })} className="w-full rounded-lg border border-gray-200 px-3 py-2 font-mono text-xs" placeholder="Partial/incomplete completion path" /></label></div></section>}
              <section className="rounded-2xl border border-gray-200 bg-white p-5 shadow-sm"><div className="flex items-center justify-between"><div><h3 className="text-sm font-semibold text-gray-900">Participant profiles</h3><p className="mt-1 text-xs text-gray-500">One row per participant: <span className="font-mono">ID, role, optional label</span>. Role-aware wording uses this profile.</p></div><span className="text-xs font-semibold text-gray-600">{participantList.length} loaded</span></div><textarea value={participantRows} onChange={(event) => setParticipantRows(event.target.value)} rows={6} placeholder={"participant-001,parent,Parent of 13-year-old\nparticipant-002,educator,Secondary teacher"} className="mt-4 w-full rounded-xl border border-gray-200 p-3 font-mono text-xs leading-relaxed outline-none focus:border-[#7290ef]" /><div className="mt-3 flex items-center justify-between"><p className="text-[10px] text-gray-400">Only allowlisted profile fields are included in runtime prompts.</p><button onClick={importProfiles} disabled={importing || !participantRows.trim()} className="rounded-lg border border-gray-200 px-3 py-1.5 text-xs font-medium text-gray-700 disabled:opacity-40">{importing ? "Importing…" : "Import profiles"}</button></div></section>
              <section className="rounded-2xl border border-gray-200 bg-white p-5 shadow-sm"><h3 className="text-sm font-semibold text-gray-900">Participant link</h3><div className="mt-3 flex items-center gap-2 rounded-xl border border-gray-200 bg-gray-50 p-2"><code className="min-w-0 flex-1 truncate px-2 text-xs text-gray-600">{prolificInterviewUrl}</code><button onClick={() => navigator.clipboard.writeText(prolificInterviewUrl)} className="rounded-lg bg-white px-3 py-1.5 text-xs font-medium text-gray-700 shadow-sm">Copy</button></div><p className="mt-2 text-[10px] text-gray-400">Use this as the Prolific study URL. OASIS stores the participant, study, and submission IDs with the transcript.</p></section>
            </div>
            <aside className="space-y-5"><section className="rounded-2xl border border-gray-200 bg-white p-5 shadow-sm"><div className="flex items-center justify-between"><h3 className="text-sm font-semibold text-gray-900">Current recruitment</h3><span className="text-xs text-gray-500">{completed}/{responseLimit}</span></div><div className="mt-4 h-2 overflow-hidden rounded-full bg-gray-100"><div className="h-full rounded-full bg-[#2f5bea]" style={{ width: `${responseProgress}%` }} /></div><dl className="mt-5 grid grid-cols-2 gap-3 text-center"><div className="rounded-xl bg-gray-50 p-3"><dt className="text-[10px] text-gray-400">Completes</dt><dd className="mt-1 text-xl font-semibold text-gray-900">{completed}</dd></div><div className="rounded-xl bg-gray-50 p-3"><dt className="text-[10px] text-gray-400">In progress</dt><dd className="mt-1 text-xl font-semibold text-gray-900">{analytics?.active_sessions || 0}</dd></div></dl></section><section className="rounded-2xl border border-gray-200 bg-white p-5 shadow-sm"><h3 className="text-sm font-semibold text-gray-900">Launch checklist</h3><div className="mt-4 space-y-2.5">{[{ ok: draft.guide.questions.length >= 3, text: "Interview guide drafted" }, { ok: checks.every((check) => check.tone === "good"), text: "Guide review has no warnings" }, { ok: !!draft.researchSpec.population.participant_role, text: "Audience role defined" }, { ok: !!draft.researchSpec.participant_information.purpose_summary && !!draft.researchSpec.participant_information.participation_summary, text: "Participant information ready" }, { ok: !!draft.researchSpec.participant_information.data_handling_summary.trim() && !!draft.researchSpec.participant_information.data_controller.trim() && !!draft.researchSpec.participant_information.data_recipients_summary.trim(), text: "Data handling explained" }, { ok: !!draft.researchSpec.participant_information.researcher_identity.trim() && !!draft.researchSpec.participant_information.researcher_contact.trim() && !!draft.researchSpec.participant_information.complaints_contact.trim() && !!draft.researchSpec.participant_information.risks_summary.trim() && !!draft.researchSpec.participant_information.compensation_summary.trim() && !!draft.researchSpec.participant_information.data_retention_period.trim() && !!draft.researchSpec.participant_information.withdrawal_instructions.trim(), text: "Consent and privacy terms complete" }, { ok: draft.researchSpec.fielding.recruitment_source !== "prolific" || (!!draft.researchSpec.fielding.completion_url.trim() && !!draft.researchSpec.fielding.no_consent_url.trim() && !!draft.researchSpec.fielding.inactivity_url.trim()), text: "Prolific completion paths connected" }].map((item) => <div key={item.text} className="flex items-center gap-2 text-xs"><span className={`flex h-5 w-5 items-center justify-center rounded-full ${item.ok ? "bg-emerald-50 text-emerald-600" : "bg-amber-50 text-amber-600"}`}>{item.ok ? "✓" : "!"}</span><span className="text-gray-600">{item.text}</span></div>)}</div><button onClick={() => setLaunchStatus(study.status === "active" ? "paused" : "active")} disabled={saving} className={`mt-5 w-full rounded-xl px-4 py-3 text-sm font-semibold text-white disabled:opacity-50 ${study.status === "active" ? "bg-amber-500 hover:bg-amber-600" : "bg-[#2f5bea] hover:bg-[#244bc8]"}`}>{study.status === "active" ? "Pause recruitment" : study.status === "paused" ? "Resume study" : "Launch study"}</button>{study.status !== "active" && <p className="mt-2 text-center text-[10px] text-gray-400">Launch saves the latest draft and activates the interviewer.</p>}</section></aside>
          </div>
        </main>
      )}

      {activeQuestionComposerCard && questionComposerIndex !== null && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-gray-950/35 p-4 backdrop-blur-[2px]" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget && !questionComposing) setQuestionComposerIndex(null); }}>
          <section role="dialog" aria-modal="true" aria-labelledby="question-composer-title" className="flex max-h-[min(760px,92vh)] w-full max-w-xl flex-col overflow-hidden rounded-2xl border border-gray-200 bg-white shadow-2xl">
            <header className="flex items-start justify-between gap-4 border-b border-gray-100 px-5 py-4">
              <div className="min-w-0">
                <div className="mb-1 flex items-center gap-2"><span className="flex h-7 w-7 items-center justify-center rounded-lg bg-[#e9efff] text-sm text-[#2f5bea]">✦</span><h2 id="question-composer-title" className="text-sm font-semibold text-gray-900">Edit Question {questionComposerIndex + 1} with AI</h2></div>
                <p className="truncate pl-9 text-xs text-gray-500">{activeQuestionComposerCard.text}</p>
              </div>
              <button type="button" onClick={() => setQuestionComposerIndex(null)} disabled={questionComposing} className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-lg text-gray-400 hover:bg-gray-100 hover:text-gray-700 disabled:opacity-30" aria-label="Close question editor">×</button>
            </header>

            <div className="flex-1 overflow-y-auto px-5 py-4">
              <div className="space-y-3" aria-live="polite">
                {activeQuestionComposerMessages.map((message, index) => <div key={index} className={`max-w-[88%] rounded-xl px-3 py-2.5 text-xs leading-relaxed ${message.role === "researcher" ? "ml-auto bg-[#2f5bea] text-white" : "bg-gray-100 text-gray-700"}`}>{message.text}</div>)}
                {questionComposing && <div className="max-w-[88%] rounded-xl bg-gray-100 px-3 py-2.5 text-xs text-gray-500">Revising only this question…</div>}
              </div>
              {activeQuestionComposerMessages.length === 1 && <div className="mt-4 flex flex-wrap gap-1.5">
                {["Make the wording more natural", "Align the subdetails", "Find assumptions", "Improve the optional probes"].map((suggestion) => <button type="button" key={suggestion} onClick={() => runQuestionComposer(suggestion)} disabled={questionComposing} className="rounded-full border border-gray-200 px-2.5 py-1 text-[10px] text-gray-600 hover:border-[#9bb0f4] hover:text-[#2f5bea] disabled:opacity-40">{suggestion}</button>)}
              </div>}
            </div>

            <form onSubmit={(event) => { event.preventDefault(); runQuestionComposer(); }} className="border-t border-gray-100 bg-gray-50/80 p-4">
              {questionComposerError && <p className="mb-2 text-[11px] text-red-600">{questionComposerError}</p>}
              <textarea autoFocus value={questionComposerInput} onChange={(event) => setQuestionComposerInput(event.target.value)} rows={3} placeholder="Tell me what to change on this card…" className="w-full resize-none rounded-xl border border-gray-200 bg-white p-3 text-sm leading-relaxed outline-none focus:border-[#7290ef] focus:ring-2 focus:ring-[#e7ecff]" />
              <div className="mt-2 flex items-center justify-between gap-3">
                <p className="text-[10px] text-gray-400">Only Question {questionComposerIndex + 1} can change. You can undo the result.</p>
                <button disabled={questionComposing || !questionComposerInput.trim()} className="shrink-0 rounded-lg bg-gray-900 px-4 py-2 text-xs font-semibold text-white disabled:opacity-35">{questionComposing ? "Revising…" : "Revise card"}</button>
              </div>
            </form>
          </section>
        </div>
      )}
    </div>
  );
}
