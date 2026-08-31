import { useEffect, useMemo, useState } from "react";
import {
  simulationLab,
  type BenchmarkCase,
  type Candidate,
  type EmpiricalResults,
  type EmpiricalRow,
  type EmpiricalStudySummary,
  type JudgeValidationResults,
  type Overview,
  type RunResult,
  type Scores,
  type Technique,
} from "../lib/simulationLabApi";

const ARM_NAMES: Record<string, string> = {
  context_only: "Context only",
  survey_card: "Survey card",
  topic_card: "Topic card",
  narrative_persona: "Narrative persona",
  behavioral_profile: "Behavioral profile",
  state_aware: "State-aware",
  grounded_world_model: "Grounded world model",
  grounded_backstory: "Grounded backstory",
  retrospective_world_model: "Retrospective world model",
};

const SCORE_LABELS: { key: keyof Scores; label: string; hint: string }[] = [
  { key: "composite", label: "Composite", hint: "Weighted diagnostic summary" },
  { key: "behavior_match", label: "Behavior", hint: "Question, hedging, disclosure, contraction, and brevity features" },
  { key: "style_match", label: "Style", hint: "Behavioral features plus word-use similarity" },
  { key: "phrase_similarity", label: "Phrasing", hint: "Character-sequence similarity" },
  { key: "token_overlap", label: "Words", hint: "Shared unique words (Jaccard)" },
  { key: "length_match", label: "Length", hint: "Shorter-to-longer token ratio" },
];

function percent(value: number) { return `${Math.round(value * 100)}%`; }

function ScorePill({ value, label }: { value: number; label: string }) {
  const palette = value >= 0.75 ? "bg-emerald-50 text-emerald-700 ring-emerald-200" : value >= 0.5 ? "bg-amber-50 text-amber-700 ring-amber-200" : "bg-slate-50 text-slate-600 ring-slate-200";
  return <span className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-semibold ring-1 ring-inset ${palette}`}><span className="font-normal opacity-70">{label}</span>{percent(value)}</span>;
}

function MetricBar({ value, label, hint }: { value: number; label: string; hint: string }) {
  return (
    <div title={hint} className="grid grid-cols-[72px_1fr_38px] items-center gap-2 text-xs">
      <span className="text-gray-500">{label}</span>
      <div className="h-1.5 overflow-hidden rounded-full bg-gray-100"><div className="h-full rounded-full bg-oasis-500" style={{ width: percent(value) }} /></div>
      <span className="text-right font-medium tabular-nums text-gray-700">{percent(value)}</span>
    </div>
  );
}

function Transcript({ item, contextMessages }: { item: BenchmarkCase; contextMessages: number }) {
  const visible = item.transcript.slice(-contextMessages);
  const hiddenCount = item.transcript.length - visible.length;
  return (
    <div className="space-y-3">
      {hiddenCount > 0 && <div className="text-center text-[11px] font-medium uppercase tracking-wider text-gray-400">{hiddenCount} earlier messages hidden</div>}
      {visible.map((turn, index) => (
        <div key={`${turn.role}-${index}`} className={`flex ${turn.role === "client" ? "justify-end" : "justify-start"}`}>
          <div className={`max-w-[88%] rounded-2xl px-3.5 py-2.5 text-sm leading-relaxed ${turn.role === "client" ? "rounded-br-md bg-oasis-500 text-white" : "rounded-bl-md border border-gray-200 bg-white text-gray-700"}`}>
            <div className={`mb-1 text-[10px] font-semibold uppercase tracking-wider ${turn.role === "client" ? "text-white/65" : "text-gray-400"}`}>{turn.role}</div>
            {turn.text}
          </div>
        </div>
      ))}
      <div className="flex items-center gap-2 py-1 text-xs text-gray-400"><div className="h-px flex-1 bg-gray-200" /><span>N + 1 predicted below</span><div className="h-px flex-1 bg-gray-200" /></div>
    </div>
  );
}

function CandidateCard({ candidate, technique, onPrompt }: { candidate: Candidate; technique: Technique; onPrompt: () => void }) {
  return (
    <div className="rounded-2xl border border-gray-200 bg-white p-4 shadow-sm">
      <div className="mb-3 flex items-start justify-between gap-3">
        <div className="flex items-center gap-2"><span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: technique.color }} /><div><div className="text-sm font-semibold text-gray-900">{technique.short_name}</div><div className="text-[11px] text-gray-400">Synthetic continuation</div></div></div>
        <ScorePill value={candidate.scores.composite} label="Match" />
      </div>
      <p className="min-h-[64px] text-sm leading-relaxed text-gray-700">“{candidate.response}”</p>
      <div className="mt-4 space-y-2 border-t border-gray-100 pt-3">
        {SCORE_LABELS.slice(1).map(metric => <MetricBar key={metric.key} value={candidate.scores[metric.key]} label={metric.label} hint={metric.hint} />)}
      </div>
      <button onClick={onPrompt} className="mt-3 text-xs font-medium text-oasis-600 hover:text-oasis-700">Inspect conditioning prompt →</button>
    </div>
  );
}

function TechniqueRail({ techniques, result }: { techniques: Technique[]; result: RunResult | null }) {
  return (
    <div className="grid gap-3 lg:grid-cols-5">
      {techniques.map((technique, index) => {
        const summary = result?.summaries.find(item => item.technique_id === technique.id);
        return (
          <div key={technique.id} className="relative rounded-2xl border border-gray-200 bg-white p-4">
            {index < techniques.length - 1 && <div className="absolute -right-3 top-7 z-10 hidden h-px w-3 bg-gray-300 lg:block" />}
            <div className="mb-2 flex items-center justify-between"><span className="flex h-6 w-6 items-center justify-center rounded-full text-[11px] font-bold text-white" style={{ backgroundColor: technique.color }}>{index + 1}</span>{summary && <span className="text-sm font-bold tabular-nums text-gray-800">{percent(summary.mean_composite)}</span>}</div>
            <div className="text-sm font-semibold text-gray-900">{technique.short_name}</div>
            <div className="mt-1 text-xs leading-relaxed text-gray-500">{technique.adds.join(" · ")}</div>
          </div>
        );
      })}
    </div>
  );
}

function Interval({ mean, low, high, delta = false }: { mean: number; low: number; high: number; delta?: boolean }) {
  const value = delta ? `${mean >= 0 ? "+" : ""}${(mean * 100).toFixed(1)}` : (mean * 100).toFixed(1);
  return <div><span className="font-semibold tabular-nums text-gray-900">{value}</span><span className="ml-1 text-[10px] tabular-nums text-gray-400">[{(low * 100).toFixed(1)}, {(high * 100).toFixed(1)}]</span></div>;
}

function EmpiricalCandidate({ row }: { row: EmpiricalRow }) {
  return (
    <div className="rounded-xl border border-gray-200 bg-white p-3">
      <div className="mb-2 flex items-center justify-between"><span className="text-[10px] font-semibold uppercase tracking-wider text-gray-400">Sample {row.sample + 1}</span><span className="text-xs font-semibold text-oasis-700">Semantic {percent(row.scores.semantic_similarity)}</span></div>
      <p className="text-sm leading-relaxed text-gray-700">“{row.response}”</p>
      <div className="mt-3 flex flex-wrap gap-1.5"><ScorePill label="Meaning" value={row.scores.judge_meaning} /><ScorePill label="Fit" value={row.scores.judge_fit} /><ScorePill label="Style" value={row.scores.judge_style} />{row.scores.world_model_compatibility !== undefined && <ScorePill label="Persona fit" value={row.scores.world_model_compatibility} />}{row.scores.disclosure_calibration !== undefined && <ScorePill label="Disclosure" value={row.scores.disclosure_calibration} />}{row.scores.talk_type_match !== undefined && <ScorePill label="Talk type" value={row.scores.talk_type_match} />}{row.scores.two_sample_coverage !== undefined && <ScorePill label="2-sample coverage" value={row.scores.two_sample_coverage} />}{row.scores.sample_diversity !== undefined && <ScorePill label="Diversity" value={row.scores.sample_diversity} />}{row.scores.unsupported_detail > 0 && <span className="rounded-full bg-red-50 px-2 py-1 text-[10px] font-semibold text-red-700 ring-1 ring-inset ring-red-200">Unsupported detail</span>}{row.scores.persona_recitation > 0 && <span className="rounded-full bg-orange-50 px-2 py-1 text-[10px] font-semibold text-orange-700 ring-1 ring-inset ring-orange-200">Persona recitation</span>}</div>
    </div>
  );
}

function EmpiricalStudy({ study }: { study: EmpiricalResults }) {
  const test = study.summaries.test;
  const deltas = study.paired_differences_vs_context.test;
  const caseIds = [...new Set(study.test_rows.map(row => row.case_id))];
  const [caseId, setCaseId] = useState(caseIds[0]);
  useEffect(() => { setCaseId(caseIds[0] || ""); }, [study.study_id]);
  const rows = study.test_rows.filter(row => row.case_id === caseId);
  const first = rows[0];
  const selected = study.development_selected_arm;
  const selectedDelta = deltas[selected]?.semantic_similarity;
  const selectedName = ARM_NAMES[selected] || selected;
  const bestName = ARM_NAMES[study.test_descriptive_best_arm] || study.test_descriptive_best_arm;
  const followupArms = study.design.literature_followup_arms || [];
  const reliableGain = selectedDelta && selectedDelta.low > 0;
  const reliableLoss = selectedDelta && selectedDelta.high < 0;
  const conclusion = reliableGain
    ? `${selectedName} reliably improved held-out semantic similarity.`
    : reliableLoss
      ? `${selectedName} reliably underperformed conversation context alone.`
      : "The development-selected conditioning did not reliably beat context alone.";
  return (
    <div className="space-y-6 pb-16">
      <header>
        <div className="mb-2 flex flex-wrap items-center gap-2"><span className="rounded-full bg-emerald-50 px-2.5 py-1 text-[11px] font-semibold uppercase tracking-wider text-emerald-700">Completed empirical run</span><span className="text-xs text-gray-400">{new Date(study.created_at).toLocaleString()}</span></div>
        <h1 className="text-3xl font-bold tracking-tight text-gray-950">Held-out continuation study</h1>
        <p className="mt-2 max-w-4xl text-sm leading-relaxed text-gray-600">Observed client replies from {study.dataset.name}, compared with {study.test_rows.length} held-out model continuations. The independent unit is the {study.dataset.evaluation_unit || "participant-level conversation"}.</p>
      </header>

      <section className="rounded-2xl border border-gray-200 bg-white p-5 shadow-sm">
        <div className="grid gap-5 lg:grid-cols-[1.5fr_1fr]">
          <div><div className="text-xs font-semibold uppercase tracking-wider text-gray-500">Bottom line</div><h2 className="mt-2 text-xl font-bold text-gray-950">{conclusion}</h2><p className="mt-2 text-sm leading-relaxed text-gray-600">Development selected <strong>{selectedName}</strong>. Its held-out difference was {selectedDelta && <strong>{selectedDelta.mean >= 0 ? "+" : ""}{(selectedDelta.mean * 100).toFixed(1)} points</strong>}; the 95% interval was {selectedDelta && <>[{(selectedDelta.low * 100).toFixed(1)}, {(selectedDelta.high * 100).toFixed(1)}]</>}. <strong>{bestName}</strong> ranked highest descriptively on the test set.</p></div>
          <div className="grid grid-cols-2 gap-3">{[
            [study.dataset.eligible_participants, "Eligible participants"],
            [study.dataset.split_counts.test, "Held-out test"],
            [study.test_rows.length, "Test generations"],
            [study.design.bootstrap_iterations.toLocaleString(), "Bootstrap draws"],
          ].map(([value, label]) => <div key={label} className="rounded-xl bg-gray-50 p-3"><div className="text-xl font-bold tabular-nums text-gray-900">{value}</div><div className="mt-1 text-[11px] text-gray-500">{label}</div></div>)}</div>
        </div>
      </section>

      <section className="overflow-hidden rounded-2xl border border-gray-200 bg-white shadow-sm">
        <div className="border-b border-gray-200 p-5"><h2 className="font-bold text-gray-900">Test-set results</h2><p className="mt-1 text-xs text-gray-500">Means and participant-bootstrap 95% intervals. Δ columns are paired differences from context-only.</p>{study.design.followup_definition && <p className="mt-2 rounded-lg border border-cyan-200 bg-cyan-50 px-3 py-2 text-xs leading-relaxed text-cyan-950"><strong>Literature follow-up:</strong> {study.design.followup_definition} These rows are exploratory because the original test results were already inspected.</p>}{study.design.oracle_definition && <p className="mt-2 rounded-lg border border-fuchsia-200 bg-fuchsia-50 px-3 py-2 text-xs leading-relaxed text-fuchsia-900"><strong>Oracle arm:</strong> {study.design.oracle_definition} It is shown as a ceiling and was excluded from development-arm selection.</p>}</div>
        <div className="overflow-x-auto"><table className="w-full min-w-[1180px] text-left text-xs"><thead className="bg-gray-50 text-[10px] uppercase tracking-wider text-gray-500"><tr><th className="px-4 py-3">Conditioning arm</th><th className="px-4 py-3">Semantic</th><th className="px-4 py-3">Δ semantic</th><th className="px-4 py-3">Judge meaning</th><th className="px-4 py-3">Conversational fit</th><th className="px-4 py-3">Persona fit</th><th className="px-4 py-3">Disclosure calibration</th><th className="px-4 py-3">Behavior match</th>{test.context_only?.talk_type_match && <th className="px-4 py-3">Talk-type match</th>}<th className="px-4 py-3">Unsupported detail</th><th className="px-4 py-3">Persona recitation</th></tr></thead><tbody className="divide-y divide-gray-100">{study.design.arms.map(arm => {
          const metrics = test[arm]; const difference = deltas[arm];
          const oracle = arm === study.design.oracle_arm;
          const followup = followupArms.includes(arm);
          return <tr key={arm} className={oracle ? "bg-fuchsia-50/60" : followup ? "bg-cyan-50/50" : arm === selected ? "bg-blue-50/40" : ""}><td className="px-4 py-3"><div className="font-semibold text-gray-900">{ARM_NAMES[arm] || arm}</div>{arm === selected && <div className="mt-0.5 text-[10px] font-medium text-blue-600">Selected on development</div>}{followup && <div className="mt-0.5 text-[10px] font-semibold uppercase tracking-wider text-cyan-700">Exploratory follow-up</div>}{oracle && <div className="mt-0.5 text-[10px] font-semibold uppercase tracking-wider text-fuchsia-700">Future-informed oracle</div>}</td><td className="px-4 py-3"><Interval {...metrics.semantic_similarity} /></td><td className="px-4 py-3">{difference ? <Interval {...difference.semantic_similarity} delta /> : <span className="text-gray-400">Reference</span>}</td><td className="px-4 py-3"><Interval {...metrics.judge_meaning} /></td><td className="px-4 py-3"><Interval {...metrics.judge_fit} /></td><td className="px-4 py-3"><Interval {...metrics.world_model_compatibility} /></td><td className="px-4 py-3"><Interval {...metrics.disclosure_calibration} /></td><td className="px-4 py-3"><Interval {...metrics.behavior_match} /></td>{metrics.talk_type_match && <td className="px-4 py-3"><Interval {...metrics.talk_type_match} /></td>}<td className="px-4 py-3"><Interval {...metrics.unsupported_detail} /></td><td className="px-4 py-3"><Interval {...metrics.persona_recitation} /></td></tr>;
        })}</tbody></table></div>
        <div className="border-t border-amber-200 bg-amber-50 p-4 text-xs leading-relaxed text-amber-900"><strong>Interpret cautiously:</strong> Open-ended next turns can be valid without resembling the single observed reply. Use semantic, behavioral, talk-type, and unsupported-detail results together rather than optimizing one score.</div>
      </section>

      <section className="overflow-hidden rounded-2xl border border-gray-200 bg-white shadow-sm">
        <div className="flex flex-col gap-3 border-b border-gray-200 p-5 sm:flex-row sm:items-end sm:justify-between"><div><h2 className="font-bold text-gray-900">Inspect held-out cases</h2><p className="mt-1 text-xs text-gray-500">{study.design.test_samples} independent samples per arm. Predictive arms use only prior evidence; the labeled oracle uses target-redacted future evidence.</p></div><label><span className="mb-1 block text-[10px] font-semibold uppercase tracking-wider text-gray-500">Test conversation</span><select className="select-styled min-w-[180px]" value={caseId} onChange={event => setCaseId(event.target.value)}>{caseIds.map(id => <option key={id}>{id}</option>)}</select></label></div>
        {first && <div className="grid lg:grid-cols-[360px_1fr]"><aside className="border-b border-gray-200 bg-gray-50/70 p-5 lg:border-b-0 lg:border-r">{first.case_metadata && <div className="mb-4 rounded-xl border border-gray-200 bg-white p-3 text-xs leading-relaxed text-gray-600"><div className="font-semibold text-gray-900">{first.case_metadata.topic}</div><div>MI quality: {first.case_metadata.mi_quality}</div>{first.case_metadata.video_url && <a className="mt-1 inline-block font-medium text-oasis-600 hover:underline" href={first.case_metadata.video_url} target="_blank" rel="noreferrer">View source demonstration ↗</a>}</div>}<div className="space-y-3">{first.visible_prefix.map((turn, index) => <div key={index} className={`flex ${turn.role === "client" ? "justify-end" : "justify-start"}`}><div className={`max-w-[90%] rounded-2xl px-3 py-2 text-xs leading-relaxed ${turn.role === "client" ? "rounded-br-md bg-oasis-500 text-white" : "rounded-bl-md border border-gray-200 bg-white text-gray-700"}`}><div className={`mb-1 text-[9px] font-semibold uppercase tracking-wider ${turn.role === "client" ? "text-white/60" : "text-gray-400"}`}>{turn.role}</div>{turn.text}</div></div>)}</div><div className="mt-4 rounded-xl border-2 border-emerald-200 bg-emerald-50 p-3"><div className="text-[9px] font-bold uppercase tracking-wider text-emerald-700">Observed next reply · {first.reference_talk_type}</div><p className="mt-1 text-sm font-medium leading-relaxed text-emerald-950">“{first.reference}”</p></div><details className="mt-3 rounded-xl border border-gray-200 bg-white p-3"><summary className="cursor-pointer text-xs font-semibold text-gray-700">Prefix-only persona card</summary><div className="mt-3 space-y-3 text-xs leading-relaxed text-gray-600"><div><strong>Stable narrative</strong><ul className="mt-1 list-disc pl-4">{first.persona.stable_narrative.map(item => <li key={item}>{item}</li>)}</ul></div><div><strong>Communication</strong><ul className="mt-1 list-disc pl-4">{first.persona.communication_policy.map(item => <li key={item}>{item}</li>)}</ul></div><div><strong>Turn state</strong><ul className="mt-1 list-disc pl-4">{first.persona.state_at_turn.map(item => <li key={item}>{item}</li>)}</ul></div></div></details>{first.grounded_world_model && <details className="mt-3 rounded-xl border border-cyan-200 bg-cyan-50 p-3"><summary className="cursor-pointer text-xs font-semibold text-cyan-950">Prefix-only private world model</summary><div className="mt-3 space-y-3 text-xs leading-relaxed text-cyan-950">{Object.entries(first.grounded_world_model).map(([key, value]) => Array.isArray(value) ? value.length > 0 && <div key={key}><strong className="capitalize">{key.replace(/_/g, " ")}</strong><ul className="mt-1 list-disc pl-4">{value.map(item => <li key={item}>{item}</li>)}</ul></div> : value && <div key={key}><strong className="capitalize">{key.replace(/_/g, " ")}</strong><p className="mt-1">{value}</p></div>)}</div></details>}{first.retrospective_world_model && <details className="mt-3 rounded-xl border border-fuchsia-200 bg-fuchsia-50 p-3"><summary className="cursor-pointer text-xs font-semibold text-fuchsia-900">Retrospective private world model</summary><div className="mt-3 space-y-3 text-xs leading-relaxed text-fuchsia-950">{Object.entries(first.retrospective_world_model).map(([key, items]) => items.length > 0 && <div key={key}><strong className="capitalize">{key.replace(/_/g, " ")}</strong><ul className="mt-1 list-disc pl-4">{items.map(item => <li key={item}>{item}</li>)}</ul></div>)}</div></details>}</aside><div className="p-5"><div className="grid gap-4 xl:grid-cols-2">{study.design.arms.map(arm => <div key={arm} className={`rounded-2xl border p-3 ${arm === study.design.oracle_arm ? "border-fuchsia-200 bg-fuchsia-50/50" : followupArms.includes(arm) ? "border-cyan-200 bg-cyan-50/50" : "border-gray-200 bg-gray-50/50"}`}><div className="mb-2 text-sm font-bold text-gray-900">{ARM_NAMES[arm] || arm}{followupArms.includes(arm) && <span className="ml-2 rounded-full bg-cyan-100 px-2 py-0.5 text-[9px] uppercase tracking-wider text-cyan-700">Follow-up</span>}{arm === study.design.oracle_arm && <span className="ml-2 rounded-full bg-fuchsia-100 px-2 py-0.5 text-[9px] uppercase tracking-wider text-fuchsia-700">Oracle</span>}</div><div className="space-y-2">{rows.filter(row => row.arm === arm).map(row => <EmpiricalCandidate key={`${arm}-${row.sample}`} row={row} />)}</div></div>)}</div></div></div>}
      </section>

      <section className="grid gap-4 md:grid-cols-2"><div className="rounded-2xl border border-violet-200 bg-violet-50 p-5"><h2 className="font-bold text-violet-950">What this supports</h2><p className="mt-2 text-sm leading-relaxed text-violet-900">This study compares prompt-conditioning methods on held-out next-client turns from {study.dataset.name}. {study.dataset.population_note} {study.design.split_policy && `Split policy: ${study.design.split_policy}.`}</p></div><div className="rounded-2xl border border-rose-200 bg-rose-50 p-5"><h2 className="font-bold text-rose-950">What this cannot support</h2><ul className="mt-2 list-disc space-y-1.5 pl-4 text-sm leading-relaxed text-rose-900">{study.limitations.map(item => <li key={item}>{item}</li>)}</ul></div></section>
    </div>
  );
}

const VALIDATION_LABELS: Record<string, string> = {
  real_observed: "Observed human reply",
  context_only: "Context only",
  grounded_world_model: "Grounded world model",
  grounded_backstory: "Grounded backstory",
  persona_dump: "Persona-recitation control",
  role_leak: "Counselor-role control",
  cross_case: "Cross-conversation control",
};

function numeric(value: number | Record<string, number> | undefined) { return typeof value === "number" ? value : 0; }

function JudgeValidation({ result }: { result: JudgeValidationResults }) {
  const [dataset, setDataset] = useState(result.design.datasets[0]);
  const population = result.population_metrics[dataset] || {};
  const populationArms = ["real_observed", "context_only", "grounded_world_model", "grounded_backstory"];
  const reliabilityLabels: Record<string, string> = { order_invariance: "Candidate-order reversal", prompt_robustness: "Rubric wording", cross_model: "Judge strength" };
  return <div className="space-y-6 pb-16">
    <header><div className="mb-2 flex flex-wrap items-center gap-2"><span className="rounded-full bg-red-50 px-2.5 py-1 text-[11px] font-semibold uppercase tracking-wider text-red-700">Not ready for optimization</span><span className="text-xs text-gray-400">Stage 1 · {new Date(result.created_at).toLocaleString()}</span></div><h1 className="text-3xl font-bold tracking-tight text-gray-950">Automated judge validation</h1><p className="mt-2 max-w-4xl text-sm leading-relaxed text-gray-600">Tests whether the LLM rubric is internally consistent, resistant to order and prompt changes, and able to detect known simulator failures. {result.claim_boundary}</p></header>

    <section className="rounded-2xl border border-red-200 bg-red-50 p-5"><div className="grid gap-5 lg:grid-cols-[1.4fr_1fr]"><div><div className="text-xs font-semibold uppercase tracking-wider text-red-700">Decision</div><h2 className="mt-2 text-xl font-bold text-red-950">Do not use the current LLM judge as the prompt-optimization target.</h2><p className="mt-2 text-sm leading-relaxed text-red-900">It passes obvious-control checks but favors polished synthetic language over observed human language and cannot separate the three genuine simulation arms.</p></div><div className="grid grid-cols-2 gap-3"><div className="rounded-xl bg-white/80 p-3"><div className="text-2xl font-bold text-red-950">{result.checks_passed}/{result.checks_total}</div><div className="mt-1 text-[11px] text-red-800">Prespecified checks passed</div></div><div className="rounded-xl bg-white/80 p-3"><div className="text-2xl font-bold text-red-950">{result.design.total_cases}</div><div className="mt-1 text-[11px] text-red-800">Blinded cases</div></div></div></div></section>

    <section className="grid gap-4 lg:grid-cols-2">{result.red_flags.map(flag => <div key={flag.id} className={`rounded-2xl border p-4 ${flag.severity === "critical" ? "border-red-200 bg-red-50/60" : "border-amber-200 bg-amber-50/60"}`}><div className={`text-[10px] font-bold uppercase tracking-wider ${flag.severity === "critical" ? "text-red-700" : "text-amber-700"}`}>{flag.severity}</div><h3 className="mt-1 text-sm font-bold text-gray-950">{flag.finding}</h3><p className="mt-2 text-xs leading-relaxed text-gray-600">{flag.implication}</p></div>)}</section>

    <section className="overflow-hidden rounded-2xl border border-gray-200 bg-white shadow-sm"><div className="border-b border-gray-200 p-5"><h2 className="font-bold text-gray-900">Prespecified control checks</h2><p className="mt-1 text-xs text-gray-500">Thresholds were frozen before inspecting results.</p></div><div className="divide-y divide-gray-100">{result.control_checks.map(check => <div key={check.name} className="grid grid-cols-[28px_1fr_auto] items-center gap-3 px-5 py-3 text-sm"><span className={`flex h-6 w-6 items-center justify-center rounded-full text-xs font-bold ${check.passed ? "bg-emerald-100 text-emerald-700" : "bg-red-100 text-red-700"}`}>{check.passed ? "✓" : "×"}</span><span className="font-medium text-gray-800">{check.name}</span><span className="text-xs tabular-nums text-gray-500">{percent(check.observed)} · threshold {percent(check.threshold)}</span></div>)}</div></section>

    <section className="overflow-hidden rounded-2xl border border-gray-200 bg-white shadow-sm"><div className="border-b border-gray-200 p-5"><h2 className="font-bold text-gray-900">Reliability is necessary, but not validity</h2><p className="mt-1 text-xs text-gray-500">Exact agreement and Cohen’s κ across all anchored items and the six pairwise winners.</p></div><div className="overflow-x-auto"><table className="w-full min-w-[680px] text-left text-xs"><thead className="bg-gray-50 text-[10px] uppercase tracking-wider text-gray-500"><tr><th className="px-4 py-3">Perturbation</th><th className="px-4 py-3">Item agreement</th><th className="px-4 py-3">Item κ</th><th className="px-4 py-3">Pair agreement</th><th className="px-4 py-3">Pair κ</th></tr></thead><tbody className="divide-y divide-gray-100">{Object.entries(result.reliability).map(([key, value]) => <tr key={key}><td className="px-4 py-3 font-semibold text-gray-900">{reliabilityLabels[key] || key}</td><td className="px-4 py-3">{percent(value.pooled_items.agreement)}</td><td className="px-4 py-3">{value.pooled_items.kappa.toFixed(2)}</td><td className="px-4 py-3">{percent(value.pairwise_winners.agreement)}</td><td className="px-4 py-3">{value.pairwise_winners.kappa.toFixed(2)}</td></tr>)}</tbody></table></div></section>

    <section className="grid gap-4 lg:grid-cols-[1fr_1.2fr]"><div className="overflow-hidden rounded-2xl border border-gray-200 bg-white shadow-sm"><div className="border-b border-gray-200 p-5"><h2 className="font-bold text-gray-900">Anchored-item pass rates</h2><p className="mt-1 text-xs text-gray-500">A ceiling across genuine arms is a failed measurement, not a successful model.</p></div><div className="divide-y divide-gray-100">{["real_observed", "context_only", "grounded_world_model", "grounded_backstory", "persona_dump", "role_leak", "cross_case"].map(candidate => <div key={candidate} className="flex items-center justify-between gap-3 px-5 py-3"><span className="text-xs font-medium text-gray-700">{VALIDATION_LABELS[candidate]}</span><ScorePill label="Pass" value={numeric(result.candidate_summaries[candidate]?.mean_item_pass_rate)} /></div>)}</div></div><div className="overflow-hidden rounded-2xl border border-gray-200 bg-white shadow-sm"><div className="border-b border-gray-200 p-5"><h2 className="font-bold text-gray-900">Blinded pairwise preferences</h2><p className="mt-1 text-xs text-gray-500">Observed human replies were not identified to the judge.</p></div><div className="divide-y divide-gray-100">{Object.entries(result.pairwise_summaries).map(([key, pair]) => <div key={key} className="p-4"><div className="mb-2 flex items-center justify-between gap-3 text-xs"><span className="font-semibold text-gray-800">{VALIDATION_LABELS[pair.left]} vs {VALIDATION_LABELS[pair.right]}</span><span className="text-gray-400">n={pair.n}</span></div><div className="grid grid-cols-[1fr_auto_1fr] items-center gap-2 text-[11px]"><div><div className="h-2 overflow-hidden rounded-full bg-gray-100"><div className="h-full bg-oasis-500" style={{width: percent(pair.left_win_rate)}} /></div><div className="mt-1">{percent(pair.left_win_rate)} left</div></div><span className="text-gray-400">{percent(pair.tie_rate)} tie</span><div className="text-right"><div className="h-2 overflow-hidden rounded-full bg-gray-100"><div className="ml-auto h-full bg-violet-500" style={{width: percent(pair.right_win_rate)}} /></div><div className="mt-1">{percent(pair.right_win_rate)} right</div></div></div></div>)}</div></div></section>

    <section className="overflow-hidden rounded-2xl border border-gray-200 bg-white shadow-sm"><div className="flex flex-col gap-3 border-b border-gray-200 p-5 sm:flex-row sm:items-end sm:justify-between"><div><h2 className="font-bold text-gray-900">Reference-grounded population diagnostics</h2><p className="mt-1 text-xs text-gray-500">Deterministic PSI-Bench-style comparisons over every frozen test conversation; these expose failures the LLM judge misses.</p></div><label><span className="mb-1 block text-[10px] font-semibold uppercase tracking-wider text-gray-500">Dataset</span><select className="select-styled min-w-[150px]" value={dataset} onChange={event => setDataset(event.target.value)}>{result.design.datasets.map(item => <option key={item} value={item}>{item === "mibot" ? "MIBot" : "AnnoMI"}</option>)}</select></label></div><div className="overflow-x-auto"><table className="w-full min-w-[850px] text-left text-xs"><thead className="bg-gray-50 text-[10px] uppercase tracking-wider text-gray-500"><tr><th className="px-4 py-3">Source</th><th className="px-4 py-3">Mean words</th><th className="px-4 py-3">Word-count SD</th><th className="px-4 py-3">Hedging</th><th className="px-4 py-3">Length alignment</th><th className="px-4 py-3">Length KS ↓</th><th className="px-4 py-3">Unigram JSD ↓</th></tr></thead><tbody className="divide-y divide-gray-100">{populationArms.map(arm => { const metrics = population[arm] || {}; return <tr key={arm} className={arm === "real_observed" ? "bg-emerald-50/50" : ""}><td className="px-4 py-3 font-semibold text-gray-900">{VALIDATION_LABELS[arm]}</td><td className="px-4 py-3">{Number(metrics.mean_words || 0).toFixed(1)}</td><td className="px-4 py-3">{Number(metrics.sd_words || 0).toFixed(1)}</td><td className="px-4 py-3">{percent(Number(metrics.hedge_rate || 0))}</td><td className="px-4 py-3">{Number(metrics.length_log_ratio_similarity || 0).toFixed(0)}%</td><td className="px-4 py-3">{Number(metrics.length_ks_distance || 0).toFixed(2)}</td><td className="px-4 py-3">{Number(metrics.unigram_js_divergence || 0).toFixed(2)}</td></tr>; })}</tbody></table></div><div className="border-t border-amber-200 bg-amber-50 p-4 text-xs leading-relaxed text-amber-900"><strong>Trajectory gap:</strong> The current experiment has one synthetic checkpoint per conversation, so it cannot yet test persona drift or unrealistically rapid recovery. That requires multi-turn rollouts.</div></section>

    <section className="grid gap-4 md:grid-cols-2"><div className="rounded-2xl border border-emerald-200 bg-emerald-50 p-5"><h2 className="font-bold text-emerald-950">Safe to use now</h2><ul className="mt-2 list-disc space-y-1.5 pl-4 text-sm leading-relaxed text-emerald-900"><li>Deterministic population comparisons against real replies.</li><li>The LLM rubric as a coarse detector of role leakage, profile recitation, and extreme overdisclosure.</li><li>Judge disagreement as a trigger for review.</li></ul></div><div className="rounded-2xl border border-red-200 bg-red-50 p-5"><h2 className="font-bold text-red-950">Not safe to optimize yet</h2><ul className="mt-2 list-disc space-y-1.5 pl-4 text-sm leading-relaxed text-red-900"><li>Absolute LLM pass rates for genuine simulation arms.</li><li>LLM pairwise “realism” preferences.</li><li>A composite score combining fidelity, safety, and next-turn similarity.</li></ul></div></section>
  </div>;
}

function SandboxLab() {
  const [overview, setOverview] = useState<Overview | null>(null);
  const [result, setResult] = useState<RunResult | null>(null);
  const [selectedCase, setSelectedCase] = useState("");
  const [selectedTechniques, setSelectedTechniques] = useState<string[]>([]);
  const [contextMessages, setContextMessages] = useState(6);
  const [mode, setMode] = useState<"demo" | "live">("demo");
  const [model, setModel] = useState("gpt-4.1-mini");
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");
  const [promptCandidate, setPromptCandidate] = useState<Candidate | null>(null);

  useEffect(() => {
    simulationLab.overview().then(data => {
      setOverview(data);
      setSelectedCase(data.cases[0]?.id || "");
      setSelectedTechniques(data.techniques.map(item => item.id));
    }).catch(err => setError(err.message)).finally(() => setLoading(false));
  }, []);

  const currentCase = overview?.cases.find(item => item.id === selectedCase) || null;
  const visibleCandidates = useMemo(() => result?.candidates.filter(item => item.case_id === selectedCase) || [], [result, selectedCase]);

  async function runBenchmark() {
    if (!overview || !selectedTechniques.length) return;
    setRunning(true); setError("");
    try {
      const next = await simulationLab.run({
        case_ids: overview.cases.map(item => item.id),
        technique_ids: selectedTechniques,
        context_messages: contextMessages,
        mode,
        model,
      });
      setResult(next);
    } catch (err) { setError(err instanceof Error ? err.message : "Run failed"); }
    finally { setRunning(false); }
  }

  function toggleTechnique(id: string) {
    setSelectedTechniques(current => current.includes(id) ? current.filter(item => item !== id) : [...current, id]);
  }

  if (loading) return <div className="flex min-h-[55vh] items-center justify-center"><div className="h-8 w-8 animate-spin rounded-full border-2 border-gray-200 border-t-oasis-500" /></div>;
  if (!overview) return <div className="rounded-2xl border border-red-200 bg-red-50 p-5 text-sm text-red-700">{error || "The simulation lab could not load."}</div>;

  return (
    <div className="space-y-6 pb-16">
      <header className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <div className="mb-2 flex items-center gap-2"><span className="rounded-full bg-oasis-50 px-2.5 py-1 text-[11px] font-semibold uppercase tracking-wider text-oasis-700">Research prototype</span><span className="text-xs text-gray-400">Held-out continuation benchmark</span></div>
          <h1 className="text-3xl font-bold tracking-tight text-gray-950">Participant Simulation Lab</h1>
          <p className="mt-2 max-w-3xl text-sm leading-relaxed text-gray-600">Compare how persona conditioning changes the next message a simulated emerging-adult client produces—without showing the model the real continuation.</p>
        </div>
        <button onClick={runBenchmark} disabled={running || selectedTechniques.length === 0} className="btn-primary min-w-[170px]">
          {running ? <><span className="h-4 w-4 animate-spin rounded-full border-2 border-white/40 border-t-white" />Generating…</> : <>Run all cases <span aria-hidden>→</span></>}
        </button>
      </header>

      <section className="rounded-2xl border border-amber-200 bg-amber-50/70 p-4">
        <div className="flex gap-3"><span className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-amber-100 text-xs font-bold text-amber-700">!</span><div><div className="text-sm font-semibold text-amber-900">Illustrative data, deliberately separated from evidence</div><p className="mt-1 text-xs leading-relaxed text-amber-800">{overview.dataset.provenance}. {overview.dataset.warning}</p></div></div>
      </section>

      <section className="grid gap-4 rounded-2xl border border-gray-200 bg-white p-5 shadow-sm lg:grid-cols-[1fr_1fr_1.2fr]">
        <label className="block"><span className="mb-2 block text-xs font-semibold uppercase tracking-wider text-gray-500">Run mode</span><div className="grid grid-cols-2 rounded-xl bg-gray-100 p-1">{(["demo", "live"] as const).map(value => <button key={value} onClick={() => setMode(value)} className={`rounded-lg px-3 py-2 text-sm font-medium ${mode === value ? "bg-white text-gray-900 shadow-sm" : "text-gray-500"}`}>{value === "demo" ? "Fixed demo" : "Live model"}</button>)}</div></label>
        <label className="block"><span className="mb-2 block text-xs font-semibold uppercase tracking-wider text-gray-500">Visible context</span><select className="select-styled w-full" value={contextMessages} onChange={event => setContextMessages(Number(event.target.value))}>{[2, 4, 6, 8, 10, 12].map(value => <option key={value} value={value}>Last {value} messages</option>)}</select></label>
        <label className="block"><span className="mb-2 block text-xs font-semibold uppercase tracking-wider text-gray-500">Generation model</span><input className="input-styled" value={model} onChange={event => setModel(event.target.value)} disabled={mode === "demo"} /><span className="mt-1.5 block text-[11px] text-gray-400">{mode === "demo" ? "Demo replies are fixed for reproducible UI review." : "Uses the OpenAI key configured in Settings."}</span></label>
        <div className="lg:col-span-3"><div className="mb-2 text-xs font-semibold uppercase tracking-wider text-gray-500">Conditioning arms</div><div className="flex flex-wrap gap-2">{overview.techniques.map(item => { const checked = selectedTechniques.includes(item.id); return <button key={item.id} onClick={() => toggleTechnique(item.id)} className={`inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-medium ${checked ? "border-oasis-300 bg-oasis-50 text-oasis-700" : "border-gray-200 bg-white text-gray-400"}`}><span className={`flex h-4 w-4 items-center justify-center rounded-full ${checked ? "bg-oasis-500 text-white" : "bg-gray-100"}`}>{checked ? "✓" : ""}</span>{item.short_name}</button>; })}</div></div>
      </section>

      {error && <div className="rounded-2xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">{error}</div>}

      <section>
        <div className="mb-3 flex items-end justify-between"><div><h2 className="text-lg font-bold text-gray-900">Conditioning ladder</h2><p className="mt-1 text-xs text-gray-500">Each arm adds one layer, making failures and gains attributable.</p></div>{result && <div className="text-xs text-gray-400">Mean across {overview.cases.length} cases</div>}</div>
        <TechniqueRail techniques={overview.techniques.filter(item => selectedTechniques.includes(item.id))} result={result} />
      </section>

      <section className="rounded-2xl border border-gray-200 bg-white shadow-sm">
        <div className="flex flex-col gap-3 border-b border-gray-200 p-4 sm:flex-row sm:items-center sm:justify-between">
          <div><h2 className="font-bold text-gray-900">Case comparison</h2><p className="mt-0.5 text-xs text-gray-500">Inspect the same held-out turn across every conditioning arm.</p></div>
          <div className="flex flex-wrap gap-2">{overview.cases.map(item => <button key={item.id} onClick={() => setSelectedCase(item.id)} className={`rounded-xl px-3 py-2 text-xs font-medium ${selectedCase === item.id ? "bg-gray-900 text-white" : "bg-gray-100 text-gray-600 hover:bg-gray-200"}`}>{item.title}</button>)}</div>
        </div>
        {currentCase && <div className="grid lg:grid-cols-[350px_1fr]">
          <aside className="border-b border-gray-200 bg-gray-50/70 p-5 lg:border-b-0 lg:border-r">
            <div className="mb-4"><div className="text-xs font-semibold uppercase tracking-wider text-oasis-600">Age {currentCase.persona.age} · {currentCase.topic}</div><h3 className="mt-1 text-lg font-bold text-gray-900">{currentCase.title}</h3><p className="mt-1 text-xs leading-relaxed text-gray-500">{currentCase.persona.setting}</p></div>
            <Transcript item={currentCase} contextMessages={contextMessages} />
            <div className="mt-4 rounded-2xl border-2 border-emerald-200 bg-emerald-50 p-4"><div className="mb-1 text-[10px] font-bold uppercase tracking-wider text-emerald-700">IRL held-out continuation</div><p className="text-sm font-medium leading-relaxed text-emerald-950">“{currentCase.held_out.text}”</p></div>
          </aside>
          <div className="p-5">
            {result ? <div className="grid gap-4 xl:grid-cols-2">{visibleCandidates.map(candidate => { const technique = overview.techniques.find(item => item.id === candidate.technique_id)!; return <CandidateCard key={candidate.technique_id} candidate={candidate} technique={technique} onPrompt={() => setPromptCandidate(candidate)} />; })}</div> : <div className="flex min-h-[420px] flex-col items-center justify-center text-center"><div className="mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-oasis-50 text-2xl text-oasis-600">↗</div><h3 className="font-semibold text-gray-900">Ready to compare continuations</h3><p className="mt-2 max-w-sm text-sm leading-relaxed text-gray-500">Run the benchmark to generate one reply per conditioning arm and score each against the held-out response.</p><button onClick={runBenchmark} className="btn-primary mt-5">Run fixed demo</button></div>}
          </div>
        </div>}
      </section>

      {result && <section className="rounded-2xl border border-sky-200 bg-sky-50/70 p-4 text-xs leading-relaxed text-sky-900"><strong>How to read this:</strong> {result.interpretation_note}</section>}

      <section className="grid gap-4 md:grid-cols-2">
        <div className="rounded-2xl border border-gray-200 bg-white p-5"><h2 className="font-bold text-gray-900">Study guardrails</h2><ol className="mt-3 space-y-3">{overview.recommended_design.map((item, index) => <li key={item} className="flex gap-3 text-sm leading-relaxed text-gray-600"><span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-gray-100 text-[10px] font-bold text-gray-600">{index + 1}</span>{item}</li>)}</ol></div>
        <div className="rounded-2xl border border-gray-200 bg-white p-5"><h2 className="font-bold text-gray-900">What to add when real data arrives</h2><div className="mt-3 space-y-3 text-sm text-gray-600"><p>Import consented transcripts into participant-level train, development, and test partitions.</p><p>Compute confidence intervals with participant-level bootstrap resampling—not turns treated as independent.</p><p>Blind expert raters to both condition and source; include real-vs-synthetic discrimination, developmental fit, and clinically important omissions.</p></div></div>
      </section>

      {promptCandidate && <div className="fixed inset-0 z-50 flex justify-end bg-gray-950/30 backdrop-blur-sm" onClick={() => setPromptCandidate(null)}><div className="h-full w-full max-w-2xl overflow-y-auto bg-white p-6 shadow-2xl" onClick={event => event.stopPropagation()}><div className="mb-5 flex items-start justify-between"><div><div className="text-xs font-semibold uppercase tracking-wider text-oasis-600">Leakage inspection</div><h2 className="mt-1 text-xl font-bold text-gray-900">Exact conditioning prompt</h2><p className="mt-1 text-xs text-gray-500">The held-out reply is intentionally absent.</p></div><button onClick={() => setPromptCandidate(null)} className="rounded-xl bg-gray-100 px-3 py-2 text-sm text-gray-600 hover:bg-gray-200">Close</button></div><pre className="whitespace-pre-wrap rounded-2xl bg-gray-950 p-5 font-mono text-xs leading-relaxed text-gray-200">{promptCandidate.prompt_preview}</pre></div></div>}
    </div>
  );
}

export default function SimulationLabPage() {
  const [study, setStudy] = useState<EmpiricalResults | null>(null);
  const [judgeValidation, setJudgeValidation] = useState<JudgeValidationResults | null>(null);
  const [studies, setStudies] = useState<EmpiricalStudySummary[]>([]);
  const [selectedStudyId, setSelectedStudyId] = useState("");
  const [view, setView] = useState<"empirical" | "validation" | "sandbox">("empirical");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([simulationLab.empiricalStudies(), simulationLab.judgeValidation().catch(() => null)]).then(async ([available, validation]) => {
      setStudies(available);
      setJudgeValidation(validation);
      const preferred = available.find(item => item.study_id === "annomi-heldout-v1") || available[0];
      if (!preferred) { setView("sandbox"); return; }
      setSelectedStudyId(preferred.study_id);
      setStudy(await simulationLab.empirical(preferred.study_id));
    }).catch(() => setView("sandbox")).finally(() => setLoading(false));
  }, []);

  async function selectStudy(studyId: string) {
    setSelectedStudyId(studyId);
    setLoading(true);
    try { setStudy(await simulationLab.empirical(studyId)); setView("empirical"); }
    finally { setLoading(false); }
  }

  if (loading) return <div className="flex min-h-[55vh] items-center justify-center"><div className="h-8 w-8 animate-spin rounded-full border-2 border-gray-200 border-t-oasis-500" /></div>;
  return (
    <div>
      {study && <div className="mb-6 flex flex-wrap items-center gap-3"><div className="inline-flex rounded-xl border border-gray-200 bg-white p-1 shadow-sm"><button onClick={() => setView("empirical")} className={`rounded-lg px-4 py-2 text-xs font-semibold ${view === "empirical" ? "bg-gray-900 text-white" : "text-gray-500 hover:bg-gray-50"}`}>Empirical results</button>{judgeValidation && <button onClick={() => setView("validation")} className={`rounded-lg px-4 py-2 text-xs font-semibold ${view === "validation" ? "bg-gray-900 text-white" : "text-gray-500 hover:bg-gray-50"}`}>Judge validation</button>}<button onClick={() => setView("sandbox")} className={`rounded-lg px-4 py-2 text-xs font-semibold ${view === "sandbox" ? "bg-gray-900 text-white" : "text-gray-500 hover:bg-gray-50"}`}>Experiment sandbox</button></div>{view === "empirical" && studies.length > 1 && <label className="flex items-center gap-2 text-xs font-semibold text-gray-500"><span>Dataset</span><select className="select-styled min-w-[180px]" value={selectedStudyId} onChange={event => selectStudy(event.target.value)}>{studies.map(item => <option key={item.study_id} value={item.study_id}>{item.name} · {item.test_cases} test</option>)}</select></label>}</div>}
      {view === "empirical" && study ? <EmpiricalStudy study={study} /> : view === "validation" && judgeValidation ? <JudgeValidation result={judgeValidation} /> : <SandboxLab />}
    </div>
  );
}
