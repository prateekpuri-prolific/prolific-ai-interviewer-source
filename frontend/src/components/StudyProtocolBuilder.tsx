import { useEffect, useState } from "react";
import type { ResearchSpec, ResearchTarget } from "../lib/api";

const lines = (value: string) => value.split("\n").map((item) => item.trim()).filter(Boolean);
const joined = (value: string[]) => value.join("\n");

export function defaultResearchSpec(): ResearchSpec {
  return {
    version: "0.1", stage: "pilot", method: "exploration",
    decision_to_inform: "", research_questions: [],
    population: {
      participant_role: "", subject: "", knowledge_relationship: "",
      inclusion_notes: "", knowledge_limits: [], terminology_notes: "",
      accessibility_notes: "", sensitivity_notes: "",
    },
    targets: [],
    clarification_policy: {
      participant_definition: "", in_scope: [], out_of_scope: [],
      ambiguous_cases: [],
      ambiguity_handling: "If it is unclear whether an experience fits, invite the participant to describe the relevant feature or interaction briefly and treat it as ambiguous rather than forcing a classification.",
    },
    interviewer_stance: { organization_label: "", disclosure: "", boundaries: [] },
    participant_information: {
      purpose_summary: "", estimated_minutes: 30,
      participation_summary: "",
      ai_disclosure: "You will be speaking with an AI interviewer.",
      data_handling_summary: "",
      data_controller: "",
      data_recipients_summary: "",
      risks_summary: "",
      benefits_summary: "",
      compensation_summary: "",
      privacy_reminder: "Please avoid sharing names or other details that identify you or anyone else.",
      rights: ["You may skip any question.", "You may stop the interview at any time."],
      consent_required: true,
      consent_label: "I have read this information and agree to take part.",
      researcher_identity: "",
      researcher_contact: "",
      complaints_contact: "",
      ethics_review_reference: "",
      data_retention_period: "",
      withdrawal_instructions: "",
      privacy_notice_url: "",
    },
    fielding: {
      recruitment_source: "prolific", response_limit: 50,
      soft_launch_limit: 10, participant_id_parameter: "PROLIFIC_PID",
      require_profile: false,
      completion_url: "",
      no_consent_url: "",
      inactivity_url: "",
    },
    analysis_notes: "",
  };
}

const newTarget = (index: number): ResearchTarget => ({
  id: `target_${index + 1}`, name: "", description: "", evidence_needed: [],
  do_not_infer: [], required: true,
});

function Field({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return <label className="block">
    <span className="block text-xs font-medium text-gray-700 mb-1">{label}</span>
    {hint && <span className="block text-[11px] text-gray-400 mb-1.5 leading-relaxed">{hint}</span>}
    {children}
  </label>;
}

export default function StudyProtocolBuilder({ value, onSave }: {
  value: ResearchSpec | null;
  onSave: (spec: ResearchSpec) => Promise<void>;
}) {
  const [spec, setSpec] = useState<ResearchSpec>(value ?? defaultResearchSpec());
  const [expanded, setExpanded] = useState(!value);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => setSpec(value ?? defaultResearchSpec()), [value]);

  const save = async () => {
    setSaving(true); setError(""); setSaved(false);
    try {
      await onSave(spec); setSaved(true); window.setTimeout(() => setSaved(false), 2500);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not save the protocol");
    } finally { setSaving(false); }
  };
  const setPopulation = (patch: Partial<ResearchSpec["population"]>) => setSpec((current) => ({ ...current, population: { ...current.population, ...patch } }));
  const setClarificationPolicy = (patch: Partial<ResearchSpec["clarification_policy"]>) => setSpec((current) => ({ ...current, clarification_policy: { ...current.clarification_policy, ...patch } }));
  const setStance = (patch: Partial<ResearchSpec["interviewer_stance"]>) => setSpec((current) => ({ ...current, interviewer_stance: { ...current.interviewer_stance, ...patch } }));
  const setParticipantInfo = (patch: Partial<ResearchSpec["participant_information"]>) => setSpec((current) => ({ ...current, participant_information: { ...current.participant_information, ...patch } }));
  const setTarget = (index: number, patch: Partial<ResearchTarget>) => setSpec((current) => ({ ...current, targets: current.targets.map((target, i) => i === index ? { ...target, ...patch } : target) }));

  return <section className="mb-6 card overflow-hidden">
    <button type="button" onClick={() => setExpanded((open) => !open)} className="w-full px-5 py-4 flex items-center justify-between text-left hover:bg-gray-50/50">
      <div>
        <div className="flex items-center gap-2">
          <h2 className="text-sm font-semibold text-gray-900">Research protocol</h2>
          <span className={`rounded-full px-2 py-0.5 text-[10px] font-medium ${spec.stage === "field" ? "bg-blue-50 text-blue-700" : "bg-amber-50 text-amber-700"}`}>
            {value ? `${spec.stage} · v${spec.version}` : "not configured"}
          </span>
        </div>
        <p className="mt-1 text-xs text-gray-500">Define the decision, population, evidence targets, safeguards, and participant terms once for every agent in this study.</p>
      </div>
      <svg className={`h-4 w-4 text-gray-400 transition-transform ${expanded ? "rotate-180" : ""}`} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" /></svg>
    </button>

    {expanded && <div className="border-t border-gray-100 px-5 py-5 space-y-6">
      <div className="rounded-xl border border-blue-100 bg-blue-50/60 px-4 py-3 text-xs text-blue-800 leading-relaxed">Runtime prompts receive only interview-relevant context. Participant information is shown before start; analysis notes remain researcher-only. Every session saves an immutable snapshot.</div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Field label="Protocol version" hint="Increment when changes affect comparability."><input className="input-styled w-full" value={spec.version} onChange={(e) => setSpec({ ...spec, version: e.target.value })} /></Field>
        <Field label="Stage"><select className="select-styled w-full" value={spec.stage} onChange={(e) => setSpec({ ...spec, stage: e.target.value as ResearchSpec["stage"] })}><option value="pilot">Pilot</option><option value="field">Field</option></select></Field>
        <Field label="Study method"><select className="select-styled w-full" value={spec.method} onChange={(e) => setSpec({ ...spec, method: e.target.value as ResearchSpec["method"] })}><option value="exploration">Exploration</option><option value="journey">Journey</option><option value="needs">Needs</option><option value="concept_test">Concept test</option><option value="evaluation">Evaluation</option><option value="standardized_comparison">Standardized comparison</option><option value="other">Other</option></select></Field>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Field label="Decision to inform" hint="What will someone decide differently after this research?"><textarea className="input-styled w-full min-h-24" value={spec.decision_to_inform} onChange={(e) => setSpec({ ...spec, decision_to_inform: e.target.value })} placeholder="e.g. Which support should we build for parents…" /></Field>
        <Field label="Research questions" hint="One per line. These guide inquiry; they are not interview questions."><textarea className="input-styled w-full min-h-24" value={joined(spec.research_questions)} onChange={(e) => setSpec({ ...spec, research_questions: lines(e.target.value) })} placeholder="How do parents learn about their child's AI use?" /></Field>
      </div>

      <details open className="rounded-xl border border-gray-200">
        <summary className="cursor-pointer list-none px-4 py-3 text-sm font-semibold text-gray-800">Target population and knowledge boundary</summary>
        <div className="border-t border-gray-100 p-4 grid grid-cols-1 md:grid-cols-2 gap-4">
          <Field label="Participant role"><input className="input-styled w-full" value={spec.population.participant_role} onChange={(e) => setPopulation({ participant_role: e.target.value })} placeholder="Parents or guardians of children aged…" /></Field>
          <Field label="Subject"><input className="input-styled w-full" value={spec.population.subject} onChange={(e) => setPopulation({ subject: e.target.value })} placeholder="Their child's use of generative AI" /></Field>
          <Field label="Relationship to the topic" hint="What can they know directly versus second-hand?"><textarea className="input-styled w-full min-h-20" value={spec.population.knowledge_relationship} onChange={(e) => setPopulation({ knowledge_relationship: e.target.value })} /></Field>
          <Field label="Recruitment / inclusion context"><textarea className="input-styled w-full min-h-20" value={spec.population.inclusion_notes} onChange={(e) => setPopulation({ inclusion_notes: e.target.value })} /></Field>
          <Field label="Knowledge limits" hint="One per line; used to prevent overclaiming."><textarea className="input-styled w-full min-h-20" value={joined(spec.population.knowledge_limits)} onChange={(e) => setPopulation({ knowledge_limits: lines(e.target.value) })} placeholder="May not observe use at school\nMay infer rather than see private chats" /></Field>
          <Field label="Terminology guidance"><textarea className="input-styled w-full min-h-20" value={spec.population.terminology_notes} onChange={(e) => setPopulation({ terminology_notes: e.target.value })} placeholder="Use the participant's words before introducing labels." /></Field>
          <Field label="Accessibility / communication"><textarea className="input-styled w-full min-h-20" value={spec.population.accessibility_notes} onChange={(e) => setPopulation({ accessibility_notes: e.target.value })} /></Field>
          <Field label="Sensitive-topic guidance"><textarea className="input-styled w-full min-h-20" value={spec.population.sensitivity_notes} onChange={(e) => setPopulation({ sensitivity_notes: e.target.value })} /></Field>
        </div>
      </details>

      <details open className="rounded-xl border border-gray-200">
        <summary className="cursor-pointer list-none px-4 py-3 text-sm font-semibold text-gray-800">Participant scope clarifications</summary>
        <div className="border-t border-gray-100 p-4 space-y-4">
          <p className="text-xs leading-relaxed text-gray-500">Used only when a participant asks what a term means or whether a borderline example counts. The interviewer gives the smallest relevant answer and then returns to the current question.</p>
          <Field label="Participant-facing definition" hint="One or two neutral sentences; avoid an exhaustive example list."><textarea className="input-styled w-full min-h-20" value={spec.clarification_policy.participant_definition} onChange={(e) => setClarificationPolicy({ participant_definition: e.target.value })} placeholder="For this study, we mean…" /></Field>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Field label="In scope" hint="One category or boundary per line."><textarea className="input-styled w-full min-h-24" value={joined(spec.clarification_policy.in_scope)} onChange={(e) => setClarificationPolicy({ in_scope: lines(e.target.value) })} /></Field>
            <Field label="Out of scope" hint="One category or boundary per line."><textarea className="input-styled w-full min-h-24" value={joined(spec.clarification_policy.out_of_scope)} onChange={(e) => setClarificationPolicy({ out_of_scope: lines(e.target.value) })} /></Field>
            <Field label="Ambiguous cases" hint="Cases that depend on the feature or interaction."><textarea className="input-styled w-full min-h-24" value={joined(spec.clarification_policy.ambiguous_cases)} onChange={(e) => setClarificationPolicy({ ambiguous_cases: lines(e.target.value) })} /></Field>
            <Field label="How to handle ambiguity"><textarea className="input-styled w-full min-h-24" value={spec.clarification_policy.ambiguity_handling} onChange={(e) => setClarificationPolicy({ ambiguity_handling: e.target.value })} /></Field>
          </div>
        </div>
      </details>

      <details open className="rounded-xl border border-gray-200">
        <summary className="cursor-pointer list-none px-4 py-3 text-sm font-semibold text-gray-800">Evidence targets</summary>
        <div className="border-t border-gray-100 p-4 space-y-4">
          <p className="text-xs text-gray-500">Define useful evidence and conclusions the interviewer must not make. Guide questions can reference these IDs.</p>
          {spec.targets.map((target, index) => <div key={`${target.id}-${index}`} className="rounded-xl bg-gray-50 p-4 space-y-3">
            <div className="grid grid-cols-1 md:grid-cols-[140px_1fr_auto] gap-3">
              <Field label="ID"><input className="input-styled w-full font-mono text-xs" value={target.id} onChange={(e) => setTarget(index, { id: e.target.value.replace(/[^a-zA-Z0-9_-]/g, "_") })} /></Field>
              <Field label="Name"><input className="input-styled w-full" value={target.name} onChange={(e) => setTarget(index, { name: e.target.value })} placeholder="Concrete usage episodes" /></Field>
              <label className="flex items-end gap-2 pb-2 text-xs text-gray-600"><input type="checkbox" checked={target.required} onChange={(e) => setTarget(index, { required: e.target.checked })} /> Required</label>
            </div>
            <Field label="Description"><textarea className="input-styled w-full min-h-16" value={target.description} onChange={(e) => setTarget(index, { description: e.target.value })} /></Field>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <Field label="Evidence needed" hint="One per line"><textarea className="input-styled w-full min-h-20" value={joined(target.evidence_needed)} onChange={(e) => setTarget(index, { evidence_needed: lines(e.target.value) })} /></Field>
              <Field label="Do not infer" hint="One per line"><textarea className="input-styled w-full min-h-20" value={joined(target.do_not_infer)} onChange={(e) => setTarget(index, { do_not_infer: lines(e.target.value) })} /></Field>
            </div>
            <button type="button" className="text-xs text-red-600 hover:text-red-700" onClick={() => setSpec({ ...spec, targets: spec.targets.filter((_, i) => i !== index) })}>Remove target</button>
          </div>)}
          <button type="button" className="btn-secondary !py-2 !px-3 !text-xs" onClick={() => setSpec({ ...spec, targets: [...spec.targets, newTarget(spec.targets.length)] })}>Add evidence target</button>
        </div>
      </details>

      <details open className="rounded-xl border border-gray-200">
        <summary className="cursor-pointer list-none px-4 py-3 text-sm font-semibold text-gray-800">Participant information and consent</summary>
        <div className="border-t border-gray-100 p-4 grid grid-cols-1 md:grid-cols-2 gap-4">
          <Field label="Purpose summary" hint="Shown before the interview starts."><textarea className="input-styled w-full min-h-24" value={spec.participant_information.purpose_summary} onChange={(e) => setParticipantInfo({ purpose_summary: e.target.value })} /></Field>
          <Field label="What participation involves" hint="Tasks, response mode, and what the participant will be asked to do."><textarea className="input-styled w-full min-h-24" value={spec.participant_information.participation_summary} onChange={(e) => setParticipantInfo({ participation_summary: e.target.value })} /></Field>
          <Field label="Data handling summary"><textarea className="input-styled w-full min-h-24" value={spec.participant_information.data_handling_summary} onChange={(e) => setParticipantInfo({ data_handling_summary: e.target.value })} /></Field>
          <Field label="Data controller"><input className="input-styled w-full" value={spec.participant_information.data_controller} onChange={(e) => setParticipantInfo({ data_controller: e.target.value })} /></Field>
          <Field label="Who receives the data"><textarea className="input-styled w-full min-h-20" value={spec.participant_information.data_recipients_summary} onChange={(e) => setParticipantInfo({ data_recipients_summary: e.target.value })} /></Field>
          <Field label="Risks or discomforts"><textarea className="input-styled w-full min-h-20" value={spec.participant_information.risks_summary} onChange={(e) => setParticipantInfo({ risks_summary: e.target.value })} /></Field>
          <Field label="Benefits"><textarea className="input-styled w-full min-h-20" value={spec.participant_information.benefits_summary} onChange={(e) => setParticipantInfo({ benefits_summary: e.target.value })} /></Field>
          <Field label="Compensation"><textarea className="input-styled w-full min-h-20" value={spec.participant_information.compensation_summary} onChange={(e) => setParticipantInfo({ compensation_summary: e.target.value })} /></Field>
          <Field label="AI disclosure"><textarea className="input-styled w-full min-h-16" value={spec.participant_information.ai_disclosure} onChange={(e) => setParticipantInfo({ ai_disclosure: e.target.value })} /></Field>
          <Field label="Privacy reminder"><textarea className="input-styled w-full min-h-16" value={spec.participant_information.privacy_reminder} onChange={(e) => setParticipantInfo({ privacy_reminder: e.target.value })} /></Field>
          <Field label="Participant rights" hint="One per line"><textarea className="input-styled w-full min-h-20" value={joined(spec.participant_information.rights)} onChange={(e) => setParticipantInfo({ rights: lines(e.target.value) })} /></Field>
          <Field label="Researcher identity"><input className="input-styled w-full" value={spec.participant_information.researcher_identity} onChange={(e) => setParticipantInfo({ researcher_identity: e.target.value })} /></Field>
          <Field label="Researcher contact"><input className="input-styled w-full" value={spec.participant_information.researcher_contact} onChange={(e) => setParticipantInfo({ researcher_contact: e.target.value })} /></Field>
          <Field label="Complaints contact"><input className="input-styled w-full" value={spec.participant_information.complaints_contact} onChange={(e) => setParticipantInfo({ complaints_contact: e.target.value })} /></Field>
          <Field label="Ethics review reference" hint="Optional. Leave blank unless approval or exemption is confirmed."><input className="input-styled w-full" value={spec.participant_information.ethics_review_reference} onChange={(e) => setParticipantInfo({ ethics_review_reference: e.target.value })} /></Field>
          <Field label="Data retention period"><input className="input-styled w-full" value={spec.participant_information.data_retention_period} onChange={(e) => setParticipantInfo({ data_retention_period: e.target.value })} /></Field>
          <Field label="Withdrawal or deletion"><textarea className="input-styled w-full min-h-20" value={spec.participant_information.withdrawal_instructions} onChange={(e) => setParticipantInfo({ withdrawal_instructions: e.target.value })} /></Field>
          <Field label="Privacy notice URL"><input type="url" className="input-styled w-full" value={spec.participant_information.privacy_notice_url} onChange={(e) => setParticipantInfo({ privacy_notice_url: e.target.value })} /></Field>
          <div className="space-y-3">
            <Field label="Estimated minutes"><input type="number" min={1} max={240} className="input-styled w-full" value={spec.participant_information.estimated_minutes ?? ""} onChange={(e) => setParticipantInfo({ estimated_minutes: e.target.value ? Number(e.target.value) : null })} /></Field>
            <label className="flex items-center gap-2 text-xs text-gray-700"><input type="checkbox" checked={spec.participant_information.consent_required} onChange={(e) => setParticipantInfo({ consent_required: e.target.checked })} /> Require affirmative consent before connecting</label>
          </div>
          <div className="md:col-span-2"><Field label="Consent checkbox label"><input className="input-styled w-full" value={spec.participant_information.consent_label} onChange={(e) => setParticipantInfo({ consent_label: e.target.value })} /></Field></div>
        </div>
      </details>

      <details className="rounded-xl border border-gray-200">
        <summary className="cursor-pointer list-none px-4 py-3 text-sm font-semibold text-gray-800">Interviewer stance and researcher notes</summary>
        <div className="border-t border-gray-100 p-4 grid grid-cols-1 md:grid-cols-2 gap-4">
          <Field label="Organization label"><input className="input-styled w-full" value={spec.interviewer_stance.organization_label} onChange={(e) => setStance({ organization_label: e.target.value })} /></Field>
          <Field label="Interviewer disclosure"><textarea className="input-styled w-full min-h-20" value={spec.interviewer_stance.disclosure} onChange={(e) => setStance({ disclosure: e.target.value })} /></Field>
          <Field label="Boundaries" hint="One per line"><textarea className="input-styled w-full min-h-20" value={joined(spec.interviewer_stance.boundaries)} onChange={(e) => setStance({ boundaries: lines(e.target.value) })} /></Field>
          <Field label="Analysis notes (never sent to interviewer)" hint="Hypotheses, sensitizing concepts, and planned comparisons belong here."><textarea className="input-styled w-full min-h-20" value={spec.analysis_notes} onChange={(e) => setSpec({ ...spec, analysis_notes: e.target.value })} /></Field>
        </div>
      </details>

      {error && <p className="text-xs text-red-600">{error}</p>}
      <div className="flex items-center justify-between">
        <p className="text-[11px] text-gray-400">Pilot first; increment the version before material fieldwork changes.</p>
        <button type="button" onClick={save} disabled={saving} className="btn-primary !py-2 !px-4 !text-xs disabled:opacity-50">{saving ? "Saving…" : saved ? "Saved" : "Save protocol"}</button>
      </div>
    </div>}
  </section>;
}
