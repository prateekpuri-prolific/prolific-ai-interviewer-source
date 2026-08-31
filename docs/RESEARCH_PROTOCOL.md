# Research protocol

OASIS separates the research design from the interviewer delivery settings.
This keeps one study portable across topics and populations without turning the
runtime into a multi-agent research planner.

## The current vertical slice

- `Study.research_spec` stores a version and pilot/field stage, the decision to
  inform, research questions, population knowledge boundaries, evidence
  targets, interviewer stance, participant information, and analysis-only
  notes.
- The runtime compiler receives only context useful for neutral interviewing.
  It never includes analysis notes or participant-facing consent copy.
- Structured guide questions can declare a purpose, evidence goals, linked
  target IDs, prohibited assumptions, priority, sensitivity, interview phase,
  and visit scope. First-time and returning participants can therefore follow
  different openings without forking the whole guide.
- Emergent probes are opt-in and bounded to a small set of neutral probe types.
  Existing guides retain their prior behavior by default.
- Web voice and text interviews enforce affirmative consent when the study
  requires it. The public widget receives participant information but no
  private research fields.
- Each session stores the protocol version/stage, consent state, and an
  immutable protocol/agent snapshot. List endpoints omit the large snapshot;
  it is available on session detail.
- Predefined participant records can carry a non-identifying, allow-listed
  profile. Role and panel-wave metadata resolve guide variants at runtime, so a
  parent is asked about their own child or children, while an educator is asked
  about their students. Broader second-hand stories remain a later module.
- The synthetic evaluation harness in `backend/evals` runs the same compiled
  interviewer and controller against parent, educator, and returning-panel
  profiles, then combines deterministic guardrails with a structured
  qualitative-research judge.

## Evidence-informed opening and rapport

OASIS treats rapport as an interview-long behavior, not as generic small talk.
The default guide-generation pattern is:

1. Give a transparent orientation: identify the AI interviewer, explain the
   study's purpose and use of responses, state privacy boundaries, and preserve
   the right to skip, pause, correct, or stop.
2. For a first interview, ask one easy question about the participant's role in
   relation to the topic and one low-stakes descriptive question about ordinary
   context. Keep the phase to roughly two to four minutes and no more than one
   follow-up per question.
3. Use one short signpost before the first substantive question. Start the core
   with a concrete experience rather than an abstract opinion.
4. For a returning panelist, replace the first-time warm-up with one
   change-since-last-time reconnection question.
5. Maintain rapport through accurate, content-grounded listening and the
   participant's own language. Do not use canned praise, repeated summaries,
   demographic fishing, or fabricated interviewer self-disclosure.

This pattern adapts established qualitative guidance on orientation,
conversational consent, easy descriptive openings, funneling, and flexible
guides ([Camden JSNA interview guidance](https://jsna.camden.gov.uk/guidance/qualitative-methods/talk-based-methods/interviews/index.html),
[topic-guide guidance](https://jsna.camden.gov.uk/guidance/qualitative-methods/talk-based-methods/topic-guides/index.html)).
Experimental human-interviewer evidence also indicates that rapport-building
can increase the amount and accuracy of information disclosed in a virtual
interview ([open-access experimental study](https://pmc.ncbi.nlm.nih.gov/articles/PMC9281184/)).
AI-interviewer research is younger, but supports socially sensitive utterance
selection and content-grounded listening over generic acknowledgements
([RaPSIL](https://aclanthology.org/2025.sigdial-1.29/),
[InterviewBot preprint](https://arxiv.org/abs/2608.10412)). The human evidence is
therefore adapted rather than copied: an AI must identify itself accurately and
must never invent shared life experience to manufacture similarity.

## Recommended workflow

1. Create the study protocol as `pilot` version `0.1`.
2. Define a small number of evidence targets and explicit `do_not_infer`
   boundaries.
3. Link each guide question to targets and state what evidence would make the
   answer useful.
4. Run internal cognitive interviews: ask testers what they thought each
   question meant, then revise wording and probes.
5. Review transcripts for leading acknowledgments, repeated questions,
   unsupported inference, missed evidence, rushed transitions, and bad closes.
6. Increment the protocol version for material changes. Promote to `field` only
   after the guide and participant information are stable enough for the first
   sample.
7. Run the checked-in synthetic profile suite for every material prompt or
   guide change, then listen to a smaller set of recorded voice interviews for
   latency, turn-taking, transcription, and TTS quality.

## Deliberately deferred

The first slice does not add automated coding, saturation claims, guide
generation, experimental branching, demographic ontologies, or a separate
planner agent. Those features should follow observed pilot failures. The next
likely additions are a post-session field-note/memo form, transcript-level
evidence coverage review, and protocol-version filters in analysis views.

Telephone interviews compile the same study context and store a snapshot, but
they cannot use the web consent checkbox. A telephone deployment must put an
appropriate spoken consent flow in the opening before treating consent as
confirmed.
