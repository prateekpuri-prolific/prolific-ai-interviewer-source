"""Design-focused interview corrections, evaluated against recorded prefixes."""
from copy import deepcopy


CONCRETE_BRIEF = """Understand one recent firsthand AI-assisted UI design or
frontend implementation attempt well enough to reconstruct its task, the inputs
the AI actually received, one observed mismatch, and the participant's own
picture of a correct result. Use the discussion guide's concrete questions as
your starting points; adapt their wording to the person's example. A visual or
interaction-design episode is fully in scope without coding in that episode.
Ordinary work descriptions and paraphrases are welcome; no confidential
artifacts, exact private prompts, credentials, or personal data are required.

Distinguish 'I gave detailed instructions' from what those instructions actually
said. Distinguish a desired style known to the participant from references or
constraints actually supplied to the AI. 'Generic', 'chaotic', 'professional',
and 'not user-friendly' are useful leads, not yet a reconstructable discrepancy.
Ask for an observable feature, action, or before/after contrast in their terms.
Do not assume a brand, screen size, failure cause, or accessibility standard.
Subjective taste is valid; do not demand numbers or technical proof for it.

Normally develop one attempt in depth and connect later corrections to the original
instructions and output. Capture one repair and its result, including any
remaining manual work. Seek a concrete check the person would use to decide the
result was ready. Follow useful detail already offered; do not repeat questions
whose purpose is satisfied or conduct a rigid questionnaire. Follow fresh details
before returning to missing background. If the participant offers a different
example, welcome it rather than insisting on the first. If a detail is
unknown or declined, accept that limit. Answer participant questions first and
honor stopping. When useful detail is exhausted, invite a final point about the
attempt and let the participant finish; do not force a second example."""


def concrete_guide(config: dict) -> dict:
    """A self-contained base guide, used as advisory questions by policy C."""
    config = reflective(config, CONCRETE_BRIEF)
    study, agent = config['study'], config['agent']
    study['research_spec']['version'] = 'frontend-ai-failures-v0.11.0'
    agent['name'] = 'Frontend AI interviewer — concrete guide v0.11.0'
    def question(key, text, purpose, evidence, probes=(), optional=False):
        return dict(id=key, text=text, purpose=purpose, evidence_goal=evidence,
                    probes=list(probes), max_follow_ups=3, depends_on=[],
                    always_ask=False, answer_requirement='any',
                    priority='optional' if optional else 'required',
                    phase='closing' if key=='closing' else 'core')
    questions = [
        question('focal_episode',
            'Think of one recent attempt to use AI on a screen, component, or visual asset for an interface. What were you trying to make or change?',
            'Anchor one firsthand attempt in an identifiable part of an interface and a design goal.',
            ['Specific screen, component, or interface asset', 'Intended change or user experience in this attempt'],
            ['Roughly when was this?', 'What did that part of the interface need to help someone do?']),
        question('tool_workflow',
            'Which AI tool were you using for that attempt?',
            'Record the tool and how it was used without guessing a model or demanding proprietary workflow details.',
            ['Tool name or explicitly unavailable', 'Chat, editor, design tool, or other workflow when known'],
            ['How were you using it in your design or development workflow?']),
        question('input_inventory',
            'In your own words, what instructions did you give the AI for that attempt? You can paraphrase the part that mattered to the design.',
            'Reconstruct the substance of instructions actually supplied, not just their length or claimed detail.',
            ['Remembered instruction or constraint actually communicated', 'Any uncertainty about the remembered wording'],
            ['What did you say about the particular detail that later went wrong?', 'Was that requirement in the first request or added after you saw the result?']),
        question('reference_context',
            'What reference material or existing project context did the AI actually receive for that attempt?',
            'Separate supplied references and their content from unshared expectations. No artifacts need be pasted.',
            ['Material supplied, or explicitly none', 'What relevant design information it contained or what remained unshared'],
            ['What did that reference show about the intended design?', 'Which parts of the intended style, if any, had not been described or shown to the AI?']),
        question('failure_location',
            'Pick one thing in the result that missed your intent. Describe what it actually looked like or did, compared with what you wanted in that same spot.',
            'Get a localized actual-versus-intended contrast that a researcher can picture.',
            ['Specific element or interaction', 'Observed appearance or behavior', 'Desired appearance or behavior at that same location'],
            ['What made you describe that part that way?', 'What was the most noticeable difference when you compared them?']),
        question('failure_conditions',
            'When you first noticed that mismatch, what were you viewing or doing in the interface?',
            'Locate the observation in a screen, interaction state, or viewing context when relevant.',
            ['Context or action in which the discrepancy was observed, or not applicable'],
            ['Was it visible throughout, or only under a particular condition?']),
        question('repair',
            'Walk me through one attempt to fix that mismatch, from what you changed to what happened in the next result.',
            'Connect a concrete correction to its observed effect, without assuming the fix succeeded.',
            ['Specific instruction change or manual edit, or no repair attempted', 'Result of that correction and any new or remaining problem'],
            ['What changed in that part after the correction?', 'Did the correction affect anything else in the interface?']),
        question('right_result',
            'What would you inspect or try in that part of the UI before you would be happy to ship it?',
            'Get the participant’s own observable acceptance check. Taste and experiential judgment count.',
            ['A concrete visual comparison, interaction check, or other way the participant would judge readiness', 'What they would expect to see or experience in that check'],
            ['What would tell you that check had passed?', 'If you kept part of the AI result, what made that part work for you?']),
        question('impact',
            'What work, if any, did you still have to do yourself before that part was usable?',
            'Capture residual work and practical effect, only when not already covered.',
            ['Remaining manual work, verification, or abandonment'],
            ['Roughly how much time did that take?'], optional=True),
        question('closing',
            'Is there anything else about this attempt you want us to understand?',
            'Invite one optional final point and respect no, uncertainty, or a wish to finish.',
            ['Final point or explicit no further detail'], optional=True),
    ]
    agent['interview_guide']['questions'] = questions
    agent['welcome_message'] = (
        "I'm an AI interviewer with the Prolific research team. We'll discuss one "
        "recent UI design or frontend attempt: what you asked AI to do, what it "
        "produced, and what getting it right would look like. Design-only examples "
        "are welcome. This takes roughly 25 minutes. You may skip or stop; please "
        "avoid confidential materials, credentials, and personal data. " + questions[0]['text']
    )
    spec = study['research_spec']
    spec['targets'] = [dict(id=q['id'], name=q['id'].replace('_',' '),
        description=q['purpose'], evidence_needed=q['evidence_goal'],
        do_not_infer=['Do not infer unshared instructions, model root cause, or population prevalence.'],
        required=q['priority']=='required') for q in questions]
    return config


def reflective(config: dict, brief: str) -> dict:
    """Opt this frontend study into the evaluated draft/critique policy."""
    config = deepcopy(config)
    study, agent = config['study'], config['agent']
    study['research_spec']['version'] = 'frontend-ai-failures-v0.10.0'
    agent['name'] = 'Frontend AI interviewer — reflective v0.10.0'
    agent['system_prompt'] = brief
    agent['interview_guide'].update(conversation_policy='reflective_v1',
                                    conversation_brief=brief)
    agent['widget_show_progress'] = False
    # Match the participant-facing estimate; the server keeps a grace period.
    agent['max_duration_seconds'] = 25 * 60
    return config


def revise(config: dict, revision: int = 1) -> dict:
    config = deepcopy(config)
    study, agent = config['study'], config['agent']
    spec = study['research_spec']
    spec['version'] = f'frontend-ai-failures-v0.9.{revision}'
    agent['name'] = f'Frontend AI interviewer — design-focused v0.9.{revision}'
    spec['population'].update(
        participant_role='Experienced UI practitioners who personally build interfaces and use AI for UI design or implementation.',
        subject='A recent firsthand UI or web design task where AI missed the intended visual or interaction result.',
        knowledge_relationship='Participants describe their own design decisions, observed results, and acceptance criteria.',
        terminology_notes='UI work includes visual composition, layout, typography, styling, animation, interaction, responsiveness, accessibility, and implementation. The chosen episode need not involve code changes.',
    )
    spec['clarification_policy'] = {
        'participant_definition': 'A firsthand AI-assisted UI or web design situation where the result missed the intended look, feel, or interaction. The particular episode need not involve coding.',
        'in_scope': ['UI or web visual design, including layouts, typography, imagery, and animation intended for an interface', 'Interaction design, responsive behavior, styling, accessibility, and UI implementation'],
        'out_of_scope': ['Backend-only infrastructure with no UI or design consequence', 'Artwork unrelated to a UI or web design task'],
        'ambiguous_cases': ['An image or animation whose intended use is unclear'],
        'ambiguity_handling': 'Accept a stated UI or web design context. Only if its connection is unclear, ask where the visual or interaction would be used. Do not require coding in the chosen example or ask again about recruitment eligibility.',
    }
    spec['participant_information']['purpose_summary'] = 'Understand concrete AI failures in UI design and implementation, including visual and interaction quality.'
    spec['interviewer_stance']['disclosure'] = 'An AI interviewer asks about one firsthand UI design or implementation experience. It does not assess the participant’s skill.'
    spec['decision_to_inform'] = 'Identify recurring AI failures in UI design and implementation with concrete inputs, observed results, and criteria for a successful result.'
    if revision >= 3:
        closing = agent['interview_guide']['questions'][-1]
        closing.update(
            max_follow_ups=1,
            purpose='Accept no further example and close. If a genuinely new lead is offered, acknowledge it and ask at most one useful concrete detail, then close.',
            evidence_goal=['Another task or explicit none', 'One concrete mismatch if a new task is offered'],
            probes=['What is one concrete thing the AI got wrong in that other task?'],
        )
        agent['system_prompt'] += (
            ' At the optional final invitation, no, none, uncertainty, and a wish to finish '
            'are valid reasons to close immediately. A no followed by a concrete example '
            'still offers a lead: interpret the entire answer. When a genuinely new task '
            'is offered, acknowledge its specific content and ask at most one focused '
            'question for a missing concrete mismatch. If that mismatch is already '
            'clear, acknowledge it and close. Do not launch another full interview '
            'or reopen the first task. After the final follow-up answer, close.'
        )
    if revision >= 4:
        closing['purpose'] += (
            ' For vague or ambiguous failures, ask what output they got versus '
            'what they wanted using only the task or artifact they explicitly named. '
            'Do not turn a vague word into a technical requirement or hypothesize '
            'an accessibility, browser, or architecture issue. If the lead is already '
            'concrete, acknowledge that specific lead before closing.'
        )
    spec['targets'] = [dict(id=q['id'], name=q['id'].replace('_', ' '),
        description=q['purpose'], evidence_needed=q['evidence_goal'],
        do_not_infer=['Do not infer population prevalence or model root cause from one account.'],
        required=q['priority']=='required') for q in agent['interview_guide']['questions']]
    return config
