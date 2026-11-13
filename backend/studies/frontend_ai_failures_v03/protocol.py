"""Design-focused interview corrections, evaluated against recorded prefixes."""
from copy import deepcopy


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
    spec['targets'] = [dict(id=q['id'], name=q['id'].replace('_', ' '),
        description=q['purpose'], evidence_needed=q['evidence_goal'],
        do_not_infer=['Do not infer population prevalence or model root cause from one account.'],
        required=q['priority']=='required') for q in agent['interview_guide']['questions']]
    return config
