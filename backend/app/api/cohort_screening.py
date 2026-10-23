"""Versioned, deterministic cohort definitions; no model-derived eligibility."""
COHORTS = ('ai_nonusers', 'older_nondegree', 'young_nonusers', 'health_nonusers', 'health_nonclaude_users', 'claude_control')
# Stable cohort IDs are retained for routing; the v2 definition is low/recent
# non-use, not lifetime abstention. Never-users remain a separate analysis group.
LOW_USE = {'ai_nonusers', 'young_nonusers', 'health_nonusers'}


def verification_questions(cohort):
    if cohort not in COHORTS:
        raise ValueError('Unknown cohort')
    questions = [
        ('country', 'Where do you currently live?', {'us': 'United States', 'uk': 'United Kingdom', 'other': 'Somewhere else', 'unsure': 'Prefer not to say'}),
        ('english', 'Are you fluent enough in English to understand and answer interview questions in English?', {'yes': 'Yes', 'no': 'No', 'unsure': 'Not sure / prefer not to say'}),
        ('age', 'Which age group are you in?', {'under18': 'Under 18', '18_29': '18–29', '30_49': '30–49', '50_64': '50–64', '65_plus': '65 or older', 'unsure': 'Prefer not to say'}),
    ]
    if cohort == 'older_nondegree':
        questions.append(('education', 'What is the highest level of education you have completed?', {
            'none': 'No formal qualifications', 'secondary': 'Secondary school / GCSEs or equivalent',
            'high_school': 'High school / A-levels or equivalent', 'below_bachelors': 'College, technical, associate or other qualification below a bachelor’s degree',
            'bachelors': 'Bachelor’s degree', 'postgraduate': 'Postgraduate degree', 'unsure': 'Not sure / prefer not to say'}))
    if cohort.startswith('health_'):
        questions.append(('health', 'Do you have a long-term health condition or disability?', {'yes': 'Yes', 'no': 'No', 'unsure': 'Not sure / prefer not to say'}))
    questions += [
        ('last_use', 'When did you last intentionally exchange messages with any AI chatbot?', {
            'never': 'Never', 'past30': 'Within the past 30 days', 'past90': '31–90 days ago', 'older': 'More than 90 days ago', 'unsure': 'Not sure / prefer not to say'}),
        ('frequency', 'Over the past 3 months, how often have you intentionally used any AI chatbot, on average? Include personal use, work, education and paid research.', {
            'none': 'Not at all in the past 3 months',
            'rare': 'Less than once a month (once or twice in total)',
            'monthly': 'About monthly', 'weekly': 'About weekly',
            'several_weekly': 'Several days a week', 'daily': 'Daily or almost daily',
            'unsure': 'Not sure / prefer not to say'}),
    ]
    return questions


CONTEXTS = {'personal': 'Personal use', 'work': 'Work or education', 'research': 'Paid research, surveys, AI evaluation or training tasks', 'none': 'I have never intentionally used a chatbot', 'unsure': 'Not sure / prefer not to say'}


def assess_cohort(config, claude, others, verification):
    """Return auditable reasons from explicit answers; uncertainty never proves eligibility."""
    cohort = config['cohort']
    fields = {key: options for key, _, options in verification_questions(cohort)}
    required = set(fields) | {'contexts'}
    if set(verification) != required:
        raise ValueError('Please answer each question shown on this questionnaire.')
    for key, options in fields.items():
        if not isinstance(verification[key], str) or verification[key] not in options:
            raise ValueError('Please select one of the answers shown for each question.')
    contexts = verification['contexts']
    if (not isinstance(contexts, list) or not contexts or any(not isinstance(v, str) or v not in CONTEXTS for v in contexts)
            or len(contexts) != len(set(contexts)) or (set(contexts) & {'none', 'unsure'} and len(contexts) != 1)):
        raise ValueError('Select the contexts that apply, or only Never / Not sure.')
    reasons = []
    if verification['country'] not in {'us', 'uk'}: reasons.append('country_not_confirmed')
    if verification['english'] != 'yes': reasons.append('english_not_confirmed')
    if verification['age'] in {'under18', 'unsure'}: reasons.append('adult_age_not_confirmed')
    if cohort == 'young_nonusers' and verification['age'] != '18_29': reasons.append('young_age_not_confirmed')
    if cohort == 'older_nondegree':
        if verification['age'] not in {'50_64', '65_plus'}: reasons.append('older_age_not_confirmed')
        if verification['education'] not in {'none', 'secondary', 'high_school', 'below_bachelors'}: reasons.append('education_not_confirmed')
    if cohort.startswith('health_') and verification['health'] != 'yes': reasons.append('health_criterion_not_confirmed')
    any_use = claude == 'yes' or bool(set(others) - {'none', 'unsure'})
    never = claude == 'no' and others == ['none']
    if any_use and (verification['last_use'] == 'never' or contexts == ['none']): reasons.append('inconsistent_use_answers')
    if never and (verification['last_use'] != 'never' or contexts != ['none'] or verification['frequency'] != 'none'): reasons.append('inconsistent_nonuse_answers')
    if verification['frequency'] not in {'none', 'unsure'} and verification['last_use'] in {'never', 'older'}: reasons.append('inconsistent_recency')
    if verification['last_use'] in {'past30', 'past90'} and verification['frequency'] == 'none': reasons.append('inconsistent_frequency')
    if cohort in LOW_USE and verification['frequency'] not in {'none', 'rare'}: reasons.append('low_chatbot_frequency_not_confirmed')
    if cohort in {'older_nondegree', 'health_nonclaude_users'} and claude != 'no': reasons.append('claude_nonuse_not_confirmed')
    if cohort == 'health_nonclaude_users' and not (set(others) - {'none', 'unsure'}): reasons.append('other_chatbot_use_not_confirmed')
    if cohort == 'claude_control' and claude != 'yes': reasons.append('claude_use_not_confirmed')
    use_group = ('never_used' if never and verification['last_use'] == 'never'
                 else 'no_recent_use' if verification['frequency'] == 'none'
                 else 'rare_use' if verification['frequency'] == 'rare'
                 else 'more_frequent_or_unconfirmed')
    return {'eligible': not reasons, 'reasons': sorted(set(reasons)), 'definition_version': 'cohort-screen-v2',
            'cohort': cohort, 'use_group': use_group, 'frequency_window_months': 3}
