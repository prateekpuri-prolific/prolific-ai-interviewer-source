from app.interview_timing import build_interview_timing


def test_fifteen_minute_policy_uses_agreed_milestones():
    policy = build_interview_timing(15 * 60)

    assert policy.target_seconds == 15 * 60
    assert policy.prioritize_at_seconds == 12 * 60
    assert policy.final_question_at_seconds == 14 * 60
    assert policy.hard_stop_seconds == 18 * 60


def test_timing_phases_do_not_hard_stop_at_research_target():
    policy = build_interview_timing(15 * 60)

    assert policy.phase(719) == "normal"
    assert policy.phase(720) == "prioritize"
    assert policy.phase(840) == "final_question"
    assert policy.phase(900) == "close_after_answer"
    assert policy.phase(1079) == "close_after_answer"
    assert policy.phase(1080) == "hard_stop"


def test_short_interviews_keep_a_bounded_grace_period():
    policy = build_interview_timing(5 * 60)

    assert policy.prioritize_at_seconds == 4 * 60
    assert policy.final_question_at_seconds == 4 * 60
    assert policy.hard_stop_seconds == 6 * 60
