"""Offline contract tests for the synthetic interview evaluation harness."""

from agents.usage import Usage

from evals.graders import deterministic_checks
from evals.run_local import (
    DEFAULT_CASES,
    DEFAULT_CONFIG,
    compile_case,
    load_cases,
    load_json,
    participant_instructions,
)
from evals.run_suite import DEFAULT_SUITE, _load_suite


def test_agents_sdk_usage_defaults_are_compatible_with_openai_sdk():
    usage = Usage()

    assert usage.input_tokens_details.cached_tokens == 0


def test_synthetic_participant_does_not_rescue_vague_questions():
    instructions = participant_instructions({
        "id": "answerability",
        "persona": "You are concise.",
        "facts": [],
    })

    assert "Do not rescue a vague or overly broad question" in instructions
    assert "ask what the interviewer means" in instructions


def test_adversarial_participant_can_ramble_and_shift_topics_on_purpose():
    instructions = participant_instructions({
        "id": "topic_hopper",
        "persona": "You jump between topics.",
        "facts": ["One relevant fact."],
        "interaction_style": "Give long circular answers and introduce a later topic.",
    })

    assert "Adversarial interaction style" in instructions
    assert "Give long circular answers" in instructions
    assert "under 180 spoken words" in instructions
    assert "usually under 70 words" not in instructions


def test_candidate_compiles_every_role_and_wave():
    config = load_json(DEFAULT_CONFIG)
    summaries = [compile_case(config, case) for case in load_cases(DEFAULT_CASES)]

    assert len(summaries) == 5
    assert all(summary["question_count"] == 7 for summary in summaries)
    parent = next(item for item in summaries if item["case_id"].startswith("parent_two"))
    teacher = next(item for item in summaries if item["case_id"].startswith("educator_homework"))
    returning = next(item for item in summaries if item["case_id"].startswith("returning"))
    assert "what, if anything, worries you" in parent["first_question"].lower()
    assert "your students using ai" in teacher["first_question"].lower()
    assert "since we last spoke" in returning["first_question"].lower()


def test_cross_domain_suite_compiles_five_studies_and_ten_cases():
    _, studies = _load_suite(DEFAULT_SUITE, None)
    summaries = [
        compile_case(study["config"], case)
        for study in studies
        for case in study["cases"]
    ]

    assert len(studies) == 5
    assert len(summaries) == 10
    assert all(7 <= summary["question_count"] <= 8 for summary in summaries)


def test_deterministic_checks_reject_wrong_role_and_stacked_questions():
    transcript = [
        {"role": "interviewer", "content": "How are your students? What happened?"},
        {"role": "participant", "content": "I am a parent, not a teacher."},
        {"role": "interviewer", "content": "Thanks for your time."},
    ]

    result = deterministic_checks(
        transcript,
        {"role": "parent"},
        "Thanks for your time.",
    )

    assert result["passed"] is False
    assert any("asks 2 questions" in failure for failure in result["hard_failures"])
    assert "parent was addressed as an educator" in result["hard_failures"]


def test_deterministic_checks_accept_clean_role_aware_close():
    transcript = [
        {
            "role": "interviewer",
            "content": "In the past few weeks, what stood out in your students' use of AI?",
        },
        {"role": "participant", "content": "One essay included an invented quotation."},
        {"role": "interviewer", "content": "Thanks for your time."},
    ]

    result = deterministic_checks(
        transcript,
        {"role": "teacher"},
        "Thanks for your time.",
    )

    assert result["passed"] is True


def test_participant_requested_early_finish_counts_as_clean_completion():
    transcript = [
        {"role": "interviewer", "content": "What work do you use AI for?"},
        {
            "role": "participant",
            "content": "I am tired and would like to finish this interview.",
        },
        {
            "role": "interviewer",
            "content": "Your responses have been saved. Thank you for your participation.",
        },
    ]

    result = deterministic_checks(transcript, {}, "Configured closing.")

    assert result["passed"] is True
    assert result["completed"] is True


def test_recent_exchange_counts_as_primary_before_second_hand_module():
    transcript = [
        {
            "role": "interviewer",
            "content": "What happened in that recent exchange with the browser assistant?",
        },
        {"role": "participant", "content": "My teenager told me about it."},
        {
            "role": "interviewer",
            "content": "Apart from your own family, what stories have you heard?",
        },
        {"role": "participant", "content": "One parent mentioned a similar issue."},
        {"role": "interviewer", "content": "Thanks for your time."},
    ]

    result = deterministic_checks(transcript, {"role": "parent"}, "Thanks for your time.")

    assert "second-hand stories were requested before primary experience" not in result["hard_failures"]


def test_deterministic_checks_flag_lexically_near_duplicate_questions():
    transcript = [
        {"role": "interviewer", "content": "What made that shopping experience frustrating for you?"},
        {"role": "participant", "content": "The delivery date kept changing."},
        {"role": "interviewer", "content": "What about that shopping experience was frustrating for you?"},
        {"role": "participant", "content": "Mostly the changing date."},
        {"role": "interviewer", "content": "Thanks for your time."},
    ]

    result = deterministic_checks(transcript, {}, "Thanks for your time.")

    assert result["passed"] is True
    assert result["near_duplicate_questions"]
