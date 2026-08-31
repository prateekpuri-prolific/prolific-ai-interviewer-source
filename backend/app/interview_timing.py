"""Shared timing policy for participant interviews.

The configured duration is a research target, not a guillotine.  The
moderator starts prioritising before the target, signals one final question,
then lets the participant finish that answer.  A short technical grace period
is retained so abandoned sessions cannot remain open indefinitely.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


TimingPhase = Literal[
    "normal",
    "prioritize",
    "final_question",
    "close_after_answer",
    "hard_stop",
]


@dataclass(frozen=True)
class InterviewTimingPolicy:
    target_seconds: int
    prioritize_at_seconds: int
    final_question_at_seconds: int
    hard_stop_seconds: int

    def phase(self, elapsed_seconds: float) -> TimingPhase:
        if elapsed_seconds >= self.hard_stop_seconds:
            return "hard_stop"
        if elapsed_seconds >= self.target_seconds:
            return "close_after_answer"
        if elapsed_seconds >= self.final_question_at_seconds:
            return "final_question"
        if elapsed_seconds >= self.prioritize_at_seconds:
            return "prioritize"
        return "normal"

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def build_interview_timing(
    configured_seconds: int | None,
    *,
    fallback_seconds: int = 7200,
) -> InterviewTimingPolicy:
    """Build a proportionate policy while preserving the 15-minute UX.

    At 15 minutes this yields the agreed 12/14/15/18-minute milestones.
    Shorter and longer interviews scale the prioritisation point while keeping
    the final-question notice one minute before the target and limiting grace
    to at most three minutes.
    """

    target = max(60, int(configured_seconds or fallback_seconds))
    prioritize_at = max(0, round(target * 0.8))
    final_question_at = max(prioritize_at, target - 60)
    grace = min(180, max(60, round(target * 0.2)))
    return InterviewTimingPolicy(
        target_seconds=target,
        prioritize_at_seconds=prioritize_at,
        final_question_at_seconds=final_question_at,
        hard_stop_seconds=target + grace,
    )
