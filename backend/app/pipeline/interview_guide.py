"""
Structured Interview Guide — System-prompt-based approach.

Instead of a complex stateful processor, we construct a detailed system
prompt that tells the LLM exactly how to conduct the structured interview.

Modern LLMs (GPT-4o, GPT-5, Gemini 2.x) follow structured protocols
remarkably well.  We layer a lightweight turn-counter on top that nudges
the model if it lingers too long on one question.

Architecture
------------
1.  `build_structured_prompt()`  – creates the full system prompt from
    the researcher's guide + their base prompt.
2.  `InterviewGuideProcessor`   – a Pipecat FrameProcessor that sits in
    the pipeline and monitors agent/user turns.  After *max_follow_ups*
    exchanges on a question it injects an LLM system message to advance.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

from loguru import logger
from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    Frame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    StartFrame,
    TextFrame,
    TranscriptionFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


# ── Clarification detection ─────────────────────────────────────────────────
#
# The hard turn-counter below is content-blind by default: it charges EVERY
# bot turn against the question's follow-up budget. In production that meant
# the agent's answers to "what do you mean?" / "sorry, I didn't get that" and
# its verbatim question repeats burned the budget, so the safety nudge fired
# while the participant had never actually answered — worst at the final
# question, where the agent repeated the question and delivered the closing
# message in the same breath.
#
# This heuristic flags user turns that are clarification/repeat requests so
# the bot's *reply* to them is not counted as a probe. Pattern sets exist for
# every language the agent form offers; detection follows the agent's
# configured ``language``. English patterns are always included as a baseline
# because participants commonly code-switch ("Sorry, das habe ich nicht
# verstanden"). For languages without a pattern set, only the English
# baseline and the generic short-question heuristic apply.

_CLARIFICATION_PATTERNS_BY_LANG: dict[str, list[str]] = {
    "en": [
        r"\bwhat do you mean\b",
        r"\bwhat does that mean\b",
        r"\bdidn'?t (get|hear|catch|understand)\b",
        r"\bdon'?t (understand|get it|follow)\b",
        r"\b(can|could) you (repeat|rephrase|clarify|explain)\b",
        r"\bsay (that|it) again\b",
        r"\bcome again\b",
        r"\bpardon\b",
        r"\bsorry\b",
        r"\bmissed (that|the question)\b",
        r"\bwhat was (that|the question)\b",
        r"\bnot sure what you('re| are) asking\b",
        r"\bnot sure (?:what|which).*you('re| are) asking\b",
        r"\bdo you mean\b",
        r"\bwhich (?:part|aspect|one|problem|example).*do you mean\b",
        r"\bcan you (?:be more specific|narrow that down)\b",
    ],
    "de": [
        r"\b(wie|was) (meinst du|meinen sie)\b",
        r"\bnicht verstanden\b",
        r"\bverstehe (das |die frage )?nicht\b",
        r"\bnicht mitbekommen\b",
        r"\bwiederholen\b",
        r"\bnoch ?mal\b",
        r"\bentschuldigung\b",
        r"\btut mir leid\b",
    ],
    "es": [
        r"\bqué (quieres|quiere) decir\b",
        r"\bno (entiendo|entendí)\b",
        r"\b(puedes|puede) repetir\b",
        r"\bperd(ón|ona|one)\b",
        r"\bdisculp[ae]\b",
        r"\bno (te |le |lo )?(escuché|oí)\b",
        r"\botra vez\b",
        r"\brepite\b",
    ],
    "fr": [
        r"\bqu'?est[- ]ce que (tu veux|vous voulez) dire\b",
        r"\bje ne comprends pas\b",
        r"\bje n'?ai pas compris\b",
        r"\b(peux-tu|pouvez-vous) (répéter|reformuler|clarifier)\b",
        r"\bpardon\b",
        r"\bdésolé\b",
        r"\bpas (entendu|compris)\b",
        r"\bencore une fois\b",
    ],
    "pt": [
        r"\bo que (você )?quer dizer\b",
        r"\bnão (entendi|entendo)\b",
        r"\bpode repetir\b",
        r"\bdesculp[ae]\b",
        r"\bperdão\b",
        r"\bnão ouvi\b",
        r"\bde novo\b",
        r"\brepetir\b",
    ],
    "nl": [
        r"\bwat bedoel(t u| je)\b",
        r"\b(ik )?begrijp (het|dat) niet\b",
        r"\bniet begrepen\b",
        r"\b(kun je|kunt u) (dat |het )?herhalen\b",
        r"\bsorry\b",
        r"\bpardon\b",
        r"\bniet verstaan\b",
        r"\bnog een keer\b",
    ],
    "it": [
        r"\bcosa (intendi|intende)\b",
        r"\bnon (capisco|ho capito)\b",
        r"\b(puoi|può) ripetere\b",
        r"\bscus[ai]\b",
        r"\bnon ho sentito\b",
        r"\bdi nuovo\b",
        r"\bripetere\b",
    ],
    # No word boundaries in CJK scripts — plain substring patterns.
    "zh": [
        r"什么意思",
        r"没听懂",
        r"没听清",
        r"听不懂",
        r"再说一遍",
        r"不明白",
        r"不好意思",
    ],
    "ja": [
        r"どういう意味",
        r"わかりません",
        r"分かりません",
        r"聞き取れません",
        r"聞こえません",
        r"もう一度",
        r"すみません",
    ],
    "ko": [
        r"무슨 뜻",
        r"이해가 안",
        r"못 들었",
        r"다시 말씀",
        r"다시 한번",
        r"죄송",
    ],
    "ar": [
        r"ماذا تقصد",
        r"لم أفهم",
        r"لا أفهم",
        r"أعد",
        r"كرر",
        r"عفوا",
        r"لم أسمع",
    ],
    "hi": [
        r"क्या मतलब",
        r"समझ नहीं",
        r"समझा नहीं",
        r"फिर से",
        r"दोबारा",
        r"सुनाई नहीं",
        r"माफ़",
    ],
}


@lru_cache(maxsize=None)
def _clarification_re(lang: str) -> re.Pattern:
    patterns = list(_CLARIFICATION_PATTERNS_BY_LANG["en"])
    if lang != "en":
        patterns += _CLARIFICATION_PATTERNS_BY_LANG.get(lang, [])
    return re.compile("|".join(patterns), re.IGNORECASE)


def strip_progress_marker(text: str) -> tuple[str, Optional[int]]:
    """Remove hidden ``[[Qn]]`` progress tags from ``text``.

    Returns ``(cleaned_text, n)`` where ``n`` is the highest question number
    seen (or ``None`` if no tag was present). Used by both the voice output
    filter and the text-chat loop so participants never see the tag.
    """
    found: Optional[int] = None

    def _sub(match: re.Match) -> str:
        nonlocal found
        try:
            n = int(match.group(1))
        except (TypeError, ValueError):
            return ""
        if found is None or n > found:
            found = n
        return ""

    cleaned = PROGRESS_MARKER_RE.sub(_sub, text)
    return cleaned, found


def looks_like_clarification(text: str, language: str = "en") -> bool:
    """True if a user turn reads as a clarification/repeat request rather
    than an answer (e.g. "sorry?", "what do you mean by that?").

    ``language`` is the agent's configured interview language ("de",
    "de-DE", "es", ...); region suffixes are ignored.
    """
    text = (text or "").strip()
    if not text:
        return False
    lang = (language or "en").split("-")[0].split("_")[0].lower()
    if _clarification_re(lang).search(text):
        return True
    if len(text.split()) <= 12 and re.search(
        r"\b(?:does|do|is|are|would)\s+.+?\s+"
        r"(?:count|qualify|apply|included|in scope)\b",
        text,
        re.IGNORECASE,
    ):
        return True
    # A question mark is not enough. Participants often answer with a
    # rhetorical or self-directed question ("Would I check every item?").
    # Treating every short question as a clarification caused adequate answers
    # to be replayed verbatim. Only the explicit, multilingual patterns above
    # are allowed to interrupt the guide.
    return False


# Participant turns that interrupt the interview flow need stronger handling
# than an ordinary clarification regex. In particular, privacy/procedure
# questions and repairs such as "you moved on before I answered" must never be
# interpreted as evidence or as permission to advance the guide.
_PROCEDURAL_RE = re.compile(
    r"\b(?:is (?:this|it) (?:being )?recorded|are you recording|raw audio|"
    r"data (?:handling|privacy|policy|storage|retention)|privacy policy|"
    r"(?:is|will|are|can) (?:all of )?(?:this|it|what i (?:say|share)|"
    r"my (?:information|answers?|responses?))(?: \w+){0,5} "
    r"(?:private(?:ly)?|confidential|anonymous|secure|protected)|"
    r"how (?:is|will|are) (?:this|it|what i (?:say|share)|my (?:information|answers?|responses?)) "
    r"(?:kept )?(?:private|confidential|anonymous|secure|protected)|"
    r"(?:is|will|are|does|do|can) (?:this |my |the )?"
    r"(?:data|answers?|responses?|recording|interview)(?: \w+){0,4} "
    r"(?:private|confidential|anonymous|secure|stored|saved|shared|deleted|retained|protected)|"
    r"where (?:is|will|does) (?:my |the )?(?:data|answers?|responses?|recording) "
    r"(?:go|get stored|live)|"
    r"what (?:service|company|provider) (?:handles|stores|transcribes|processes)|"
    r"what do you know about (?:this|the) study|"
    r"what (?:information|details) do you have about (?:this|the) study|"
    r"what (?:happens|will happen) to (?:my|the) (?:data|answers?|recording)|"
    r"how (?:is|will|are) (?:my |the )?(?:data|answers?|responses?) (?:be )?used|"
    r"(?:will|can|could|is|are) (?:this|it|what i (?:say|share)|my "
    r"(?:data|information|answers?|responses?)) (?:be )?shared with "
    r"(?:anyone|others?|someone|third parties|the research team)|"
    r"(?:will|can|could|do|does) (?:you|anyone|the research team) "
    r"(?:share|send|give|show) (?:this|it|what i (?:say|share)|my "
    r"(?:data|information|answers?|responses?))(?: to| with)?|"
    r"who (?:will|can|could|does) (?:you )?(?:share|send|give|show) "
    r"(?:this|it|my (?:data|information|answers?|responses?)) (?:to|with)|"
    r"who (?:can|will) (?:see|hear|read)|is (?:this|it) (?:anonymous|confidential)|"
    r"who (?:is running|runs|is conducting) (?:this|the) study|"
    r"how (?:can|do) i contact (?:you|the researcher|the research team)|"
    r"(?:can|how do) i (?:withdraw|delete|remove) (?:my )?(?:data|answers?|responses?)|"
    r"how long (?:is|will|does)|how much (?:longer|time)|can i (?:stop|leave|skip)|"
    r"do i have to answer|what is this (?:for|about)|what(?:'s| is) the (?:purpose|study))\b",
    re.IGNORECASE,
)
_FLOW_REPAIR_RE = re.compile(
    r"\b(?:you (?:moved|went) on|you skipped|"
    r"(?:can|could) (?:we|you) go back|let'?s go back|"
    r"go back to (?:that|your|the (?:last|previous)) question|"
    r"return to (?:that|the) question|"
    r"what about (?:the|your) (?:other|last|previous) question|"
    r"i (?:didn'?t|haven'?t) answer(?:ed)?|before (?:we|you) move on)\b",
    re.IGNORECASE,
)
_CONTINUATION_RE = re.compile(
    r"^(?:wait|hold on|one (?:second|moment)|i(?:'m| am) not done|"
    r"let me finish|i have more|let me keep going|can i continue)[.! ]*$",
    re.IGNORECASE,
)
_PROCEDURAL_RESUME_RE = re.compile(
    r"^(?:(?:yes|yeah|yep|sure|okay|ok)(?:,? that answers (?:it|my question))?|"
    r"that answers (?:it|my question)|"
    r"that(?:'s| is) clear|makes sense|i(?:'m| am) ready|ready|continue|"
    r"go ahead|let(?:'s| us) continue|no other questions?|nothing else)"
    r"[.! ]*$",
    re.IGNORECASE,
)
_PROCEDURAL_MORE_HELP_RE = re.compile(
    r"^(?:no|not really|i still (?:have|need|want)|i have another question|"
    r"can i ask (?:something|another question)|tell me more)[.! ]*$",
    re.IGNORECASE,
)
_EXPLICIT_STOP_RE = re.compile(
    r"\b(?:stop|end|finish|leave) (?:this |the )?(?:interview|session)\b|"
    r"\bi (?:want|need|would like) to (?:stop|end|finish|leave) "
    r"(?:this |the )?(?:interview|session)\b|"
    r"\bi(?:'m| am) (?:finished|done) with (?:this |the )?(?:interview|session)\b",
    re.IGNORECASE,
)
_INCOMPLETE_END_RE = re.compile(
    r"(?:\.{2,}|\b(?:and|but|because|so|when|that|if|to|the|a|an|my|i)[.!]?)$",
    re.IGNORECASE,
)


def classify_participant_turn(text: str, language: str = "en") -> str:
    """Classify a participant turn for interview-flow control.

    The classifier is intentionally conservative. It only claims an
    interruption when the wording is strong enough that treating the turn as
    an answer would be clearly wrong. Everything else remains ``answer`` and
    is left to the evidence-aware model/controller.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return "answer"
    if _EXPLICIT_STOP_RE.search(cleaned):
        return "stop"
    if _PROCEDURAL_RE.search(cleaned):
        return "procedural"
    if _FLOW_REPAIR_RE.search(cleaned):
        return "flow_repair"
    if looks_like_clarification(cleaned, language):
        return "clarification"
    if _CONTINUATION_RE.search(cleaned):
        return "continuation"
    # A short transcript ending in an obvious hanging clause is usually an
    # endpointing/STT cut-off, not a complete answer. Longer turns can end in
    # colloquial fragments legitimately, so do not second-guess those.
    if len(cleaned.split()) <= 12 and _INCOMPLETE_END_RE.search(cleaned):
        return "continuation"
    return "answer"


def is_participant_interruption(text: str, language: str = "en") -> bool:
    """Whether a turn must preserve the active interview question."""
    return classify_participant_turn(text, language) in {
        "procedural",
        "flow_repair",
        "clarification",
        "continuation",
    }


def participant_interruption_response(
    text: str,
    participant_information: Optional[dict] = None,
    current_question: str = "",
    language: str = "en",
) -> Optional[str]:
    """Return a grounded, deterministic response for high-risk interruptions.

    Scope/definition clarifications still go to the model because they require
    study-specific semantic judgment. Data-handling answers come only from
    the participant-facing protocol so the moderator cannot invent a policy.
    A procedural answer pauses the guide and asks whether the participant is
    ready; the caller must wait for that reply before resuming the question.
    """
    kind = classify_participant_turn(text, language)
    info = participant_information or {}
    answer = ""

    if kind == "continuation":
        return "Please continue—I'm listening."
    if kind == "flow_repair":
        answer = "You're right—let's go back to the question I was asking."
    elif kind == "procedural":
        lowered = (text or "").lower()
        data_summary = str(info.get("data_handling_summary") or "").strip()
        privacy_reminder = str(info.get("privacy_reminder") or "").strip()
        purpose = str(info.get("purpose_summary") or "").strip()
        researcher_identity = str(info.get("researcher_identity") or "").strip()
        researcher_contact = str(info.get("researcher_contact") or "").strip()
        complaints_contact = str(info.get("complaints_contact") or "").strip()
        ethics_reference = str(info.get("ethics_review_reference") or "").strip()
        retention = str(info.get("data_retention_period") or "").strip()
        withdrawal = str(info.get("withdrawal_instructions") or "").strip()
        risks = str(info.get("risks_summary") or "").strip()
        benefits = str(info.get("benefits_summary") or "").strip()
        compensation = str(info.get("compensation_summary") or "").strip()
        data_controller = str(info.get("data_controller") or "").strip()
        data_recipients = str(info.get("data_recipients_summary") or "").strip()
        rights = [
            str(item).strip()
            for item in (info.get("rights") or [])
            if str(item).strip()
        ]
        estimated = info.get("estimated_minutes")

        asks_retention = bool(
            re.search(
                r"\b(?:retention|how long\b.{0,50}\b(?:keep|kept|store|stored|"
                r"retain|retained|save|saved)|when\b.{0,40}\b(?:delete|deleted))\b",
                lowered,
            )
        )
        asks_recipients = bool(
            re.search(
                r"\b(?:who\b.{0,35}\b(?:receive|receives|get|gets|see|sees|"
                r"read|reads|review|reviews|access|share|shared)|who receives|"
                r"who gets|third[- ]part(?:y|ies)|processor)\b",
                lowered,
            )
        )

        # A participant may ask for more than one privacy fact in the same
        # interruption (for example, who can see a transcript *and* how long
        # it is kept).  The consent page already exposes each approved fact,
        # so answer every requested facet instead of letting the first broad
        # data-handling match hide the rest.
        if asks_retention and asks_recipients:
            details = []
            for value in (data_recipients, retention):
                if value and value not in details:
                    details.append(value)
            answer = " ".join(details) or data_summary or (
                "I don't have enough information here to state who receives "
                "the data or how long it is retained."
            )

        elif asks_retention:
            answer = retention or data_summary or (
                "I don't have enough information here to state the study's retention period."
            )
        elif any(term in lowered for term in ("delete my", "deletion", "withdraw my", "remove my")):
            answer = withdrawal or (
                "I don't have enough information here to state the withdrawal or deletion process."
            )
        elif any(term in lowered for term in ("who is running", "who runs", "who is conducting", "who are the researchers", "contact the researcher")):
            details = [value for value in (researcher_identity, researcher_contact) if value]
            answer = " ".join(details) or (
                "I don't have enough information here to identify the research team or contact route."
            )
        elif any(term in lowered for term in ("complain", "complaint", "raise a concern")):
            answer = complaints_contact or researcher_contact or (
                "I don't have enough information here to state the complaints route."
            )
        elif any(term in lowered for term in ("ethics", "irb", "review board", "approved")):
            answer = ethics_reference or (
                "I don't have an ethics-review reference available for this study."
            )
        elif any(term in lowered for term in ("risk", "harm", "uncomfortable", "discomfort")):
            answer = risks or (
                "I don't have a more specific risk statement available."
            )
        elif any(term in lowered for term in ("benefit", "help me")):
            answer = benefits or (
                "I don't have a more specific benefits statement available."
            )
        elif any(term in lowered for term in ("pay", "paid", "payment", "compensation", "reward")):
            answer = compensation or (
                "I don't have the compensation details available here."
            )
        elif any(term in lowered for term in ("data controller", "responsible for my data")):
            answer = data_controller or (
                "I don't have the data-controller identity available here."
            )
        elif asks_recipients:
            answer = data_recipients or data_summary or (
                "I don't have enough information here to state who receives the data."
            )
        elif any(
            term in lowered
            for term in (
                "record",
                "audio",
                "data",
                "see",
                "hear",
                "read",
                "anonymous",
                "confidential",
                "private",
                "privacy",
                "protect",
                "secure",
                "store",
                "share",
            )
        ):
            answer = data_summary or (
                "I don't have enough information here to state the study's "
                "data-handling policy."
            )
            if (
                privacy_reminder
                and any(
                    term in lowered
                    for term in ("private", "privacy", "confidential", "personal")
                )
                and privacy_reminder.rstrip(".!").lower()
                not in answer.rstrip(".!").lower()
            ):
                answer = f"{answer} {privacy_reminder}"
        elif any(
            term in lowered
            for term in (
                "purpose",
                "what is this for",
                "what's this for",
                "what is this about",
            )
        ):
            answer = purpose or (
                "I don't have a more specific purpose statement available."
            )
        elif any(
            term in lowered
            for term in ("how long", "how much longer", "how much time")
        ) and estimated:
            answer = (
                f"The interview is estimated to take about {estimated} "
                "minutes, and you can finish early."
            )
        elif any(
            term in lowered
            for term in ("stop", "leave", "skip", "have to answer")
        ):
            answer = rights[0] if rights else (
                "You can skip a question or finish the interview at any time."
            )
        else:
            details = [value for value in (purpose, data_summary) if value]
            answer = " ".join(details) or (
                "I don't have that study detail available."
            )

    if not answer:
        return None
    if kind == "procedural":
        return (
            f"{answer} Would you like to continue, or is there anything else "
            "you'd like to ask first?"
        )
    question = (current_question or "").strip()
    if question:
        return f"{answer} When you're ready, let's return to this: {question}"
    return answer


def looks_like_procedural_resume(text: str) -> bool:
    """Whether a participant explicitly says the paused interview may resume."""
    return bool(_PROCEDURAL_RESUME_RE.search((text or "").strip()))


def looks_like_more_procedural_help(text: str) -> bool:
    """Whether a participant says the procedural answer was not sufficient."""
    return bool(_PROCEDURAL_MORE_HELP_RE.search((text or "").strip()))


def procedural_resume_response(current_question: str = "") -> str:
    """Resume the exact question suspended by a procedural interruption."""
    question = (current_question or "").strip()
    if question:
        return f"Okay—let's continue. {question}"
    return "Okay—let's continue."


# ── Progress marker ─────────────────────────────────────────────────────────
#
# When the progress bar is enabled, the model prefixes each main question with a
# hidden tag like ``[[Q3]]``. The tag never reaches the participant: the voice
# path strips it in StructuredOutputFilter, the text path strips it before
# sending. Probes, clarifications, and the closing carry no tag, so the bar only
# advances on main questions. A missed tag is harmless: the bar holds until the
# next tagged question.

PROGRESS_MARKER_RE = re.compile(r"\[\[\s*Q\s*(\d+)\s*\]\]", re.IGNORECASE)
TURN_MARKER_RE = re.compile(
    r"\[\[\s*TURN\s+(\{.*?\})\s*\]\]", re.IGNORECASE | re.DOTALL
)
# Models occasionally emit a valid one-line JSON marker but omit its final
# ``]]`` when the resolution map is long. Parse that conservatively and always
# scrub any leftover TURN-prefixed line so private orchestration data can never
# reach a participant or transcript.
TURN_MARKER_FALLBACK_RE = re.compile(
    r"\[\[\s*TURN\s+(\{[^\r\n]*\})(?:\s*\]\])?", re.IGNORECASE
)
TURN_MARKER_LEAK_RE = re.compile(
    r"\[\[\s*TURN\b[^\r\n]*(?:\r?\n|$)", re.IGNORECASE
)
_INTENT_COMPONENT_RE = re.compile(r"[^a-z0-9_.:-]+")
_RESOLUTION_STATUSES = {
    "covered",
    "partial",
    "unknown",
    "declined",
    "not_applicable",
}
_TURN_ACTIONS = {"main", "probe", "thread", "clarify", "close"}

# Exploratory interviews may adapt a listed probe to the participant's exact
# detail.  Keep a small deterministic guard for conspicuously leading wording;
# the model prompt remains responsible for the broader neutrality policy.
_LEADING_PROBE_RE = re.compile(
    r"\b(?:obviously|surely|clearly|of course|you must|don't you agree|"
    r"wouldn't you|isn't it true)\b",
    re.IGNORECASE,
)
_PROBE_GROUNDING_STOPWORDS = {
    "about",
    "after",
    "again",
    "could",
    "did",
    "does",
    "experience",
    "from",
    "have",
    "that",
    "their",
    "then",
    "there",
    "they",
    "this",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "would",
    "your",
}
_PROVENANCE_VALUES = {"direct", "second_hand", "opinion", "mixed", "unknown"}
_TERMINAL_STATUS_PATTERNS = {
    "unknown": re.compile(
        r"\b(don'?t know|do not know|not sure|unsure|can'?t (?:know|tell|remember)|"
        r"cannot (?:know|tell|remember)|no idea|don'?t remember|couldn'?t tell)\b",
        re.IGNORECASE,
    ),
    "declined": re.compile(
        r"\b(prefer not|rather not|don'?t want to (?:say|share|answer)|"
        r"do not want to (?:say|share|answer)|not comfortable|skip (?:that|this))\b",
        re.IGNORECASE,
    ),
    "not_applicable": re.compile(
        r"^\s*(?:none|nothing|nope|no)\s*[.!]?\s*$|"
        r"\b(not applicable|doesn'?t apply|does not apply|never (?:used|tried|had)|"
        r"haven'?t (?:used|tried|had)|have not (?:used|tried|had)|"
        r"don'?t use|do not use|nothing to compare|no comparison|"
        r"don'?t have any (?:concerns?|worries)|do not have any (?:concerns?|worries)|"
        r"have no (?:concerns?|worries)|not worried|"
        r"no (?:real |particular |specific )?(?:concerns?|worries)|"
        r"nothing (?:really )?(?:worries|concerns) me|"
        r"nothing (?:comes|came) to mind|can'?t think of (?:anything|an example)|"
        r"cannot think of (?:anything|an example)|no examples?)\b",
        re.IGNORECASE,
    ),
}


def conditional_module_skip_index(
    questions: list[dict],
    decision: Optional["TurnDecision"],
    current_index: int,
) -> Optional[tuple[int, int]]:
    """Validate an LLM applicability decision against the authored guide.

    Returns ``(answered_index, next_index)`` only when the model identified an
    opt-in question as not applicable. The server, not the model, owns the
    scope of the skip: only consecutive cards sharing that module are skipped.
    """
    if decision is None or decision.not_applicable_question is None:
        return None
    answered_index = decision.not_applicable_question - 1
    if not (0 <= answered_index < len(questions)):
        return None
    if answered_index not in {current_index, current_index - 1}:
        return None
    question = questions[answered_index]
    if not question.get("skip_remaining_module_if_not_applicable"):
        return None
    module = str(question.get("module") or "").strip()
    if not module:
        return None

    next_index = answered_index + 1
    while next_index < len(questions):
        next_module = str(questions[next_index].get("module") or "").strip()
        if next_module != module:
            break
        next_index += 1
    return answered_index, next_index


@dataclass(frozen=True)
class TurnDecision:
    """Hidden, model-authored account of the evidence state for one turn."""

    action: str
    question: Optional[int] = None
    intent: Optional[str] = None
    resolved: dict[str, str] = field(default_factory=dict)
    focal_entity: Optional[str] = None
    provenance: Optional[str] = None
    not_applicable_question: Optional[int] = None


@dataclass(frozen=True)
class TurnReview:
    """Controller-reviewed visible turn and its hidden control metadata."""

    text: str
    marker: Optional[int]
    decision: Optional[TurnDecision]
    replaced: bool = False
    reason: Optional[str] = None
    question_index: Optional[int] = None
    closed: bool = False


def _intent_component(value: object, fallback: str) -> str:
    cleaned = _INTENT_COMPONENT_RE.sub("_", str(value or "").strip().lower())
    return cleaned.strip("_") or fallback


def question_intents(question: dict, index: int) -> dict[str, object]:
    """Return stable IDs used by the model and controller for one question."""
    question_id = _intent_component(question.get("id"), f"q{index + 1}")
    evidence = {
        f"{question_id}:e{goal_index}": str(goal).strip()
        for goal_index, goal in enumerate(question.get("evidence_goal") or [], 1)
        if str(goal).strip()
    }
    probes = {
        f"{question_id}:p{probe_index}": str(probe).strip()
        for probe_index, probe in enumerate(question.get("probes") or [], 1)
        if str(probe).strip()
    }
    emergent = {
        f"{question_id}:emergent:{kind}"
        for kind in (
            "clarification",
            "description",
            "meaning",
            "consequence",
            "contrast",
            "source",
        )
    }
    return {
        "question_id": question_id,
        "main": f"{question_id}:main",
        "evidence": evidence,
        "probes": probes,
        "emergent": emergent,
    }


def evidence_skipping_enabled(guide: dict, allow_follow_ups: bool) -> bool:
    """Resolve the independent evidence-skip setting with legacy parity."""
    configured = guide.get("skip_answered_questions")
    if configured is None:
        return allow_follow_ups
    return bool(configured)


def resolve_orchestration_settings(guide: dict) -> dict[str, object]:
    """Resolve one coherent interview policy from a stored guide.

    New guides name a profile explicitly. Legacy guides are migrated in
    memory. In particular, the previously common but contradictory
    combination "follow-ups enabled, zero capacity everywhere" becomes
    ``repair_only`` instead of generating a forced-advance nudge every turn.
    """
    guide = guide or {}
    profile = str(guide.get("orchestration_profile") or "").strip().lower()
    profile_rules = {
        "fixed": (False, False, False),
        "repair_only": (False, True, False),
        "semi_structured": (True, True, False),
        "exploratory": (True, True, True),
    }
    if profile not in profile_rules:
        legacy_follow_ups = guide.get("allow_follow_ups", True) is not False
        legacy_skip = evidence_skipping_enabled(guide, legacy_follow_ups)
        legacy_emergent = bool(guide.get("allow_emergent_follow_ups", False))
        has_follow_up_capacity = any(
            int(question.get("max_follow_ups", DEFAULT_MAX_FOLLOW_UPS) or 0) > 0
            and (
                bool(question.get("probes"))
                or legacy_emergent
                or "max_follow_ups" not in question
            )
            for question in guide.get("questions") or []
        )
        if not legacy_follow_ups or not has_follow_up_capacity:
            profile = "repair_only" if legacy_skip else "fixed"
        elif legacy_emergent:
            profile = "exploratory"
        else:
            profile = "semi_structured"
    allow_follow_ups, skip_answered, allow_emergent = profile_rules[profile]
    return {
        "profile": profile,
        "allow_follow_ups": allow_follow_ups,
        "skip_answered_questions": skip_answered,
        "allow_emergent_follow_ups": allow_emergent,
        "allow_coverage_clarifications": profile == "repair_only",
        "allow_threading": profile in {"semi_structured", "exploratory"},
    }


def parse_turn_decision(payload: str) -> Optional[TurnDecision]:
    """Parse and conservatively validate a hidden ``[[TURN {...}]]`` value."""
    try:
        raw = json.loads(payload)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    action = str(raw.get("action") or "").strip().lower()
    if action not in _TURN_ACTIONS:
        return None
    question = raw.get("question")
    if question is not None:
        try:
            question = int(question)
        except (TypeError, ValueError):
            return None
        if question < 1:
            return None
    intent = raw.get("intent")
    if intent is not None:
        intent = str(intent).strip() or None
    resolved: dict[str, str] = {}
    if isinstance(raw.get("resolved"), dict):
        for evidence_id, status in raw["resolved"].items():
            normalized = str(status).strip().lower()
            if normalized in _RESOLUTION_STATUSES:
                resolved[str(evidence_id).strip()] = normalized
    focal_entity = raw.get("focal_entity")
    if focal_entity is not None:
        focal_entity = str(focal_entity).strip()[:80] or None
    provenance = raw.get("provenance")
    if provenance is not None:
        provenance = str(provenance).strip().lower()
        if provenance not in _PROVENANCE_VALUES:
            provenance = None
    not_applicable_question = raw.get("not_applicable_question")
    if not_applicable_question is not None:
        try:
            not_applicable_question = int(not_applicable_question)
        except (TypeError, ValueError):
            return None
        if not_applicable_question < 1:
            return None
    return TurnDecision(
        action=action,
        question=question,
        intent=intent,
        resolved=resolved,
        focal_entity=focal_entity,
        provenance=provenance,
        not_applicable_question=not_applicable_question,
    )


def strip_turn_marker(text: str) -> tuple[str, Optional[TurnDecision]]:
    """Remove hidden turn-state markers and return the last valid decision."""
    decision: Optional[TurnDecision] = None

    def _sub(match: re.Match) -> str:
        nonlocal decision
        parsed = parse_turn_decision(match.group(1))
        if parsed is not None:
            decision = parsed
        return ""

    cleaned = TURN_MARKER_RE.sub(_sub, text)
    if decision is None:
        cleaned = TURN_MARKER_FALLBACK_RE.sub(_sub, cleaned)
    # Defense in depth: malformed or truncated private state is removed even
    # when it cannot be parsed into a decision.
    cleaned = TURN_MARKER_LEAK_RE.sub("", cleaned)
    return cleaned, decision


def strip_control_markers(
    text: str,
) -> tuple[str, Optional[int], Optional[TurnDecision]]:
    """Strip every model control tag before participant display or speech."""
    cleaned, decision = strip_turn_marker(text)
    cleaned, marker = strip_progress_marker(cleaned)
    return cleaned, marker, decision


def format_turn_marker(decision: TurnDecision) -> str:
    """Serialize a reviewed decision for the model-only conversation history."""
    payload: dict[str, object] = {
        "action": decision.action,
        "question": decision.question,
        "intent": decision.intent,
        "resolved": decision.resolved,
    }
    if decision.focal_entity:
        payload["focal_entity"] = decision.focal_entity
    if decision.provenance:
        payload["provenance"] = decision.provenance
    if decision.not_applicable_question is not None:
        payload["not_applicable_question"] = decision.not_applicable_question
    return "[[TURN " + json.dumps(payload, separators=(",", ":")) + "]]"


def control_marker_prefix(
    marker: Optional[int], decision: Optional[TurnDecision]
) -> str:
    """Build the hidden prefix retained in assistant context, never display."""
    parts = []
    if decision is not None:
        parts.append(format_turn_marker(decision))
    if marker is not None:
        parts.append(f"[[Q{marker}]]")
    return " ".join(parts)


def progress_marker_instruction(total: int) -> str:
    """Prompt fragment asking the model to emit the hidden progress tag."""
    return (
        "\n\n### Progress marker (hidden, required on EVERY main question)\n"
        f"This interview has {total} main questions, numbered 1 to {total}. "
        "Every single time you ask a MAIN question, begin that message with a "
        "hidden tag of the exact form [[Qn]], where n is the number of that "
        "main question. Your first main question starts with [[Q1]], your "
        f"second main question starts with [[Q2]], and so on up to [[Q{total}]]. "
        "This is not a one-time thing: tag the second, third, and every later "
        "main question too, not just the first. Do NOT add the tag to follow-up "
        "probes, clarification answers, acknowledgements, or the closing "
        "message. Do NOT add the tag to an opportunistic downstream thread. "
        "The tag is removed automatically before the participant sees "
        "or hears it, so never mention it, never read it aloud, and never "
        "explain it. Put nothing before the tag."
    )


def turn_decision_instruction(
    questions: list[dict],
    allow_follow_ups: bool = True,
    skip_answered_questions: bool = True,
) -> str:
    """Prompt fragment for same-call evidence state, hidden from participants."""
    conditional_questions = [
        index + 1
        for index, question in enumerate(questions)
        if question.get("skip_remaining_module_if_not_applicable")
    ]
    applicability_rule = (
        "The optional `not_applicable_question` field classifies the "
        "participant's latest answer. Set it to one of these question numbers "
        f"only: {conditional_questions}. Set it when the participant "
        "semantically indicates that no applicable example, task, or answer "
        "exists for the question they just answered. Understand natural "
        "paraphrases; do not require a literal word such as `none`. Otherwise "
        "set it to null. Never set it for an unasked question. The server "
        "validates this field and controls the scope of any skip."
        if conditional_questions
        else "Always set `not_applicable_question` to null; conditional module "
        "skipping is not enabled for any question in this guide."
    )
    if not allow_follow_ups and not skip_answered_questions:
        catalog = "\n".join(
            f"- Question {index + 1} main intent: "
            f"{question_intents(question, index)['main']}"
            for index, question in enumerate(questions)
        )
        return f"""

### Turn decision marker (hidden, required on EVERY response)
Before the visible words of every response, emit exactly one compact JSON
marker on one line:
[[TURN {{"action":"main|clarify|close","question":2,"intent":"a main intent ID below or null","resolved":{{}},"focal_entity":null,"provenance":"unknown","not_applicable_question":null}}]]

This marker is private control data and is automatically removed. Use
`action="main"` only for the next predefined main question, in guide order.
Use `action="clarify"` only when answering a participant's procedural or scope
question. After the answer, ask whether they would like to continue or have
another question, then STOP. Resume the pending predefined main question only
after they confirm. Never use `probe`
or `thread`, never generate a new content question, and never skip a main
question because its topic appeared in an earlier answer. Use `close` only
after the participant answers the final predefined question.

{applicability_rule}

Allowed main intent IDs:
{catalog}
"""
    if not allow_follow_ups:
        catalog_lines: list[str] = []
        for index, question in enumerate(questions):
            intents = question_intents(question, index)
            protected = " (protected; always ask)" if question.get("always_ask") else ""
            catalog_lines.append(
                f"- Question {index + 1} main intent: {intents['main']}{protected}"
            )
            for evidence_id, goal in intents["evidence"].items():
                catalog_lines.append(f"  Evidence {evidence_id}: {goal}")
            catalog_lines.append(
                "  Coverage clarification intent: "
                f"{intents['question_id']}:emergent:clarification"
            )
        catalog = "\n".join(catalog_lines)
        return f"""

### Turn decision marker (hidden, required on EVERY response)
Before the visible words of every response, emit exactly one compact JSON
marker on one line:
[[TURN {{"action":"main|probe|clarify|close","question":2,"intent":"an allowed intent ID below or null","resolved":{{"evidence_id":"covered|partial|unknown|declined|not_applicable"}},"focal_entity":"optional short non-identifying task label","provenance":"direct|second_hand|opinion|mixed|unknown","not_applicable_question":null}}]]

This marker is private control data and is automatically removed. Follow-up
probing is disabled, but evidence-aware skipping is enabled. Report explicit
evidence from the participant's latest answer across the full guide. Mark an
evidence item `covered` only when the answer supplies concrete, analyzable
detail; a mention or inference is only `partial`. A future question may be
skipped only when EVERY one of its evidence items is `covered`. Questions
marked protected must always be asked, even if their evidence appeared early.

Before choosing the next question, perform a mandatory coverage audit of the
participant's latest answer against EVERY evidence ID below. Record every
explicit answer even when it belongs to a later card. In particular, when the
participant names an AI product, model, input type, observed error, workaround,
or verification rule, you MUST resolve the matching evidence ID before choosing
visible wording. Never ask a broad main question when the latest answer already
contains any of that card's evidence; use one narrow coverage clarification for
the missing item instead. If every item is covered, skip the card.

Use `action="main"` for the first required question in guide order. When a
future unprotected question is only partially answered, you may use
`action="probe"` once with that question's coverage-clarification intent and
ask one short question for the single most important missing detail. This is
evidence repair, not permission for exploratory or repeated follow-ups. Name
the relevant task in the visible wording whenever prior coverage causes you
to move past one or more cards. Use `action="clarify"` only for a participant's
procedural or scope interruption. Answer briefly, ask whether they would like
to continue or have another question, and STOP; do not resume the guide in the
same turn. Use `close` only when no required card remains.

{applicability_rule}

Allowed intents and evidence IDs:
{catalog}
"""
    catalog_lines: list[str] = []
    for index, question in enumerate(questions):
        intents = question_intents(question, index)
        protected = " (protected; always ask)" if question.get("always_ask") else ""
        catalog_lines.append(
            f"- Question {index + 1} main intent: {intents['main']}{protected}"
        )
        for evidence_id, goal in intents["evidence"].items():
            catalog_lines.append(f"  Evidence {evidence_id}: {goal}")
        for probe_id, probe in intents["probes"].items():
            catalog_lines.append(f"  Listed probe {probe_id}: {probe}")
        catalog_lines.append(
            "  Emergent probe IDs: " + ", ".join(sorted(intents["emergent"]))
        )

    catalog = "\n".join(catalog_lines)
    skip_policy_rule = (
        "Evidence-aware skipping is enabled. Skip only unprotected questions "
        "whose complete evidence set is concretely `covered`; always ask a "
        "question marked protected. Name the focal task when a skip would "
        "otherwise leave a pronoun or reference ambiguous."
        if skip_answered_questions
        else "Evidence-aware skipping is disabled. Evidence may guide probes, "
        "but every predefined main question must still be asked in order."
    )
    return f"""

### Turn decision marker (hidden, required on EVERY response)
Before the visible words of every response, emit exactly one compact JSON
marker on one line:
[[TURN {{"action":"main|probe|thread|clarify|close","question":2,"intent":"an ID below or null","resolved":{{"evidence_id":"covered|partial|unknown|declined|not_applicable"}},"focal_entity":"optional non-identifying label","provenance":"direct|second_hand|opinion|mixed|unknown","not_applicable_question":null}}]]

This marker is private control data and is automatically removed before the
participant sees or hears the response. Put nothing before it, never mention
it, and do not wrap it in Markdown. Use strict JSON with double quotes.

{applicability_rule}

{skip_policy_rule}

Decision rules:
- `question` is the 1-based guide question your visible response belongs to.
  Use null only for the closing. `intent` is the one ID your visible question
  seeks. Use null for a clarification answer or closing.
- In `resolved`, report evidence that the participant has already supplied
  BEFORE the new visible question. Review the entire conversation, including
  answers that covered a later question early. Report only explicit evidence;
  do not infer it from demographics or silence.
- Resolution comes BEFORE turn selection. First scan the participant's latest
  answer against the evidence IDs for EVERY guide question, including the
  opening and questions not yet asked. Put every explicit match from that
  answer in `resolved` as `partial` or `covered`. Only then decide what to ask.
  This is essential when a participant anticipates several questions, starts
  telling their story during the readiness exchange, or gives one long answer.
  `resolved` may therefore contain IDs from several different questions.
- Every `resolved` key must be one of the allowed Evidence IDs below (the IDs
  ending in `:e1`, `:e2`, and so on). Never put a main intent, probe intent,
  research target ID, evidence description, or newly invented ID there. For
  example: `"resolved":{{"recent_event:e1":"covered"}}`.
- Keep `resolved` sparse. Omit evidence that has not been mentioned; do not
  inventory every unasked goal. Sparse does NOT mean omitting supported future
  evidence: include every ID explicitly supported by the latest answer. Before
  proposing a main question or probe, also audit the evidence IDs belonging to
  that question against the whole transcript and include any already supported.
- `covered` means the participant supplied concrete, analyzable evidence.
  `partial` means they mentioned it but a useful detail, basis, sequence,
  consequence, or boundary is still missing; partial evidence does not resolve
  the goal and can justify one focused probe. `unknown` means they
  explicitly cannot know or remember. `declined` means they chose not to
  answer. `not_applicable` means the premise genuinely does not apply. All four
  resolve that evidence need; do not keep asking it in different words.
- Mention is not coverage. When an answer names an outcome, comparison,
  workaround, reason, tradeoff, or desired change without enough basis or
  detail to analyze it, mark that evidence `partial`, not `covered`.
- Missing or unmentioned evidence is unresolved, NEVER `unknown`. Use
  `unknown`, `declined`, or `not_applicable` only when the participant just
  said that explicitly, and only for evidence belonging to the question they
  were answering—not for unasked future questions.
- If all evidence needs for the current question are resolved, advance. If a
  later main question's evidence was already supplied, skip that question and
  continue to the next unresolved one, preserving guide order among the
  remaining questions.
- Choose the next visible question only AFTER updating evidence from the latest
  answer across the entire guide. Find the first unresolved question in guide
  order, then use this strict action rule:
  1. Use `action="main"` only when the participant has supplied NO evidence for
     that question yet.
  2. If any evidence for that question is already `partial` or `covered` but
     other evidence is still missing, use `action="probe"` and ask narrowly for
     one missing need.
  3. If all its evidence is resolved, skip it before composing visible text.
  Never replay a broad main question as a transition or recap. A decision that
  marks evidence for Question N covered while also choosing Question N's main
  wording is internally inconsistent and forbidden.
- Example: the participant answers Question 1 and spontaneously gives the
  incident requested by Question 2. Record Question 2's evidence immediately.
  Do not ask Question 2's main wording. If one detail is still missing, ask one
  Question 2 probe for that detail; otherwise continue to the next unresolved
  card.
- Default to the guide order. When the participant spontaneously introduces a
  concrete, high-information detail that belongs to a later question, you may
  ask exactly ONE short opportunistic follow-up now with `action="thread"`.
  Set `question` and `intent` to that later question and one of its listed or
  emergent probe IDs. Also mark at least one evidence ID from that later
  question as `partial` or `covered` in `resolved`; without that answer-grounded
  evidence, do not thread.
- Thread only when the follow-up depends on the participant's immediately
  preceding words and postponing it would noticeably break conversational
  momentum. A vague mention, a list of topics, a long unfocused answer, or the
  mere availability of a later probe is not enough. When an answer touches
  many later topics, record the evidence but choose at most the single most
  concrete and decision-relevant thread—or ask none.
- A thread is a narrow probe, never the later main question. Do not thread into
  an elevated-sensitivity question or a closing question. Do not emit [[Qn]]
  for a thread, do not chain one thread into another, and do not use threading
  to bypass unresolved guide questions.
- Example: while on Question 2, the participant says they switched products
  and that the workaround took longer. If those details partially support
  Question 5, a natural immediate follow-up uses `action="thread"`,
  `question=5`, a Question 5 probe/emergent intent, and the matching Question 5
  evidence IDs in `resolved`. Do NOT label that narrow follow-up as
  `action="main"`, and do not repeat Question 5's broad main wording.
- After the participant answers a thread, record any downstream evidence they
  supplied, then return to the active question or advance to the first
  unresolved question in guide order. Do not pursue a new detour. When the
  threaded question is eventually reached, ask only for evidence still
  missing; skip it if all of its evidence is already resolved.
- When the first unresolved question in guide order is already `partial`, do
  not ask its broad main wording. Advance with one narrow missing-evidence
  question using `action="probe"`, that question's number, and one of its
  probe/emergent intent IDs. This is an in-order focused advance, not a second
  thread. It is the preferred way to resume after a thread or a rambling answer.
- If one unusually rich early answer appears to cover most of the interview,
  do not replay the broad main questions—but also do not jump straight to the
  participant-led closing. Ask one focused `thread` about the most important
  uncertainty, basis, consequence, or tradeoff that would materially deepen
  the evidence. After that answer, skip everything genuinely resolved and
  continue or close. This is a depth check, not permission to unpack every
  detail the participant volunteered.
- That one depth-check `thread` is the sole exception to the normal rule that
  resolved cards are skipped: a coarse evidence checklist can be complete
  while the basis or meaning still deserves one natural question. It is
  allowed only when one substantive answer resolved most non-closing cards,
  the question uses a narrow non-main intent grounded in an exact detail, and
  no earlier thread was used from this anchor. Unless the participant asks to
  stop or declines, do not close a rich-episode interview after only one
  substantive answer without this depth check.
- In that depth check, prefer understanding the concrete episode, source of
  knowledge, decision point, consequence, or durable behavior change before
  elaborating an unsolicited solution or feature request. A participant's
  early product idea must not replace understanding the experience that
  produced it. Ground the question in one of their exact details when doing so
  remains concise and neutral.
- Never reuse an intent ID. For an emergent follow-up, choose the closest
  question-specific emergent ID. Reusing that type on the same question is a
  signal to advance instead.
- `focal_entity` is a short non-identifying referent such as `older_child`,
  `current_tool`, or `last_purchase`; keep it stable until the participant
  changes the subject. `provenance` describes the evidence in their latest
  answer, not the truth of their claim.

Allowed intent and evidence IDs:
{catalog}
"""


# ── Prompt builder ──────────────────────────────────────────────────────────


def resolve_role_guide(guide: dict, participant_context: dict | None) -> dict:
    """Resolve role- and panel-wave-specific guide wording.

    A question may define ``role_variants`` and
    ``returning_role_variants``. Each variant can be a replacement string for
    ``text`` or a dict overriding ``text``, ``probes``, and other question
    metadata. The stored guide remains unchanged for auditability.
    """
    from app.research.protocol import normalize_participant_context

    resolved = deepcopy(guide or {})
    context = normalize_participant_context(participant_context)
    role = context.get("role", "other")
    try:
        returning = int(context.get("wave", 1)) > 1
    except (TypeError, ValueError):
        returning = False

    def choose_variant(question: dict):
        variant_sets = []
        if returning:
            variant_sets.append(question.get("returning_role_variants") or {})
        variant_sets.append(question.get("role_variants") or {})
        for variants in variant_sets:
            if not isinstance(variants, dict):
                continue
            if role in variants:
                return variants[role]
            if "default" in variants:
                return variants["default"]
        return None

    applicable_questions = []
    for question in resolved.get("questions") or []:
        visit_scope = question.get("visit_scope") or "all"
        if visit_scope == "first_time" and returning:
            continue
        if visit_scope == "returning" and not returning:
            continue
        variant = choose_variant(question)
        if isinstance(variant, str):
            question["text"] = variant
        elif isinstance(variant, dict):
            for key, value in variant.items():
                question[key] = deepcopy(value)
        applicable_questions.append(question)
    resolved["questions"] = applicable_questions

    closing_variants = resolved.get("closing_role_variants") or {}
    if isinstance(closing_variants, dict):
        closing = closing_variants.get(role) or closing_variants.get("default")
        if closing:
            resolved["closing_message"] = str(closing)
    return resolved

def build_structured_prompt(
    base_prompt: str,
    guide: dict,
    emit_progress: bool = False,
) -> str:
    """
    Build a system prompt that encodes the interview guide.

    Parameters
    ----------
    base_prompt : str
        The researcher's base system prompt (personality, context, etc.).
    guide : dict
        The interview_guide JSON from the Agent model.
        Expected shape:
        {
          "questions": [
            {
              "text": "...",
              "probes": ["...", "..."],
              "max_follow_ups": 3,
              "transition": "..."        # optional
            },
            ...
          ],
          "closing_message": "..."        # optional
        }

    Returns
    -------
    str
        Full system prompt with the interview protocol embedded.
    """
    questions = guide.get("questions", [])
    closing = guide.get(
        "closing_message",
        "Thank you for your time. This concludes our interview.",
    )
    # How to pick probes. "ordered" walks the list top to bottom; "relevance"
    # picks the most relevant one per answer. Same follow-up cap either way,
    # and no extra latency since the model already has the whole list.
    relevance = (guide.get("probe_selection") or "ordered").lower() == "relevance"
    orchestration = resolve_orchestration_settings(guide)
    orchestration_profile = str(orchestration["profile"])
    allow_follow_ups = bool(orchestration["allow_follow_ups"])
    skip_answered_questions = bool(orchestration["skip_answered_questions"])
    allow_emergent = bool(orchestration["allow_emergent_follow_ups"])

    if not questions:
        return base_prompt

    # Build the question guide section: each question with its probe list and
    # a hard follow-up cap. The model is told NEVER to reuse a probe. This
    # stops the "could you tell me about your background... could you tell me
    # about your background..." repetition we saw in production logs.
    guide_lines = []
    for i, q in enumerate(questions, 1):
        text = q.get("text", "")
        probes = q.get("probes", []) if allow_follow_ups else []
        max_fu = q.get("max_follow_ups", 2) if allow_follow_ups else 0
        transition = q.get("transition", "")
        intents = question_intents(q, i - 1)

        guide_lines.append(f"### Question {i} of {len(questions)}")
        guide_lines.append(f"Main question: {text}")
        guide_lines.append(f"Main intent ID: {intents['main']}")
        if q.get("module"):
            guide_lines.append(f"Module: {q['module']}")
        if q.get("skip_remaining_module_if_not_applicable"):
            guide_lines.append(
                "Applicability rule: if the participant explicitly answers "
                "none or says this does not apply, skip the remaining "
                "consecutive questions in this module."
            )
        if q.get("always_ask"):
            guide_lines.append(
                "Evidence-skip rule: protected anchor; always ask this main "
                "question even when its evidence appeared earlier."
            )
        if q.get("depends_on"):
            guide_lines.append(
                "Sequence dependency (server-enforced): ask only after these "
                "question IDs have been reached: " + ", ".join(q["depends_on"])
            )
        guide_lines.append(f"Interview phase: {q.get('phase', 'core')}")
        if q.get("purpose"):
            guide_lines.append(
                f"Research purpose (do not read aloud): {q['purpose']}"
            )
        if q.get("target_ids"):
            guide_lines.append(
                "Research target IDs (analysis metadata only; NEVER use these "
                "inside TURN resolved): " + ", ".join(q["target_ids"])
            )
        if intents["evidence"]:
            guide_lines.append(
                "Evidence to seek: " + "; ".join(intents["evidence"].values())
            )
            guide_lines.append("Evidence to seek and resolve:")
            for evidence_id, goal in intents["evidence"].items():
                guide_lines.append(f"  {evidence_id}: {goal}")
        if q.get("do_not_assume"):
            guide_lines.append(
                "Do not assume or imply: " + "; ".join(q["do_not_assume"])
            )
        guide_lines.append(
            f"Priority: {q.get('priority', 'required')}; "
            f"sensitivity: {q.get('sensitivity', 'standard')}"
        )
        if probes:
            if relevance:
                guide_lines.append(
                    "Possible follow-up probes (a menu, not a checklist. Use "
                    "only a probe that follows naturally from the participant's "
                    "answer; never reuse one):"
                )
            else:
                guide_lines.append(
                    "Follow-up probes (use these in order, pick a different "
                    "one each time):"
                )
            for j, (probe_id, p) in enumerate(intents["probes"].items(), 1):
                guide_lines.append(f"  {j}. {p} [intent ID: {probe_id}]")
        guide_lines.append(
            f"Maximum follow-ups for this question: {max_fu}. This is a "
            "ceiling, not a target. Move on sooner when the evidence goal is "
            f"covered; after {max_fu} follow-up(s) you MUST move on."
        )
        if transition:
            guide_lines.append(f"Transition phrase to next question: {transition}")
        guide_lines.append("")

    question_guide = "\n".join(guide_lines)

    if not allow_follow_ups:
        probe_rule = (
            "- Follow-ups are disabled. Ask ONLY the predefined main questions "
            "above, in order. After each substantive answer, move directly to "
            "the next main question. Do not ask a probe, clarification question, "
            "or any other content question that is not in the guide. You may "
            "answer a participant's procedural question, then resume with the "
            "same pending main question."
        )
        unused_probe_phrase = "the next predefined main question"
    elif relevance:
        probe_rule = (
            "- Treat listed probes as a menu, not a checklist. Pick one only "
            "when it is relevant and its answer is not already known. Never "
            "reuse a probe."
        )
        unused_probe_phrase = "the most useful next move"
    else:
        probe_rule = (
            "- Ask probes from the list in order, picking a different one each "
            "turn."
        )
        unused_probe_phrase = "the next unused probe"

    if allow_emergent:
        probe_rule += (
            "\n- You may replace a listed probe with a brief, neutral, "
            "content-grounded probe when the participant introduces a concrete "
            "detail, change, concern, benefit, contradiction, or uncertainty. "
            "Ground it in the participant's exact detail whenever possible. "
            "Use only clarification, description, meaning, consequence, "
            "contrast, or source-of-knowledge. Never introduce a claim or "
            "premise the participant did not give you. Every emergent probe "
            "counts toward the same follow-up cap."
        )
    elif allow_follow_ups:
        probe_rule += (
            "\n- With emergent follow-ups disabled, do NOT invent your own "
            "probes; use only the guide."
        )

    coverage_rule = (
        "Preserve their order, but skip an unprotected question only when "
        "every one of its evidence goals was concretely covered earlier. "
        "A mention, inference, or partial answer is not enough to skip it."
        if skip_answered_questions
        else "Preserve their order and ask every predefined main question."
    )

    structured_section = f"""

---

## INTERVIEW PROTOCOL

You are conducting a structured interview. There are {len(questions)} guide
questions. The orchestration profile is `{orchestration_profile}`. {coverage_rule}
The conversation normally follows this turn pattern:

  Q1 (main) → user → zero to the configured maximum follow-ups → Q2 → ...

Hard rules:
- Treat rapport as a process throughout the interview, not as small talk that
  ends after the opening. The warm-up phase lowers the burden of answering:
  be transparent, unhurried, curious, and non-judgmental. A warm-up may enter
  the study topic directly when its question is concrete and low-burden.
- Every opening question must be readily answerable on first hearing. Do not
  ask the participant to summarize their life, day-to-day role, or where a
  broad topic "shows up." Warm-up questions should feel purposeful rather
  than like a demographic intake form. Do not ask for a name, exact location,
  employer, school, child identifiers, or profile facts already supplied by
  the recruiter.
- Never fabricate a personal experience, feeling, family, job, opinion, or
  similarity to manufacture rapport. You are an AI interviewer. You may
  accurately explain your role or why the research team values the
  participant's perspective.
- Show that you listened through a brief content-grounded acknowledgement or
  by using the participant's own words in the next question. Avoid generic
  social filler, repeated paraphrases, canned praise, and false intimacy.
- When leaving the warm-up, give one short, transparent signpost before the
  first core question. If the next topic is sensitive, remind the participant
  that they control what they share. Do not narrate ordinary question changes.
- NEVER ask the same question twice. Once you've asked the main question,
  do not ask it again, even rephrased — move to the probes.
- Treat the participant's answer as conversation state, not just a response to
  the last prompt. Before asking anything, check whether they already answered
  its intent elsewhere. When evidence-aware skipping is enabled, protected
  anchors still must be asked and only fully covered unprotected cards may be
  skipped.
- Preserve momentum without surrendering the guide. If a participant gives a
  long, haphazard answer spanning several topics, silently update all supported
  evidence states, briefly ground the next turn in their most relevant exact
  detail, and ask one focused question. Never answer rambling with a compound
  question, a topic inventory, or several follow-ups.
- If the participant says they do not know, cannot observe it, prefer not to
  answer, or that it does not apply, accept that as a resolved response. Do not
  convert an unavailable fact into a hypothetical question unless the guide
  explicitly asks for imagination or recommendations.
{probe_rule}
- Listen before covering the guide. If follow-ups are enabled and an answer
  contains a specific, high-information moment, follow the participant's exact
  detail before a generic menu probe. If follow-ups are disabled, proceed to
  the next predefined main question without probing that detail.
- Before asking a follow-up, name the exact missing evidence to yourself. Ask
  only if the answer would add something the participant has not already said.
  If they could answer by repeating a sentence they just gave you, skip that
  probe or choose a narrower observation, sequence, consequence, contrast, or
  recurrence question. Do not use "give me an example" when the participant
  has already supplied a concrete episode.
- In exploratory interviews, adapt a listed probe to the participant's exact
  words when that makes it more specific and natural. Keep the listed intent
  ID in TURN metadata even when the participant-visible wording is adapted.
- Do not ask a follow-up merely to fill the available budget. Move to the
  next main question when the evidence goal is adequately covered.
- Ask exactly ONE question per turn, then STOP and wait for the
  participant's answer. Never combine a probe with the next main question,
  and never move to the next main question in the same turn in which you
  asked something. Your reply must contain at most ONE question mark —
  everything you say after your first question is cut off before the
  participant hears it, so end your turn there.
- The transition phrases in the guide are stage directions for you. Never
  read labels like "Transition:" aloud and never write them in parentheses —
  say the phrase naturally as part of your sentence, or skip it.
- If the participant asks a definition or scope clarification, answer it
  briefly (one or two sentences), ask whether they would like to continue or
  have another question, then STOP. Resume the CURRENT probe or question only
  after they confirm. Answering a clarification does NOT count as a follow-up
  and does NOT advance the interview. For a definition or scope question, use the
  study's scope clarification policy when one is supplied. Give only the
  minimum definition or boundary relevant to what they asked; do not volunteer
  a catalogue of examples that could prime their later answers. If the policy
  does not resolve a borderline case, do not guess: ask one brief neutral
  question about the feature or interaction and wait. Resolve it on the next
  turn, then return to the original question.
- Treat privacy, recording, data-use, timing, consent, and participant-rights
  questions as interruptions, never as interview evidence. Answer only from
  the supplied participant information and preserve the exact question that
  was suspended. After answering, ask whether the participant would like to
  continue or has another question, then STOP. Do not resume the guide in that
  same turn. Return to the suspended question only after the participant says
  they are ready; if they ask something else, answer it first. Never say you
  lack study information when that information is present in the participant-
  facing protocol.
- If the participant says they were not finished, the transcript ends in an
  obviously incomplete thought, or they point out that you skipped their
  question, stop the guide flow. Invite them to continue or return to the
  suspended question. Do not advance, resolve evidence, or defend the pace.
- When a participant says a question is vague or does not make sense, do not
  paraphrase it into another abstraction. Ground one simpler question in a
  concrete example they already gave. If they still cannot answer after two
  attempts, accept that and move on.
- Generated probes must be easy to answer on first hearing. Prefer a recent
  event, an observable action, or a direct contrast between two examples the
  participant has already mentioned. Avoid phrases such as "what makes the
  interaction productive," "how did that shape your approach," or other
  broad requests that force the participant to invent an analytic frame.
- Do not turn a concrete judgment boundary into a hypothetical delegation
  test such as "under what conditions would you let AI decide that?" Ask why
  the participant retained the decision in the episode they described. Save
  future hypotheticals for the guide's explicit future-workflow question.
- Do NOT acknowledge or evaluate answers ("great", "interesting",
  "that's helpful"). A neutral "Mm-hm." or going straight to the next
  probe is correct. Validation biases the participant.
- Apart from the one warm-up-to-core signpost or a sensitive-topic warning,
  do NOT narrate mechanics ("now I'll ask a follow-up", "transitioning").
- When you reach the maximum follow-ups for a question, move on to the
  next main question, even if the participant's answers were brief.
- For a question marked sensitivity=elevated, use its wording as written,
  avoid asking for identifying details, and accept a refusal without probing.
- Purpose, evidence, target IDs, priority, and sensitivity are private stage
  directions. Never read or mention them to the participant.

{question_guide}

### Closing
After the final question's follow-ups are done AND the participant has given
their final answer, say exactly: "{closing}"
Never deliver the closing message in the same turn as a question, and never
deliver it while the participant is still waiting for you to repeat or
clarify something — answer them, wait for their reply, then close.

### Keeping track of where you are
Your own previous replies may appear split across several messages in the
history (for example an acknowledgment and a question shown separately).
Treat everything you said between two participant messages as ONE turn.
Before each turn, look at the conversation history and figure out:
1. Which question number are we on (1..{len(questions)})?
2. Which probes from this question's list have I already asked? Count
   follow-ups by matching the probes you actually asked, not by counting
   your own messages. Clarification answers do not count.
3. Therefore, what comes next: {unused_probe_phrase}, or — once the
   participant has answered and the evidence goal is covered — the next main
   question?

If unsure which probe you already asked, do not repeat it. Choose a different
relevant move or advance. You do not need to exhaust the listed probes.
"""

    if not allow_follow_ups and not skip_answered_questions:
        structured_section += """

### FIXED SURVEY OVERRIDE
Follow-ups are disabled. This rule overrides any general probing, threading,
evidence-gap, or clarification-question guidance above. Ask every predefined
main question in order and ask no other content questions. Do not paraphrase,
skip, or replace a main question because its topic appeared earlier. If the
participant asks for clarification, explain briefly without asking a new
question, then repeat the same pending predefined main question.
"""
    elif not allow_follow_ups:
        structured_section += """

### EVIDENCE-AWARE FIXED SURVEY OVERRIDE
Exploratory and listed follow-ups are disabled. Ask protected anchors exactly
as written. You may skip an unprotected main question only after every one of
its evidence goals is concretely covered. If an unprotected future card is
partial, ask at most one short missing-evidence clarification; do not explore
beyond that repair. When moving across skipped cards, name the participant's
task in the next question so references such as "that task" stay unambiguous.
"""

    prompt = base_prompt.rstrip() + structured_section
    if emit_progress:
        prompt += turn_decision_instruction(
            questions, allow_follow_ups, skip_answered_questions
        )
        prompt += progress_marker_instruction(len(questions))
    return prompt


# ── Structured output filter ────────────────────────────────────────────────
#
# Production sessions showed the model cramming several protocol steps into a
# single spoken turn despite the prompt's "ask exactly ONE question per turn"
# rule, e.g.:
#
#   "What was that experience like for you at the time?
#    (Transition: Thanks, that's really helpful background.)
#    Thinking about a specific recent moment, could you walk me through it?"
#
# — a probe, a leaked stage-direction label, and the NEXT main question all
# in one breath, which burned the whole interview in a couple of turns. The
# prompt alone cannot guarantee this never happens, so this filter enforces
# it on the output path: each LLM response is cut after its first question
# sentence, and "(Transition: ...)" label leakage is unwrapped to the bare
# phrase. The assistant context aggregator sits downstream, so the model's
# own history matches what was actually spoken.

_SENTENCE_RE = re.compile(r"[^.!?。！？…]*[.!?。！？…]+[\"'）)\]]*\s*", re.DOTALL)
_TRANSITION_LABEL_RE = re.compile(
    r"[(\[]\s*transition[^:：]*[:：]\s*([^)\]]*)[)\]]", re.IGNORECASE
)
_QUESTION_MARKS = ("?", "？")


class StructuredOutputFilter(FrameProcessor):
    """Enforces the one-question-per-turn protocol on the LLM's output.

    Place between the LLM and the TTS service (before the transcript logger,
    so persisted transcripts match what participants hear). Within each LLM
    response it:

    - buffers streamed ``TextFrame`` tokens and re-emits them sentence by
      sentence;
    - unwraps leaked stage directions ("(Transition: thanks!)" → "thanks!");
    - drops every sentence after the first one containing a question mark.

    Outside structured interviews the pipeline simply doesn't include this
    processor.

    When ``progress_callback`` is provided, hidden ``[[Qn]]`` tags are pulled
    out of the stream and reported (monotonically, clamped to
    ``total_questions``) so the participant UI can advance a progress bar. The
    tag is always removed before any text is pushed downstream to TTS.
    """

    def __init__(
        self,
        name: str = "StructuredOutputFilter",
        progress_callback=None,
        turn_callback=None,
        total_questions: int = 0,
    ):
        super().__init__(name=name)
        self._in_response = False
        self._pending = ""
        self._question_done = False
        self._dropped_chars = 0
        self._progress_callback = progress_callback
        self._turn_callback = turn_callback
        self._total_questions = total_questions
        self._reported_progress = 0
        self._turn_reviewed = False

    def _reset(self):
        self._in_response = False
        self._pending = ""
        self._question_done = False
        self._dropped_chars = 0
        self._turn_reviewed = False

    async def _report_progress(self, question_number: int):
        """Report a main-question number to the UI: forward-only, clamped."""
        if not self._progress_callback:
            return
        n = question_number
        if self._total_questions:
            n = min(n, self._total_questions)
        if n <= self._reported_progress:
            return
        self._reported_progress = n
        try:
            await self._progress_callback(n, self._total_questions)
        except Exception as exc:  # progress is best-effort, never break the call
            logger.debug("[guide] progress callback failed: {}", exc)

    async def _emit_sentence(self, sentence: str, direction: FrameDirection):
        # Pull all control tags out first so they never reach TTS, even on a
        # sentence we go on to drop after the turn's first question.
        sentence, marker, decision = strip_control_markers(sentence)
        if not self._turn_reviewed:
            self._turn_reviewed = True
            if self._turn_callback:
                try:
                    review = await self._turn_callback(decision, marker, sentence)
                    if isinstance(review, TurnReview):
                        sentence = review.text
                        marker = review.marker
                except Exception as exc:  # guardrails must not break audio
                    logger.warning("[guide] turn review callback failed: {}", exc)
        if marker is not None:
            await self._report_progress(marker)
        if self._question_done:
            self._dropped_chars += len(sentence)
            return
        cleaned = _TRANSITION_LABEL_RE.sub(r"\1", sentence)
        if cleaned.strip():
            await self.push_frame(TextFrame(text=cleaned), direction)
        if any(q in cleaned for q in _QUESTION_MARKS):
            self._question_done = True

    async def _drain_pending(self, direction: FrameDirection, final: bool):
        pos = 0
        for m in _SENTENCE_RE.finditer(self._pending):
            if m.start() != pos:
                break
            await self._emit_sentence(m.group(0), direction)
            pos = m.end()
        self._pending = self._pending[pos:]
        if final and self._pending:
            await self._emit_sentence(self._pending, direction)
            self._pending = ""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            self._reset()
            self._in_response = True
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMFullResponseEndFrame):
            await self._drain_pending(direction, final=True)
            if self._dropped_chars:
                logger.info(
                    "[guide] output filter cut {} chars after the turn's "
                    "first question",
                    self._dropped_chars,
                )
            self._reset()
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, InterruptionFrame):
            self._reset()
            await self.push_frame(frame, direction)
            return

        if (
            self._in_response
            and direction == FrameDirection.DOWNSTREAM
            and isinstance(frame, TextFrame)
            and not isinstance(
                frame, (TranscriptionFrame, InterimTranscriptionFrame)
            )
        ):
            self._pending += frame.text
            await self._drain_pending(direction, final=False)
            return

        await self.push_frame(frame, direction)


# ── Shared protocol helpers ─────────────────────────────────────────────────

DEFAULT_MAX_FOLLOW_UPS = 3

DEFAULT_CLOSING = "Thank you for your time. This concludes our interview."


def question_max_follow_ups(question: dict) -> int:
    """Per-question follow-up cap, with a fallback for bad/missing values."""
    try:
        return int(question.get("max_follow_ups", DEFAULT_MAX_FOLLOW_UPS))
    except (TypeError, ValueError):
        return DEFAULT_MAX_FOLLOW_UPS


def build_protocol_guidance(
    questions: list[dict],
    closing: str,
    current_index: int,
) -> tuple[Optional[dict], bool]:
    """Build the nudge that moves the agent on (or wraps up on the last one).

    Returns (message, is_closing). The message is a user-role note; see
    guidance_message for why it is not a system message. Used by both the
    voice processor and the text controller.
    """
    from app.engagement.adaptive import guidance_message

    next_idx = current_index + 1
    if next_idx >= len(questions):
        return (
            guidance_message(
                "[Interview protocol] You have completed all the questions "
                "in the guide. If the participant's last message was a "
                "question, answer it briefly first. Then acknowledge their "
                "answer briefly and deliver the closing message: "
                f'"{closing}". Do not ask any new interview questions, '
                "and never combine a question with the closing message in "
                "the same turn."
            ),
            True,
        )

    current_q = questions[current_index]
    next_q = questions[next_idx]
    transition = (current_q.get("transition") or "").strip()
    next_text = next_q.get("text", "")
    transition_hint = (
        f' Use this transition if it fits: "{transition}".' if transition else ""
    )
    return (
        guidance_message(
            f"[Interview protocol] You have explored question "
            f"{current_index + 1} of {len(questions)} sufficiently. If the "
            f"participant's last message was a question, answer it briefly "
            f"first. Then move directly, or use one brief neutral bridge if "
            f'needed, to question {next_idx + 1}: "{next_text}". Ask only '
            f"that question and wait for their answer."
            f"{transition_hint}"
        ),
        False,
    )


def enforce_one_question_per_turn(text: str) -> tuple[str, int]:
    """Cut a complete reply after its first question and unwrap leaked
    "(Transition: ...)" labels. Returns (cleaned_text, dropped_chars).

    This is the StructuredOutputFilter logic for callers that hold the whole
    message at once (text chat). Strip progress markers before calling.
    """
    sentences: list[str] = []
    pos = 0
    for m in _SENTENCE_RE.finditer(text):
        if m.start() != pos:
            break
        sentences.append(m.group(0))
        pos = m.end()
    tail = text[pos:]
    if tail:
        sentences.append(tail)

    out: list[str] = []
    dropped = 0
    question_done = False
    for sentence in sentences:
        if question_done:
            dropped += len(sentence)
            continue
        cleaned = _TRANSITION_LABEL_RE.sub(r"\1", sentence)
        out.append(cleaned)
        if any(q in cleaned for q in _QUESTION_MARKS):
            question_done = True
    return "".join(out), dropped


_CLARIFICATION_CONFIRMATION_RE = re.compile(
    r"\b(?:would you like to continue|are you ready to continue|"
    r"is there anything else (?:you(?:'d| would) like to ask|you want to ask)|"
    r"does that (?:help|answer your question))\b",
    re.IGNORECASE,
)
_QUESTION_OPENING_RE = re.compile(
    r"^(?:what|which|who|whose|when|where|why|how|do|does|did|is|are|was|"
    r"were|can|could|would|will|should|have|has|had|tell me|describe)\b",
    re.IGNORECASE,
)
_AI_PRODUCT_NAME_RE = re.compile(
    r"\b(?:chatgpt|claude(?:\s+(?:code|design))?|gemini|(?:github|microsoft)\s+copilot|"
    r"copilot|perplexity|cursor|midjourney|dall-e|stable diffusion|meta ai|grok|"
    r"notebooklm)\b",
    re.IGNORECASE,
)
_AI_MODEL_NAME_RE = re.compile(
    r"\b(?:gpt[- ]?\d(?:\.\d+)?|o[134](?:-mini)?|sonnet|opus|haiku|"
    r"claude\s+\d(?:\.\d+)?|gemini\s+\d(?:\.\d+)?|llama\s+\d(?:\.\d+)?|"
    r"mistral(?:\s+\w+)?)\b",
    re.IGNORECASE,
)
_EXPLICIT_MODEL_UNKNOWN_RE = re.compile(
    r"(?:\b(?:do not|don't|did not|didn't|cannot|can't)\b.{0,48}\b(?:model|version)\b|"
    r"\b(?:not sure|unsure|unknown)\b.{0,48}\b(?:model|version)\b|"
    r"\b(?:model|version)\b.{0,48}\b(?:not sure|unsure|unknown|do not know|don't know)\b)",
    re.IGNORECASE,
)


def clarification_pause_response(text: str) -> str:
    """Keep a semantic clarification answer and enforce a confirmation pause.

    Models sometimes answer a participant's scope question correctly and then
    append the suspended or next guide question. The one-question output
    filter cannot distinguish that guide question from a legitimate check-in,
    so normalize the turn first: preserve declarative answer sentences, drop
    any question the model appended, and add one standard confirmation. An
    empty return means no usable clarification answer could be recovered.
    """
    cleaned = _TRANSITION_LABEL_RE.sub(r"\1", str(text or "")).strip()
    if not cleaned:
        return ""

    sentences: list[str] = []
    pos = 0
    for match in _SENTENCE_RE.finditer(cleaned):
        if match.start() != pos:
            break
        sentences.append(match.group(0))
        pos = match.end()
    if pos < len(cleaned):
        sentences.append(cleaned[pos:])

    answer_parts: list[str] = []
    for sentence in sentences:
        question_positions = [
            position
            for mark in _QUESTION_MARKS
            if (position := sentence.find(mark)) >= 0
        ]
        if not question_positions:
            answer_parts.append(sentence)
            continue

        before_question = sentence[: min(question_positions)].strip()
        confirmation = _CLARIFICATION_CONFIRMATION_RE.search(before_question)
        if confirmation:
            before_question = before_question[: confirmation.start()].strip()
        if (
            before_question
            and not _QUESTION_OPENING_RE.match(before_question)
            and len(before_question.split()) >= 3
        ):
            answer_parts.append(before_question.rstrip(" ,;—-") + ".")
        break

    answer = "".join(answer_parts).strip()
    if not answer or answer.lower().rstrip(".! ") in {
        "okay",
        "ok",
        "sure",
        "yes",
        "no",
    }:
        return ""
    return (
        f"{answer.rstrip()} Would you like to continue, or is there anything "
        "else you'd like to ask first?"
    )


class EvidenceStateTracker:
    """Session-owned semantic memory and duplicate-question circuit breaker.

    The model reports evidence in the same response it was already going to
    generate. This tracker validates IDs, remembers resolutions and asked
    intents, and deterministically substitutes the next unresolved main
    question when the proposed turn would repeat or over-probe. It therefore
    adds no planner call and no participant-visible metadata.
    """

    def __init__(
        self,
        questions: list[dict],
        closing: str,
        *,
        probe_selection: str = "ordered",
        allow_emergent_follow_ups: bool = False,
        allow_follow_ups: bool = True,
        skip_answered_questions: bool = True,
        allow_coverage_clarifications: bool = False,
        allow_threading: bool = True,
        language: str = "en",
        orchestration_profile: str = "semi_structured",
    ):
        self._questions = questions
        self._closing = closing
        self._probe_selection = (
            "relevance" if str(probe_selection).lower() == "relevance" else "ordered"
        )
        self._allow_emergent_follow_ups = bool(allow_emergent_follow_ups)
        self._allow_follow_ups = bool(allow_follow_ups)
        self._skip_answered_questions = bool(skip_answered_questions)
        self._allow_coverage_clarifications = bool(
            allow_coverage_clarifications
        )
        self._allow_threading = bool(allow_threading)
        self._language = language or "en"
        self._orchestration_profile = orchestration_profile
        self._catalog = [
            question_intents(question, index)
            for index, question in enumerate(questions)
        ]
        self._question_positions = {
            str(question.get("id")): index
            for index, question in enumerate(questions)
            if str(question.get("id") or "").strip()
        }
        self._valid_evidence = {
            evidence_id
            for catalog in self._catalog
            for evidence_id in catalog["evidence"]
        }
        self._asked_intents: set[str] = set()
        self._resolutions: dict[str, str] = {}
        self._events: list[dict[str, object]] = []
        self._missing_decisions = 0
        self._pending_substantive_missing_decision = False
        self._missing_decision_participant_words = 0
        self._missing_decision_recoveries = 0
        self._inline_decision_recoveries = 0
        self._invalid_decisions = 0
        self._invalid_resolution_ids: set[str] = set()
        self._terminal_resolutions_rejected = 0
        self._duplicate_intents_blocked = 0
        self._resolved_probes_blocked = 0
        self._covered_questions_skipped = 0
        self._semantic_advances = 0
        self._order_corrections = 0
        self._threaded_from: set[int] = set()
        self._pending_thread: Optional[tuple[int, int]] = None
        self._threaded_probes_accepted = 0
        self._depth_threads_accepted = 0
        self._threading_blocked = 0
        self._thread_resumes = 0
        self._focused_advances_accepted = 0
        self._probe_policy_corrections = 0
        self._probe_wording_corrections = 0
        self._adaptive_probe_wording_accepted = 0
        self._emergent_probes_blocked = 0
        self._answer_repairs_accepted = 0
        self._answer_repairs_blocked = 0
        self._clarification_misclassifications_blocked = 0
        self._latest_focal_entity: Optional[str] = None
        self._latest_provenance: Optional[str] = None

    def _question_resolved(self, index: int) -> bool:
        evidence = set(self._catalog[index]["evidence"])
        return bool(evidence) and all(
            self._resolutions.get(evidence_id) in {
                "covered",
                "unknown",
                "declined",
                "not_applicable",
            }
            for evidence_id in evidence
        )

    def _question_protected(self, index: int) -> bool:
        return bool(self._questions[index].get("always_ask"))

    def _main_asked(self, index: int) -> bool:
        return str(self._catalog[index]["main"]) in self._asked_intents

    def _blocking_question(self, requested_index: int) -> Optional[int]:
        """Return the earliest anchor/dependency that must precede a card."""
        for index in range(0, min(requested_index, len(self._questions))):
            if self._question_protected(index) and not self._main_asked(index):
                return index
        if not 0 <= requested_index < len(self._questions):
            return None
        for dependency in self._questions[requested_index].get("depends_on") or []:
            dependency_index = self._question_positions.get(str(dependency))
            if dependency_index is not None and not self._main_asked(dependency_index):
                return dependency_index
        return None

    def _question_skippable(self, index: int) -> bool:
        evidence = set(self._catalog[index]["evidence"])
        statuses = {self._resolutions.get(evidence_id) for evidence_id in evidence}
        return (
            self._skip_answered_questions
            and not self._question_protected(index)
            and bool(evidence)
            and (
                statuses == {"covered"}
                # Terminal non-answers are accepted only for the question the
                # participant was actually answering, so retaining this
                # legacy path cannot silently dispose of unasked future cards.
                or statuses.issubset(
                    {"covered", "unknown", "declined", "not_applicable"}
                )
            )
        )

    def _next_unresolved(self, start: int) -> Optional[int]:
        blocker = self._blocking_question(min(max(0, start), len(self._questions)))
        if blocker is not None:
            return blocker
        for index in range(max(0, start), len(self._questions)):
            dependency = self._blocking_question(index)
            if dependency is not None:
                return dependency
            if self._question_skippable(index):
                self._covered_questions_skipped += 1
                continue
            return index
        return None

    def _peek_next_unresolved(self, start: int) -> Optional[int]:
        """Find the next question without mutating observability counters."""
        blocker = self._blocking_question(min(max(0, start), len(self._questions)))
        if blocker is not None:
            return blocker
        for index in range(max(0, start), len(self._questions)):
            dependency = self._blocking_question(index)
            if dependency is not None:
                return dependency
            if not self._question_skippable(index):
                return index
        return None

    def safe_next_index(self, start: int) -> Optional[int]:
        """Public, side-effect-free navigation used by both transports."""
        if 0 <= start < len(self._questions) and self._main_asked(start):
            start += 1
        return self._peek_next_unresolved(start)

    def safe_marker_index(self, requested_index: int) -> int:
        """Clamp a model marker only at an authored anchor/dependency."""
        blocker = self._blocking_question(requested_index)
        return blocker if blocker is not None else requested_index

    def _allowed_intents(self, index: int) -> set[str]:
        catalog = self._catalog[index]
        allowed = {str(catalog["main"])}
        if self._allow_follow_ups:
            allowed.update(catalog["probes"].keys())
            allowed.update(catalog["emergent"])
        elif self._allow_coverage_clarifications:
            allowed.add(f"{catalog['question_id']}:emergent:clarification")
        return allowed

    def _question_has_evidence(self, index: int) -> bool:
        return any(
            evidence_id in self._resolutions
            for evidence_id in self._catalog[index]["evidence"]
        )

    def _question_has_asked_followup_intent(self, index: int) -> bool:
        """Whether the card received a focused probe/repair, not its main."""
        catalog = self._catalog[index]
        focused = set(catalog["probes"]) | set(catalog["emergent"])
        return any(intent in self._asked_intents for intent in focused)

    def _available_listed_probe(self, index: int) -> Optional[tuple[str, str]]:
        if not self._allow_follow_ups:
            return None
        for intent, text in self._catalog[index]["probes"].items():
            if intent not in self._asked_intents and str(text).strip():
                return str(intent), str(text).strip()
        return None

    def _listed_probe_text(self, index: int, intent: str) -> Optional[str]:
        text = self._catalog[index]["probes"].get(intent)
        cleaned = str(text or "").strip()
        return cleaned or None

    def _normalize_probe_policy(
        self,
        text: str,
        decision: TurnDecision,
        question_index: int,
        participant_text: str = "",
    ) -> tuple[str, TurnDecision, bool, Optional[str]]:
        """Bind model-selected follow-ups to the configured probe policy.

        Ordered and semi-structured guides retain server-owned probe wording.
        An explicit exploratory profile may adapt a relevance-selected listed
        probe to a concrete participant detail; this is equivalent to the
        emergent probing that profile already authorizes.  The adaptation is
        accepted only when it is short, non-leading, and grounded in the latest
        answer. Emergent intent IDs are usable only when the guide explicitly
        enables them.
        """
        if decision.action not in {"probe", "thread"} or not decision.intent:
            return text, decision, False, None

        catalog = self._catalog[question_index]
        intent = decision.intent
        replacement_intent: Optional[str] = None
        replacement_text: Optional[str] = None
        reason: Optional[str] = None

        is_coverage_clarification = (
            self._allow_coverage_clarifications
            and not self._allow_follow_ups
            and intent == f"{catalog['question_id']}:emergent:clarification"
            and decision.action == "probe"
        )
        if is_coverage_clarification:
            return text, decision, False, None
        if intent in catalog["emergent"] and not self._allow_emergent_follow_ups:
            self._emergent_probes_blocked += 1
            reason = "emergent_probes_disabled"
            available = self._available_listed_probe(question_index)
            if available is not None:
                replacement_intent, replacement_text = available
        elif intent in catalog["probes"]:
            if self._probe_selection == "ordered":
                available = self._available_listed_probe(question_index)
                if available is not None and available[0] != intent:
                    replacement_intent, replacement_text = available
                    reason = "ordered_probe_corrected"
            if replacement_intent is None:
                replacement_intent = intent
                replacement_text = self._listed_probe_text(question_index, intent)
                proposed_text = str(text or "").strip()
                adaptive_wording = (
                    replacement_text != proposed_text
                    and self._orchestration_profile == "exploratory"
                    and self._probe_selection == "relevance"
                    and self._allow_emergent_follow_ups
                    and self._adaptive_probe_is_safe(
                        proposed_text, participant_text
                    )
                )
                if adaptive_wording:
                    self._adaptive_probe_wording_accepted += 1
                    self._record_event(
                        action="accept_adaptive_probe",
                        question=question_index + 1,
                        intent=intent,
                    )
                    return text, decision, False, None
                if replacement_text != proposed_text:
                    reason = "listed_probe_wording_corrected"

        if replacement_intent is None or replacement_text is None:
            return text, decision, False, reason

        normalized = TurnDecision(
            action=decision.action,
            question=decision.question,
            intent=replacement_intent,
            resolved=decision.resolved,
            focal_entity=decision.focal_entity,
            provenance=decision.provenance,
        )
        changed = (
            replacement_intent != decision.intent
            or replacement_text != str(text or "").strip()
        )
        if changed:
            self._probe_policy_corrections += 1
            if replacement_text != str(text or "").strip():
                self._probe_wording_corrections += 1
            self._record_event(
                action="normalize_probe",
                reason=reason,
                question=question_index + 1,
                proposed_intent=decision.intent,
                accepted_intent=replacement_intent,
            )
        return replacement_text, normalized, changed, reason

    @staticmethod
    def _adaptive_probe_is_safe(text: str, participant_text: str) -> bool:
        """Apply a narrow release guard to exploratory probe paraphrases."""
        cleaned = str(text or "").strip()
        participant = str(participant_text or "").lower()
        if (
            not cleaned
            or not participant
            or len(cleaned.split()) > 32
            or sum(cleaned.count(mark) for mark in _QUESTION_MARKS) != 1
            or _LEADING_PROBE_RE.search(cleaned)
        ):
            return False
        content_tokens = {
            token
            for token in re.findall(r"[a-z0-9']+", cleaned.lower())
            if len(token) >= 4 and token not in _PROBE_GROUNDING_STOPWORDS
        }
        return any(token in participant for token in content_tokens)

    def clarification_resume_text(self, index: int) -> str:
        """Return one bounded prompt for a clarification repair."""
        available = self._available_listed_probe(index)
        if available is not None:
            return available[1]
        if 0 <= index < len(self._questions):
            return str(self._questions[index].get("text") or "").strip()
        return ""

    def _thread_target_is_safe(self, index: int) -> bool:
        """Keep opportunistic probes out of sensitive and closing material."""
        if not 0 <= index < len(self._questions):
            return False
        question = self._questions[index]
        return (
            str(question.get("sensitivity") or "standard").lower() != "elevated"
            and str(question.get("phase") or "core").lower() not in {"close", "closing"}
            and bool(self._catalog[index]["evidence"])
        )

    def _clear_pending_thread(self, reason: str) -> None:
        if self._pending_thread is None:
            return
        anchor, target = self._pending_thread
        self._thread_resumes += 1
        self._record_event(
            action="thread_resume",
            reason=reason,
            anchor_question=anchor + 1,
            threaded_question=target + 1,
        )
        self._pending_thread = None

    def _record_event(self, **values: object) -> None:
        # Bound diagnostic context for long-running panel interviews.
        if len(self._events) < 100:
            self._events.append(values)

    def _high_confidence_resolution_hints(
        self, participant_text: str
    ) -> dict[str, str]:
        """Recover a few unmistakable fields when a TURN marker is omitted.

        This is deliberately narrow. It does not attempt generic semantic
        grading; it only recognizes explicit AI product/model names for cards
        whose authored evidence contract asks for those exact fields.
        """
        text = str(participant_text or "")
        if not text:
            return {}
        product_named = bool(_AI_PRODUCT_NAME_RE.search(text))
        model_named = bool(_AI_MODEL_NAME_RE.search(text))
        model_unknown = bool(_EXPLICIT_MODEL_UNKNOWN_RE.search(text))
        if not product_named and not model_named and not model_unknown:
            return {}

        hints: dict[str, str] = {}
        for catalog in self._catalog:
            for evidence_id, goal in catalog["evidence"].items():
                normalized = str(goal).lower()
                if (
                    product_named
                    and re.search(r"\b(?:ai product|product name|tool name)\b", normalized)
                ):
                    hints[evidence_id] = "covered"
                elif re.search(r"\b(?:model family|model version)\b", normalized):
                    if model_named or model_unknown:
                        hints[evidence_id] = "covered"
                    elif product_named:
                        hints[evidence_id] = "partial"
        return hints

    def _recover_missing_main_decision(
        self,
        text: str,
        marker: Optional[int],
        participant_text: str,
    ) -> Optional[TurnDecision]:
        """Rebuild an exact main-card decision for a marker omission.

        Recovery requires both a confidently identified authored question and
        high-confidence evidence in the participant's answer. Otherwise the
        legacy fail-open display behavior is preserved.
        """
        question_index: Optional[int] = None
        if marker is not None and 1 <= marker <= len(self._questions):
            question_index = marker - 1
        else:
            proposed = " ".join(str(text or "").lower().split())
            proposed_lead = proposed.split("?", 1)[0].strip()
            for index, question in enumerate(self._questions):
                authored = " ".join(
                    str(question.get("text") or "").lower().split()
                )
                if proposed_lead and proposed_lead == authored.split("?", 1)[0].strip():
                    question_index = index
                    break
        if question_index is None:
            return None
        resolved = self._high_confidence_resolution_hints(participant_text)
        if not resolved:
            return None
        return TurnDecision(
            action="main",
            question=question_index + 1,
            intent=str(self._catalog[question_index]["main"]),
            resolved=resolved,
        )

    def _replacement(
        self, start: int, reason: str
    ) -> TurnReview:
        self._clear_pending_thread(f"replacement:{reason}")
        next_index = self._next_unresolved(start)
        self._semantic_advances += 1
        if next_index is None:
            decision = TurnDecision(action="close", question=None, intent=None)
            self._record_event(action="replace", reason=reason, replacement="close")
            return TurnReview(
                text=self._closing,
                marker=None,
                decision=decision,
                replaced=True,
                reason=reason,
                question_index=len(self._questions),
                closed=True,
            )

        intent = str(self._catalog[next_index]["main"])
        self._asked_intents.add(intent)
        decision = TurnDecision(
            action="main",
            question=next_index + 1,
            intent=intent,
        )
        self._record_event(
            action="replace",
            reason=reason,
            replacement_question=next_index + 1,
            replacement_intent=intent,
        )
        visible = str(self._questions[next_index].get("text") or "").strip()
        if next_index > start:
            visible = self._contextual_question_text(next_index, visible)
        return TurnReview(
            text=visible,
            marker=next_index + 1,
            decision=decision,
            replaced=True,
            reason=reason,
            question_index=next_index,
        )

    def _contextual_question_text(self, index: int, text: str) -> str:
        """Keep pronouns unambiguous after one or more evidence-based skips."""
        # Self-contained anchors such as "Thinking back about six months..."
        # do not need the prior task pasted onto them. Doing that creates
        # stilted transitions ("Thinking about X, thinking back..."). Only add
        # context when the destination wording actually points backward.
        if not re.search(
            r"\b(?:that|this|the same)\s+"
            r"(?:task|work|output|result|experience|example|tool|process|workflow)\b",
            text,
            flags=re.IGNORECASE,
        ):
            return text
        label = re.sub(r"[^a-zA-Z0-9 _-]", "", self._latest_focal_entity or "")
        label = re.sub(r"[_-]+", " ", label).strip()
        if not label or len(label.split()) > 10 or not text:
            return text
        lowered = text[0].lower() + text[1:]
        return f"Thinking about the {label}, {lowered}"

    def _probe_replacement(self, index: int, reason: str) -> Optional[TurnReview]:
        available = self._available_listed_probe(index)
        if available is None:
            return None
        intent, text = available
        self._asked_intents.add(intent)
        decision = TurnDecision(
            action="probe",
            question=index + 1,
            intent=intent,
        )
        self._record_event(
            action="replace",
            reason=reason,
            replacement_question=index + 1,
            replacement_intent=intent,
        )
        return TurnReview(
            text=text,
            marker=None,
            decision=decision,
            replaced=True,
            reason=reason,
            question_index=index,
        )

    def _coverage_repair_replacement(
        self, index: int, reason: str
    ) -> Optional[TurnReview]:
        """Spend repair-only's single clarification before allowing advance."""
        if not self._allow_coverage_clarifications or self._allow_follow_ups:
            return None
        # A card is incomplete both when one goal is explicitly `partial` and
        # when one goal is covered while another is still wholly missing.
        # The latter is common when someone volunteers the product name but
        # not the model version. In either case, repair the missing field
        # instead of replaying the broad card.
        if not self._question_has_evidence(index) or self._question_resolved(index):
            return None
        intent = f"{self._catalog[index]['question_id']}:emergent:clarification"
        if intent in self._asked_intents:
            return None
        visible = next(
            (
                str(value).strip()
                for value in self._catalog[index]["probes"].values()
                if str(value).strip()
            ),
            "",
        )
        if not visible:
            # Repair-only guides may intentionally omit exploratory probes.
            # They still get one neutral, card-grounded quality-control repair
            # when the participant supplied only a partial answer.
            visible = self._fallback_coverage_repair_text(index)
        self._asked_intents.add(intent)
        self._answer_repairs_accepted += 1
        self._record_event(
            action="replace",
            reason=reason,
            replacement_question=index + 1,
            replacement_intent=intent,
        )
        return TurnReview(
            text=visible,
            marker=None,
            decision=TurnDecision(
                action="probe",
                question=index + 1,
                intent=intent,
            ),
            replaced=True,
            reason=reason,
            question_index=index,
        )

    def _fallback_coverage_repair_text(self, index: int) -> str:
        """Compile a short, answerable repair from a card's evidence contract."""
        question = self._questions[index]
        main = str(question.get("text") or "").lower()
        catalog_evidence = self._catalog[index]["evidence"]
        unresolved_goals = [
            str(goal)
            for evidence_id, goal in catalog_evidence.items()
            if self._resolutions.get(evidence_id)
            not in {"covered", "unknown", "declined", "not_applicable"}
        ]
        evidence = " ".join(
            unresolved_goals
            or [str(goal) for goal in question.get("evidence_goal") or []]
        ).lower()
        source = " ".join(
            [main, str(question.get("purpose") or "").lower(), evidence]
        )
        if re.search(r"\b(?:model family|model version)\b", evidence):
            return (
                "Do you know which model family or version it was? If not, "
                "that's fine."
            )
        if re.search(r"\b(?:product name|tool name|provider)\b", evidence):
            return "Which AI product or tool was it, if you know?"
        if re.search(r"\b(?:context|constraint|requirement)\b", evidence):
            return (
                "Did you give it any important constraints or prior context? "
                "Please describe only the type, not the contents."
            )
        if re.search(
            r"\b(?:input|instruction|prompt|source material|"
            r"context supplied|constraint supplied)\b",
            evidence,
        ) or re.search(r"\bwhat did you (?:give|provide|supply)\b", main):
            return (
                "What type of information or material did you give the AI—for "
                "example, instructions, a spreadsheet, a document, an image, "
                "or prior conversation? Please describe the type only, not "
                "its contents."
            )
        if re.search(
            r"\b(?:which (?:ai|tool|product)|what model)\b",
            main,
        ):
            return "Which product or model was it, if you know?"
        if re.search(
            r"\b(?:observed ai behavior|error|omission|failure mode|"
            r"output could not be used)\b",
            evidence,
        ) or re.search(r"\b(?:error|omission|problem)\b", main):
            return (
                "What kind of error or omission did you notice? Please keep "
                "the example general."
            )
        if re.search(
            r"\b(?:who or what determined|final decision|determined acceptability)\b",
            evidence,
        ):
            return "Who made the final decision that the result was usable?"
        if re.search(
            r"\b(?:check|verify|verification|evidence|threshold|usable)\b",
            source,
        ):
            return "What did you check to decide whether the result was usable?"
        if re.search(
            r"\b(?:next action|next step|workaround|response|did next)\b",
            source,
        ):
            return "What did you do next, specifically?"
        if re.search(
            r"\b(?:consequence|impact|stake|mattered|important)\b",
            source,
        ):
            return "Why did getting it right matter in that situation?"
        if re.search(r"\b(?:outcome|result|what happened)\b", source):
            return "What happened as a result?"
        if re.search(r"\b(?:concern|worry|risk|afraid)\b", source):
            return "What specifically concerns you about it?"
        if re.search(r"\b(?:task|episode|experience|situation|moment)\b", source):
            return "What was the specific task or situation?"
        return "What's one concrete example of what you mean?"

    def _focused_repair_replacement(
        self, index: int, reason: str
    ) -> Optional[TurnReview]:
        return self._probe_replacement(
            index, reason
        ) or self._coverage_repair_replacement(index, reason)

    def review(
        self,
        text: str,
        marker: Optional[int],
        decision: Optional[TurnDecision],
        current_index: int,
        participant_text: str = "",
    ) -> TurnReview:
        conditional_skip = conditional_module_skip_index(
            self._questions, decision, current_index
        )
        if conditional_skip is not None:
            answered_index, skip_to = conditional_skip
            for index in range(answered_index, skip_to):
                for evidence_id in self._catalog[index]["evidence"]:
                    self._resolutions[evidence_id] = "not_applicable"
            return self._replacement(skip_to, "module_not_applicable")

        if decision is None:
            self._missing_decisions += 1
            recovered_decision = self._recover_missing_main_decision(
                text, marker, participant_text
            )
            if recovered_decision is not None:
                decision = recovered_decision
                self._inline_decision_recoveries += 1
                self._record_event(
                    action="recover",
                    reason="missing_decision_with_explicit_tool_evidence",
                    question=decision.question,
                    resolutions=len(decision.resolved),
                )
            else:
                participant_words = len((participant_text or "").split())
                if (
                    participant_words >= 20
                    and any(mark in (text or "") for mark in _QUESTION_MARKS)
                ):
                    self._pending_substantive_missing_decision = True
                    self._missing_decision_participant_words = participant_words
                return TurnReview(
                    text=text,
                    marker=marker,
                    decision=None,
                    question_index=(marker - 1) if marker else current_index,
                )

        had_substantive_missing_decision = self._pending_substantive_missing_decision
        missing_decision_participant_words = self._missing_decision_participant_words
        self._pending_substantive_missing_decision = False
        self._missing_decision_participant_words = 0

        current_evidence = (
            set(self._catalog[current_index]["evidence"])
            if 0 <= current_index < len(self._catalog)
            else set()
        )
        valid_resolutions: dict[str, str] = {}
        for evidence_id, status in decision.resolved.items():
            if evidence_id not in self._valid_evidence:
                continue
            if status in {"covered", "partial"}:
                valid_resolutions[evidence_id] = status
                continue
            # Terminal non-answer states are powerful: they stop probing. Do
            # not trust them unless the participant explicitly used matching
            # language while answering the current question.
            pattern = _TERMINAL_STATUS_PATTERNS.get(status)
            if (
                evidence_id in current_evidence
                and pattern is not None
                and pattern.search(participant_text or "")
            ):
                valid_resolutions[evidence_id] = status
            else:
                self._terminal_resolutions_rejected += 1
        invalid_resolution_ids = {
            evidence_id
            for evidence_id in decision.resolved
            if evidence_id not in self._valid_evidence
        }
        if invalid_resolution_ids:
            self._invalid_decisions += 1
            self._invalid_resolution_ids.update(invalid_resolution_ids)
        for evidence_id, status in valid_resolutions.items():
            # Never let a later, weaker self-report downgrade resolved state.
            if status == "partial" and self._resolutions.get(evidence_id) in {
                "covered",
                "unknown",
                "declined",
                "not_applicable",
            }:
                continue
            self._resolutions[evidence_id] = status
        if decision.focal_entity:
            self._latest_focal_entity = decision.focal_entity
        if decision.provenance:
            self._latest_provenance = decision.provenance

        if decision.action == "close":
            current_main_asked = (
                0 <= current_index < len(self._catalog)
                and str(self._catalog[current_index]["main"])
                in self._asked_intents
            )
            next_index = self._next_unresolved(
                current_index + 1 if current_main_asked else current_index
            )
            if next_index is not None:
                self._invalid_decisions += 1
                return self._replacement(next_index, "premature_close")
            self._record_event(action="accept", turn_action="close")
            self._clear_pending_thread("close")
            return TurnReview(
                text=text,
                marker=None,
                decision=decision,
                question_index=len(self._questions),
                closed=True,
            )

        if decision.question is None or not (
            1 <= decision.question <= len(self._questions)
        ):
            self._invalid_decisions += 1
            self._record_event(action="accept_fallback", reason="invalid_question")
            return TurnReview(
                text=text,
                marker=marker,
                decision=decision,
                question_index=(marker - 1) if marker else current_index,
            )

        question_index = decision.question - 1
        if marker is not None and marker != decision.question:
            self._invalid_decisions += 1

        if decision.action == "clarify":
            participant_kind = classify_participant_turn(
                participant_text, self._language
            )
            if participant_kind not in {
                "procedural",
                "flow_repair",
                "clarification",
                "continuation",
            }:
                # The model may not relabel an adequate answer as a request for
                # clarification. This was the source of exact broad-question
                # repeats after rhetorical answers ending in a question mark.
                self._invalid_decisions += 1
                self._clarification_misclassifications_blocked += 1
                current_main_asked = (
                    0 <= current_index < len(self._catalog)
                    and self._main_asked(current_index)
                )
                return self._replacement(
                    current_index + 1 if current_main_asked else current_index,
                    "clarify_without_participant_request",
                )
            anchor_index = (
                self._pending_thread[0]
                if self._pending_thread is not None
                else question_index
            )
            self._record_event(
                action="accept", turn_action="clarify", question=decision.question
            )
            visible = str(text or "").strip()
            if participant_kind in {"clarification", "procedural"}:
                visible = clarification_pause_response(visible) or visible
            return TurnReview(
                text=visible,
                marker=None,
                decision=decision,
                question_index=anchor_index,
            )

        blocking_question = self._blocking_question(question_index)
        if blocking_question is not None:
            self._invalid_decisions += 1
            self._order_corrections += 1
            return self._replacement(
                blocking_question, "unmet_anchor_or_dependency"
            )

        intent = decision.intent
        if not intent or intent not in self._allowed_intents(question_index):
            self._invalid_decisions += 1
            if decision.action in {"probe", "thread"}:
                # Invalid metadata must fail closed for follow-ups. Previously
                # an omitted or invented intent bypassed listed-probe wording
                # and emergent-probe policy because the permissive diagnostic
                # fallback returned the model's visible text unchanged.
                self._probe_policy_corrections += 1
                self._record_event(
                    action="block_probe",
                    reason="invalid_followup_intent",
                    question=decision.question,
                    intent=intent,
                )
                replacement = self._probe_replacement(
                    question_index, "invalid_followup_intent"
                )
                if replacement is not None:
                    self._probe_wording_corrections += 1
                    return replacement
                return self._replacement(
                    current_index + 1
                    if decision.action == "thread"
                    else question_index + 1,
                    "invalid_followup_intent",
                )
            self._record_event(
                action="accept_fallback",
                reason="invalid_intent",
                question=decision.question,
                intent=intent,
            )
            return TurnReview(
                text=text,
                marker=marker,
                decision=decision,
                question_index=question_index,
            )

        if decision.action in {"probe", "thread"} and not self._allow_follow_ups:
            has_incomplete_evidence = (
                self._question_has_evidence(question_index)
                and not self._question_resolved(question_index)
            )
            valid_coverage_repair = (
                decision.action == "probe"
                and self._allow_coverage_clarifications
                and (
                    not self._question_protected(question_index)
                    or (
                        question_index == current_index
                        and self._main_asked(question_index)
                    )
                )
                and has_incomplete_evidence
                and intent
                == f"{self._catalog[question_index]['question_id']}:emergent:clarification"
            )
            if not valid_coverage_repair:
                self._probe_policy_corrections += 1
                self._answer_repairs_blocked += 1
                return self._replacement(question_index, "coverage_repair_not_allowed")
            if intent in self._asked_intents:
                self._answer_repairs_blocked += 1
                return self._replacement(
                    question_index + 1, "coverage_repair_limit_reached"
                )
            self._answer_repairs_accepted += 1

        policy_replaced = False
        policy_reason: Optional[str] = None
        if decision.action in {"probe", "thread"}:
            text, decision, policy_replaced, policy_reason = (
                self._normalize_probe_policy(
                    text, decision, question_index, participant_text
                )
            )
            intent = decision.intent
            if (
                decision.action == "thread"
                and question_index == current_index
                and policy_replaced
            ):
                # A same-card "thread" is malformed metadata, not a reason to
                # discard a safe listed fallback or advance prematurely.
                decision = TurnDecision(
                    action="probe",
                    question=decision.question,
                    intent=decision.intent,
                    resolved=decision.resolved,
                    focal_entity=decision.focal_entity,
                    provenance=decision.provenance,
                )
            if (
                policy_reason == "emergent_probes_disabled"
                and not policy_replaced
            ):
                self._probe_policy_corrections += 1
                return self._replacement(
                    current_index + 1
                    if decision.action == "thread"
                    else question_index + 1,
                    policy_reason,
                )

        # A rare model response can contain an excellent participant-visible
        # question but omit its hidden TURN marker. If the following response
        # supplies a nearly complete evidence map and proposes the
        # participant-led closing card, do not punish the participant by
        # replaying one or two early broad cards whose state was lost with the
        # marker. This recovery is deliberately narrow: both participant
        # answers must be substantive, at least all but two non-closing cards
        # must now be resolved, and it applies only to the guide's closing
        # question—not the final closing message.
        question_phase = str(
            self._questions[question_index].get("phase") or "core"
        ).lower()
        if (
            had_substantive_missing_decision
            and missing_decision_participant_words >= 20
            and len((participant_text or "").split()) >= 20
            and decision.action == "main"
            and question_phase in {"close", "closing"}
        ):
            non_closing_indexes = [
                index
                for index, question in enumerate(self._questions)
                if str(question.get("phase") or "core").lower()
                not in {"close", "closing"}
            ]
            unresolved_non_closing = [
                index
                for index in non_closing_indexes
                if not self._question_resolved(index)
            ]
            if (
                len(non_closing_indexes) >= 3
                and len(unresolved_non_closing) <= 2
            ):
                self._asked_intents.add(intent)
                self._missing_decision_recoveries += 1
                self._record_event(
                    action="accept",
                    turn_action="missing_decision_recovery",
                    question=question_index + 1,
                    unresolved_questions=[index + 1 for index in unresolved_non_closing],
                )
                return TurnReview(
                    text=text,
                    marker=question_index + 1,
                    decision=decision,
                    question_index=question_index,
                )

        if decision.action == "thread":
            current_main_asked = (
                0 <= current_index < len(self._catalog)
                and str(self._catalog[current_index]["main"])
                in self._asked_intents
            )
            target_evidence = set(self._catalog[question_index]["evidence"])
            grounded_target_evidence = target_evidence.intersection(valid_resolutions)
            target_main_intent = str(self._catalog[question_index]["main"])
            non_closing_indexes = [
                index
                for index, question in enumerate(self._questions)
                if str(question.get("phase") or "core").lower()
                not in {"close", "closing"}
                and str(question.get("sensitivity") or "standard").lower()
                != "elevated"
            ]
            resolved_non_closing = sum(
                self._question_resolved(index) for index in non_closing_indexes
            )
            rich_depth_threshold = max(
                2,
                (4 * len(non_closing_indexes) + 4) // 5,
            )
            resolved_depth_check = (
                len(non_closing_indexes) >= 3
                and resolved_non_closing >= rich_depth_threshold
                and len((participant_text or "").split()) >= 20
                and self._question_resolved(question_index)
            )
            block_reason: Optional[str] = None
            if not self._allow_threading:
                block_reason = "threading_disabled"
            elif self._pending_thread is not None:
                block_reason = "thread_chain"
            elif current_index in self._threaded_from:
                block_reason = "thread_limit"
            elif not current_main_asked and not self._question_resolved(current_index):
                block_reason = "thread_before_active_main"
            elif question_index <= current_index:
                block_reason = "thread_not_downstream"
            elif not self._thread_target_is_safe(question_index):
                block_reason = "thread_unsafe_target"
            elif intent == target_main_intent:
                block_reason = "thread_used_main_intent"
            elif intent in self._asked_intents:
                block_reason = "thread_duplicate_intent"
            elif not grounded_target_evidence:
                block_reason = "thread_without_target_evidence"
            elif self._question_resolved(question_index) and not resolved_depth_check:
                block_reason = "thread_target_resolved"

            if block_reason is not None:
                self._threading_blocked += 1
                self._order_corrections += 1
                self._record_event(
                    action="block_thread",
                    reason=block_reason,
                    anchor_question=current_index + 1,
                    proposed_question=question_index + 1,
                )
                return self._replacement(current_index + 1, block_reason)

            self._asked_intents.add(intent)
            self._threaded_from.add(current_index)
            self._pending_thread = (current_index, question_index)
            self._threaded_probes_accepted += 1
            if resolved_depth_check:
                self._depth_threads_accepted += 1
            self._record_event(
                action="accept",
                turn_action=("depth_thread" if resolved_depth_check else "thread"),
                anchor_question=current_index + 1,
                question=question_index + 1,
                intent=intent,
                grounded_evidence=sorted(grounded_target_evidence),
            )
            return TurnReview(
                text=text,
                marker=None,
                decision=decision,
                replaced=policy_replaced,
                reason=policy_reason,
                # Keep the runtime cursor on the active guide question. The
                # hidden decision still records which downstream card owns the
                # opportunistic probe.
                question_index=current_index,
            )

        # If the participant already supplied part of a card before its broad
        # wording was asked, never replay that main wording. Prefer an unused
        # listed probe so the next turn adds evidence instead of restarting.
        if (
            decision.action == "main"
            and intent not in self._asked_intents
            and self._skip_answered_questions
            and not self._question_protected(question_index)
            and self._question_has_evidence(question_index)
            and not self._question_resolved(question_index)
        ):
            focused = self._focused_repair_replacement(
                question_index, "broad_main_after_partial_answer"
            )
            if focused is not None:
                return focused

        # Enforce the first unresolved guide question. A model may jump only
        # over questions whose evidence state is actually resolved, and may
        # never go backward. This runs below semantic resolution ingestion so
        # a rich answer can legitimately cover later guide items early.
        if decision.action == "main":
            current_main_asked = (
                0 <= current_index < len(self._catalog)
                and str(self._catalog[current_index]["main"])
                in self._asked_intents
            )
            current_requires_coverage_repair = (
                current_main_asked
                and self._allow_coverage_clarifications
                and not self._allow_follow_ups
                and self._question_has_evidence(current_index)
                and not self._question_resolved(current_index)
                and not self._question_has_asked_followup_intent(current_index)
            )
            skipped_current = (
                0 <= current_index < len(self._catalog)
                and not current_main_asked
                and self._question_skippable(current_index)
            )
            expected_start = (
                current_index
                if current_requires_coverage_repair
                else current_index + 1
                if current_main_asked or skipped_current
                else current_index
            )
            expected = self._peek_next_unresolved(expected_start)
            # Once the participant has supplied evidence for the active card
            # and the interviewer has already asked a focused intent on it,
            # trust the model's decision to move to the immediately following
            # main card. Replaying an unasked broad main at this point sounds
            # like the moderator ignored the conversation.
            if (
                expected == current_index
                and question_index == current_index + 1
                and self._question_has_evidence(current_index)
                and self._question_has_asked_followup_intent(current_index)
            ):
                expected = question_index
            if expected is None:
                self._order_corrections += 1
                return self._replacement(len(self._questions), "no_unresolved_question")
            if question_index != expected:
                self._order_corrections += 1
                reason = (
                    "skipped_unresolved_question"
                    if question_index > expected
                    else "backward_question"
                )
                if (
                    0 <= expected < len(self._questions)
                    and self._question_has_evidence(expected)
                    and not self._question_resolved(expected)
                    and not self._question_has_asked_followup_intent(expected)
                ):
                    focused = self._focused_repair_replacement(
                        expected, "partial_answer_repair"
                    )
                    if focused is not None:
                        return focused
                return self._replacement(expected, reason)
            skipped_count = question_index - expected_start + int(skipped_current)
            if skipped_count > 0:
                self._covered_questions_skipped += skipped_count
                contextual = self._contextual_question_text(question_index, text)
                if contextual != text:
                    text = contextual
                    policy_replaced = True
                    policy_reason = "task_context_after_skip"
        elif decision.action == "probe" and question_index != current_index:
            current_main_asked = (
                0 <= current_index < len(self._catalog)
                and str(self._catalog[current_index]["main"])
                in self._asked_intents
            )
            expected_start = current_index + 1 if current_main_asked else current_index
            expected = self._peek_next_unresolved(expected_start)
            target_evidence = set(self._catalog[question_index]["evidence"])
            has_prior_target_evidence = any(
                evidence_id in self._resolutions for evidence_id in target_evidence
            )
            has_current_evidence = self._question_has_evidence(current_index)
            has_asked_current_intent = self._question_has_asked_followup_intent(
                current_index
            )
            # A participant may have partially answered the first unresolved
            # downstream card before its broad main question was reached. In
            # that case preserve the model's narrow missing-evidence probe and
            # advance the cursor to that card, rather than replacing it with
            # wording that asks for the whole account again.
            if (
                (
                    expected == question_index
                    or (
                        question_index == current_index + 1
                        and has_current_evidence
                        and has_asked_current_intent
                    )
                )
                and has_prior_target_evidence
                and intent not in self._asked_intents
                and not self._question_resolved(question_index)
            ):
                self._asked_intents.add(intent)
                self._focused_advances_accepted += 1
                self._clear_pending_thread("focused_advance")
                self._record_event(
                    action="accept",
                    turn_action="focused_advance",
                    question=question_index + 1,
                    intent=intent,
                )
                return TurnReview(
                    text=text,
                    marker=question_index + 1,
                    decision=decision,
                    replaced=policy_replaced,
                    reason=policy_reason,
                    question_index=question_index,
                )

            self._order_corrections += 1
            return self._replacement(
                expected if expected is not None else len(self._questions),
                "probe_on_wrong_question",
            )

        if intent in self._asked_intents:
            self._duplicate_intents_blocked += 1
            return self._replacement(question_index + 1, "duplicate_intent")

        required_main_not_yet_asked = (
            decision.action == "main"
            and str(self._catalog[question_index]["main"])
            not in self._asked_intents
            and (
                not self._skip_answered_questions
                or self._question_protected(question_index)
            )
        )
        if self._question_resolved(question_index) and not required_main_not_yet_asked:
            if decision.action == "probe":
                self._resolved_probes_blocked += 1
            return self._replacement(question_index + 1, "question_resolved")

        self._asked_intents.add(intent)
        self._clear_pending_thread(
            "returned_to_anchor"
            if decision.action == "probe" and question_index == current_index
            else "advanced_in_order"
        )
        accepted_marker = decision.question if decision.action == "main" else marker
        self._record_event(
            action="accept",
            turn_action=decision.action,
            question=decision.question,
            intent=intent,
            resolutions=len(valid_resolutions),
        )
        return TurnReview(
            text=text,
            marker=accepted_marker,
            decision=decision,
            replaced=policy_replaced,
            reason=policy_reason,
            question_index=question_index,
        )

    def stats(self) -> dict[str, object]:
        statuses = {
            status: sum(1 for value in self._resolutions.values() if value == status)
            for status in sorted(_RESOLUTION_STATUSES)
        }
        return {
            "turn_decisions_missing": self._missing_decisions,
            "missing_decision_recoveries": self._missing_decision_recoveries,
            "inline_decision_recoveries": self._inline_decision_recoveries,
            "turn_decisions_invalid": self._invalid_decisions,
            "invalid_resolution_ids": sorted(self._invalid_resolution_ids),
            "terminal_resolutions_rejected": self._terminal_resolutions_rejected,
            "duplicate_intents_blocked": self._duplicate_intents_blocked,
            "resolved_probes_blocked": self._resolved_probes_blocked,
            "covered_questions_skipped": self._covered_questions_skipped,
            "semantic_advances": self._semantic_advances,
            "order_corrections": self._order_corrections,
            "threaded_probes_accepted": self._threaded_probes_accepted,
            "depth_threads_accepted": self._depth_threads_accepted,
            "threading_blocked": self._threading_blocked,
            "thread_resumes": self._thread_resumes,
            "thread_pending": self._pending_thread is not None,
            "focused_advances_accepted": self._focused_advances_accepted,
            "probe_policy_corrections": self._probe_policy_corrections,
            "probe_wording_corrections": self._probe_wording_corrections,
            "adaptive_probe_wording_accepted": (
                self._adaptive_probe_wording_accepted
            ),
            "emergent_probes_blocked": self._emergent_probes_blocked,
            "answer_repairs_accepted": self._answer_repairs_accepted,
            "answer_repairs_blocked": self._answer_repairs_blocked,
            "clarification_misclassifications_blocked": (
                self._clarification_misclassifications_blocked
            ),
            "orchestration_profile": self._orchestration_profile,
            "asked_intent_count": len(self._asked_intents),
            "tracked_evidence_count": len(self._resolutions),
            "resolved_evidence_count": sum(
                1 for value in self._resolutions.values() if value != "partial"
            ),
            "resolution_status_counts": statuses,
            "latest_focal_entity": self._latest_focal_entity,
            "latest_provenance": self._latest_provenance,
            "turn_events": list(self._events),
        }


# ── Stateful nudge processor (voice) ────────────────────────────────────────


class InterviewGuideProcessor(FrameProcessor):
    """Lightweight safety net for structured interviews.

    The structured prompt does the heavy lifting: it tells the LLM exactly
    which questions to ask and in what order. This processor is a backup that
    counts agent turns per question and, if the model lingers past the
    configured ``max_follow_ups``, injects a system message nudging it to
    advance to the next question (or to wrap up if we're at the end).

    It also exposes ``current_question_index`` so other components can read
    progress, and emits structured log lines for observability.

    The processor is intentionally conservative:
        - It never blocks frames; it just *appends* a system message when
          needed.
        - One initial agent ask + ``max_follow_ups`` follow-ups = the
          allowed budget per question. The nudge is fired on the next
          ``UserStoppedSpeakingFrame`` after the budget is exhausted, so the
          LLM sees the nudge while it's about to respond.
        - It pairs with the structured prompt; without that prompt the
          processor still works but the model may not know how to interpret
          the nudges.
    """

    def __init__(
        self,
        guide: dict,
        language: str = "en",
        name: str = "InterviewGuideProcessor",
    ):
        super().__init__(name=name)
        self._guide = guide or {}
        self._language = language or "en"
        self._questions = list(self._guide.get("questions") or [])
        self._orchestration = resolve_orchestration_settings(self._guide)
        self._allow_follow_ups = bool(self._orchestration["allow_follow_ups"])
        self._skip_answered_questions = bool(
            self._orchestration["skip_answered_questions"]
        )
        self._closing = self._guide.get("closing_message", DEFAULT_CLOSING)
        self.current_question_index = 0
        self._bot_turns_on_question = 0
        self._nudge_pending = False
        self._closed = False
        # Counters for the end-of-interview stats() summary.
        self._advances = 0
        self._nudges_injected = 0
        self._counted_bot_turns = 0
        self._evidence = EvidenceStateTracker(
            self._questions,
            self._closing,
            probe_selection=self._guide.get("probe_selection", "ordered"),
            allow_emergent_follow_ups=bool(
                self._orchestration["allow_emergent_follow_ups"]
            ),
            allow_follow_ups=self._allow_follow_ups,
            skip_answered_questions=self._skip_answered_questions,
            allow_coverage_clarifications=bool(
                self._orchestration["allow_coverage_clarifications"]
            ),
            allow_threading=bool(self._orchestration["allow_threading"]),
            language=self._language,
            orchestration_profile=str(self._orchestration["profile"]),
        )
        # Bot turns that happen *before* the participant has spoken (the
        # spoken welcome message in particular) used to be counted against
        # question 1's follow-up budget, which made the agent advance after
        # the very first probe. We gate counting on having seen at least one
        # ``UserStoppedSpeakingFrame`` so the welcome is free.
        self._user_has_spoken = False
        # Everything the participant said since the last counted bot turn.
        # Used to recognize clarification requests so neither the budget
        # counter nor the advance/close nudge punishes them.
        self._user_buffer: list[str] = []
        self._awaiting_interruption_confirmation = False
        self._last_question_text = (
            str(self._questions[0].get("text") or "").strip()
            if self._questions
            else ""
        )

    # ── Public read-only helpers ────────────────────────────────────────

    @property
    def total_questions(self) -> int:
        return len(self._questions)

    @property
    def is_finished(self) -> bool:
        return self._closed or self.current_question_index >= len(self._questions)

    @property
    def current_prompt_text(self) -> str:
        """Exact suspended prompt used after a voice interruption."""
        if self._last_question_text:
            return self._last_question_text
        if 0 <= self.current_question_index < len(self._questions):
            return str(
                self._questions[self.current_question_index].get("text") or ""
            ).strip()
        return ""

    def _current_max_follow_ups(self) -> int:
        if not self._allow_follow_ups:
            return 0
        if self.current_question_index >= len(self._questions):
            return DEFAULT_MAX_FOLLOW_UPS
        return question_max_follow_ups(self._questions[self.current_question_index])

    def _build_advance_message(self) -> Optional[dict]:
        """Nudge asking the LLM to move on, flipping _closed on the last one."""
        msg, is_closing = build_protocol_guidance(
            self._questions, self._closing, self.current_question_index
        )
        if is_closing:
            self._closed = True
        return msg

    def sync_to_marker(self, marker: Optional[int]) -> None:
        """Follow a model-emitted main-question marker, forward only.

        This keeps the voice turn counter aligned when the model moves on
        before exhausting a question's maximum follow-up budget.
        """
        if marker is None:
            return
        idx = marker - 1
        safe_idx = self._evidence.safe_marker_index(idx)
        if idx > safe_idx:
            logger.warning(
                "[guide] blocked marker jump q={} -> q={}; next safe q={}",
                self.current_question_index + 1,
                idx + 1,
                safe_idx + 1,
            )
            idx = safe_idx
        if 0 <= idx < len(self._questions) and idx > self.current_question_index:
            logger.debug(
                "[guide] marker sync q={} -> q={}",
                self.current_question_index + 1,
                idx + 1,
            )
            self.current_question_index = idx
            self._bot_turns_on_question = 0
            self._nudge_pending = False

    def review_turn(
        self,
        text: str,
        marker: Optional[int],
        decision: Optional[TurnDecision],
        participant_text: str = "",
    ) -> TurnReview:
        """Validate semantic turn state before modular voice reaches TTS."""
        if not participant_text:
            participant_text = " ".join(self._user_buffer)
        participant_kind = classify_participant_turn(
            participant_text, self._language
        )
        if self._awaiting_interruption_confirmation:
            if looks_like_procedural_resume(participant_text):
                self._awaiting_interruption_confirmation = False
                return TurnReview(
                    text=procedural_resume_response(self.current_prompt_text),
                    marker=None,
                    decision=TurnDecision(
                        action="clarify",
                        question=(
                            self.current_question_index + 1
                            if self.current_question_index < len(self._questions)
                            else None
                        ),
                    ),
                    replaced=True,
                    reason="interruption_resumed",
                    question_index=self.current_question_index,
                )
            if (
                participant_kind == "answer"
                and not looks_like_more_procedural_help(participant_text)
            ):
                # A participant may answer the suspended research question
                # directly instead of saying "continue".
                self._awaiting_interruption_confirmation = False
        if not self._allow_follow_ups and not self._skip_answered_questions:
            if decision is not None and decision.not_applicable_question is not None:
                conditional_review = self._evidence.review(
                    text,
                    marker,
                    decision,
                    self.current_question_index,
                    participant_text,
                )
                if conditional_review.reason == "module_not_applicable":
                    if conditional_review.closed:
                        self._closed = True
                    elif conditional_review.question_index is not None:
                        self.sync_to_marker(conditional_review.question_index + 1)
                    return conditional_review
            if self._closed or not self._questions:
                return TurnReview(
                    text=self._closing,
                    marker=None,
                    decision=TurnDecision(action="close"),
                    replaced=str(text or "").strip() != str(self._closing).strip(),
                    reason="fixed_closing_enforced",
                    question_index=len(self._questions),
                    closed=True,
                )
            if (
                decision is not None
                and decision.action == "clarify"
                and participant_kind
                in {"procedural", "flow_repair", "clarification", "continuation"}
            ):
                visible = str(text or "").strip()
                if participant_kind in {"clarification", "procedural"}:
                    visible = clarification_pause_response(visible) or visible
                    self._awaiting_interruption_confirmation = True
                return TurnReview(
                    text=visible,
                    marker=None,
                    decision=decision,
                    question_index=self.current_question_index,
                )
            index = min(self.current_question_index, len(self._questions) - 1)
            replacement = str(self._questions[index].get("text") or "").strip()
            self._last_question_text = replacement
            exact_decision = TurnDecision(
                action="main",
                question=index + 1,
                intent=question_intents(self._questions[index], index)["main"],
            )
            changed = (
                str(text or "").strip() != replacement
                or marker != index + 1
                or decision != exact_decision
            )
            return TurnReview(
                text=replacement,
                marker=index + 1,
                decision=exact_decision,
                replaced=changed,
                reason="fixed_question_enforced" if changed else None,
                question_index=index,
            )
        review = self._evidence.review(
            text, marker, decision, self.current_question_index, participant_text
        )
        if (
            participant_kind in {"clarification", "procedural"}
            and review.decision is not None
            and review.decision.action == "clarify"
        ):
            self._awaiting_interruption_confirmation = True
        if review.closed:
            self._closed = True
        elif review.question_index is not None:
            self.sync_to_marker(review.question_index + 1)
        visible = str(review.text or "").strip()
        if review.decision is not None and review.decision.action in {
            "main",
            "probe",
            "thread",
        }:
            question_end = visible.find("?")
            if question_end >= 0:
                self._last_question_text = visible[: question_end + 1]
        return review

    # ── Frame handling ──────────────────────────────────────────────────

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            self.current_question_index = 0
            self._bot_turns_on_question = 0
            self._nudge_pending = False
            self._closed = False
            self._user_has_spoken = False
            self._user_buffer = []
            self._awaiting_interruption_confirmation = False
            self._last_question_text = (
                str(self._questions[0].get("text") or "").strip()
                if self._questions
                else ""
            )
            self._evidence = EvidenceStateTracker(
                self._questions,
                self._closing,
                probe_selection=self._guide.get("probe_selection", "ordered"),
                allow_emergent_follow_ups=bool(
                    self._orchestration["allow_emergent_follow_ups"]
                ),
                allow_follow_ups=self._allow_follow_ups,
                skip_answered_questions=self._skip_answered_questions,
                allow_coverage_clarifications=bool(
                    self._orchestration["allow_coverage_clarifications"]
                ),
                allow_threading=bool(self._orchestration["allow_threading"]),
                language=self._language,
                orchestration_profile=str(self._orchestration["profile"]),
            )

        # Accumulate user speech so we can tell answers apart from
        # clarification requests. The buffer is cleared whenever a bot turn
        # is evaluated, so at any point it holds "what the participant said
        # since the agent last finished speaking".
        elif isinstance(frame, TranscriptionFrame):
            if frame.text and frame.text.strip():
                self._user_buffer.append(frame.text.strip())

        # Count agent turns. Two gates so the budget tracks *probes* and
        # not every TTS event:
        #   1. The participant must have spoken at least once. This stops
        #      the spoken welcome from being charged against question 1.
        #   2. The participant must have spoken since the last counted bot
        #      turn. This stops silence prompts ("Take your time...") and
        #      any other TTSSpeakFrame the agent emits while waiting on
        #      the user from being charged as additional probes. Without
        #      this gate, two silence prompts could exhaust a question's
        #      whole follow-up budget without the model ever asking a real
        #      probe, which is what made the structured template feel
        #      "stuck on the same question".
        elif (
            isinstance(frame, BotStoppedSpeakingFrame)
            and not self._closed
            and self._user_has_spoken
        ):
            # The buffer holds the user turn this bot reply responded to.
            user_text = " ".join(self._user_buffer)
            self._user_buffer = []
            # Lock the counter until the participant speaks again. The
            # next TTS event (silence prompt, repeated greeting, etc.)
            # won't be counted unless a real user turn happens first.
            self._user_has_spoken = False
            if is_participant_interruption(user_text, self._language):
                # The agent was answering "what do you mean?" / repeating
                # itself — that is not a probe, so it must not burn the
                # question's follow-up budget.
                logger.debug(
                    "[guide] q={} bot turn not counted "
                    "(reply to clarification: {!r})",
                    self.current_question_index + 1,
                    user_text[:80],
                )
            else:
                self._bot_turns_on_question += 1
                self._counted_bot_turns += 1
                budget = 1 + self._current_max_follow_ups()
                logger.debug(
                    "[guide] q={} agent_turn={}/{}",
                    self.current_question_index + 1,
                    self._bot_turns_on_question,
                    budget,
                )
                if self._bot_turns_on_question >= budget:
                    # repair_only is evidence-state driven: the same model
                    # turn chooses the next unresolved card and the controller
                    # validates it. A blind budget nudge on every zero-probe
                    # question created one forced advance per answer and moved
                    # the cursor before that answer was classified.
                    self._nudge_pending = (
                        self._orchestration["profile"] != "repair_only"
                    )

        # When the user finishes their reply and a nudge is pending, push a
        # system message before the LLM context aggregator triggers a run.
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._user_has_spoken = True
            if self._nudge_pending:
                # If what the participant just said is a clarification or
                # repeat request, hold the nudge: advancing (or worse,
                # closing) now would steamroll them. The model answers the
                # clarification this turn — uncounted, see above — and the
                # nudge fires on their next substantive turn instead.
                current_text = " ".join(self._user_buffer)
                if is_participant_interruption(current_text, self._language):
                    logger.info(
                        "[guide] q={} holding nudge — participant asked a "
                        "clarification: {!r}",
                        self.current_question_index + 1,
                        current_text[:80],
                    )
                    await self.push_frame(frame, direction)
                    return
                msg = self._build_advance_message()
                if msg is not None:
                    self._nudges_injected += 1
                    logger.info(
                        "[guide] advancing from q={} -> q={}",
                        self.current_question_index + 1,
                        self.current_question_index + 2,
                    )
                    await self.push_frame(
                        LLMMessagesAppendFrame(messages=[msg], run_llm=False),
                        FrameDirection.DOWNSTREAM,
                    )
                    if not self._closed:
                        self.current_question_index += 1
                        self._bot_turns_on_question = 0
                        self._advances += 1
                    else:
                        logger.info(
                            "[guide] protocol summary: {}", self.stats()
                        )
                self._nudge_pending = False

        await self.push_frame(frame, direction)

    # ── Test/inspection helper ─────────────────────────────────────────

    def snapshot(self) -> dict:
        return {
            "current_question_index": self.current_question_index,
            "bot_turns_on_question": self._bot_turns_on_question,
            "nudge_pending": self._nudge_pending,
            "closed": self._closed,
            "total_questions": self.total_questions,
            "user_has_spoken": self._user_has_spoken,
        }

    def stats(self) -> dict:
        """End-of-interview summary. High forced_advances/nudges_injected
        relative to total_questions means the model needed a lot of pushing."""
        reached = (
            self.total_questions
            if self._closed
            else min(self.current_question_index + 1, self.total_questions)
        )
        return {
            "channel": "voice",
            "total_questions": self.total_questions,
            "questions_reached": reached,
            "completed": self._closed,
            "forced_advances": self._advances,
            "nudges_injected": self._nudges_injected,
            "counted_bot_turns": self._counted_bot_turns,
            **self._evidence.stats(),
        }


# ── Stateful controller (text chat) ─────────────────────────────────────────


class TextStructuredController:
    """The InterviewGuideProcessor + StructuredOutputFilter guardrails, for
    text chat. The text loop only fed the structured prompt to the model and
    hoped it complied, so looping, never advancing, and cramming several
    questions into one reply all went unchecked. This fixes that.

    The text loop alternates assistant/user with no silence prompts or split
    STT, so counting is simpler than the voice processor (no user_has_spoken
    gate). The welcome message is appended by the loop, not run through
    register_bot_turn, so it stays free like in voice.

    Position comes from the turn-budget counter and from sync_to_marker, which
    trusts the model's hidden [[Qn]] tag when it's there. The tag wins: it's
    the model saying which question it just asked, so the counting and the
    clarification regex defer to it.
    """

    def __init__(self, guide: dict, language: str = "en"):
        guide = guide or {}
        self._questions = list(guide.get("questions") or [])
        self._closing = guide.get("closing_message", DEFAULT_CLOSING)
        self._orchestration = resolve_orchestration_settings(guide)
        self._allow_follow_ups = bool(self._orchestration["allow_follow_ups"])
        self._skip_answered_questions = bool(
            self._orchestration["skip_answered_questions"]
        )
        self._language = language or "en"
        self.current_question_index = 0
        self._bot_turns_on_question = 0
        self._nudge_pending = False
        self._closed = False
        self._advances = 0
        self._nudges_injected = 0
        self._counted_bot_turns = 0
        self._max_index_reached = 0
        self._question_indices_asked: set[int] = set()
        self._time_constrained = False
        self._evidence = EvidenceStateTracker(
            self._questions,
            self._closing,
            probe_selection=guide.get("probe_selection", "ordered"),
            allow_emergent_follow_ups=bool(
                self._orchestration["allow_emergent_follow_ups"]
            ),
            allow_follow_ups=self._allow_follow_ups,
            skip_answered_questions=self._skip_answered_questions,
            allow_coverage_clarifications=bool(
                self._orchestration["allow_coverage_clarifications"]
            ),
            allow_threading=bool(self._orchestration["allow_threading"]),
            language=self._language,
            orchestration_profile=str(self._orchestration["profile"]),
        )
        self._last_question_text = (
            str(self._questions[0].get("text") or "").strip()
            if self._questions
            else ""
        )

    @property
    def total_questions(self) -> int:
        return len(self._questions)

    @property
    def is_finished(self) -> bool:
        return self._closed or self.current_question_index >= len(self._questions)

    @property
    def current_prompt_text(self) -> str:
        """Exact most-recent question, used to resume after interruptions."""
        if self._last_question_text:
            return self._last_question_text
        if 0 <= self.current_question_index < len(self._questions):
            return str(
                self._questions[self.current_question_index].get("text") or ""
            ).strip()
        return ""

    @property
    def closing_message(self) -> str:
        return str(self._closing or DEFAULT_CLOSING).strip()

    def begin_time_limited_final_question(self) -> tuple[str, int] | None:
        """Select one remaining card for a clearly signposted final turn.

        Required cards are preferred when guides distinguish priorities.  The
        controller is advanced to the selected card so ordinary review and
        progress reporting remain coherent if the participant asks for a
        clarification before answering.
        """
        if self._closed or not self._questions:
            return None
        first_remaining = self.current_question_index + (
            1 if self._bot_turns_on_question > 0 else 0
        )
        remaining = list(range(first_remaining, len(self._questions)))
        if not remaining:
            return None
        selected = next(
            (
                index
                for index in remaining
                if str(self._questions[index].get("priority") or "required")
                == "required"
            ),
            remaining[0],
        )
        self.current_question_index = selected
        self._max_index_reached = max(self._max_index_reached, selected)
        self._bot_turns_on_question = 0
        self._nudge_pending = False
        question = str(self._questions[selected].get("text") or "").strip()
        self._last_question_text = question
        self._question_indices_asked.add(selected)
        if not question:
            return None
        return question, selected + 1

    def close_for_time(self) -> str:
        """Mark the protocol closed and return its participant-facing close."""
        self._time_constrained = True
        self._closed = True
        self.current_question_index = len(self._questions)
        self._nudge_pending = False
        return self.closing_message

    def _current_max_follow_ups(self) -> int:
        if not self._allow_follow_ups:
            return 0
        if self.current_question_index >= len(self._questions):
            return DEFAULT_MAX_FOLLOW_UPS
        return question_max_follow_ups(self._questions[self.current_question_index])

    def maybe_advance_message(self, user_text: str) -> Optional[dict]:
        """Call when a new participant message arrives, before the LLM call.
        If the last bot turn used up the question's budget and this answer is
        substantive, advance and return the guidance note to append to the
        context. A clarification holds the nudge; otherwise returns None.
        """
        if not self._nudge_pending or self._closed:
            return None
        if is_participant_interruption(user_text, self._language):
            logger.info(
                "[guide:text] q={} holding nudge, participant asked a "
                "clarification: {!r}",
                self.current_question_index + 1,
                (user_text or "")[:80],
            )
            return None
        msg, is_closing = build_protocol_guidance(
            self._questions, self._closing, self.current_question_index
        )
        self._nudge_pending = False
        if msg is None:
            return None
        self._nudges_injected += 1
        if is_closing:
            self._closed = True
            logger.info("[guide:text] protocol summary: {}", self.stats())
        else:
            logger.info(
                "[guide:text] advancing from q={} -> q={}",
                self.current_question_index + 1,
                self.current_question_index + 2,
            )
            self.current_question_index += 1
            self._max_index_reached = max(
                self._max_index_reached, self.current_question_index
            )
            self._bot_turns_on_question = 0
            self._advances += 1
        return msg

    def sync_to_marker(self, marker: Optional[int]) -> None:
        """Move position to the model's [[Qn]] tag (1-based), forward only.
        A marker at or behind the current question is ignored. If the model
        moved on by itself, drop the pending nudge so we don't fight it.
        """
        if marker is None:
            return
        idx = marker - 1
        safe_idx = self._evidence.safe_marker_index(idx)
        if idx > safe_idx:
            logger.warning(
                "[guide:text] blocked marker jump q={} -> q={}; next safe q={}",
                self.current_question_index + 1,
                idx + 1,
                safe_idx + 1,
            )
            idx = safe_idx
        if 0 <= idx < len(self._questions) and idx > self.current_question_index:
            logger.debug(
                "[guide:text] marker sync q={} -> q={}",
                self.current_question_index + 1,
                idx + 1,
            )
            self.current_question_index = idx
            self._max_index_reached = max(self._max_index_reached, idx)
            self._bot_turns_on_question = 0
            self._nudge_pending = False

    def review_turn(
        self,
        text: str,
        marker: Optional[int],
        decision: Optional[TurnDecision],
        participant_text: str = "",
    ) -> TurnReview:
        """Validate the model's proposed turn and replace semantic repeats."""
        if not self._allow_follow_ups and not self._skip_answered_questions:
            if decision is not None and decision.not_applicable_question is not None:
                conditional_review = self._evidence.review(
                    text,
                    marker,
                    decision,
                    self.current_question_index,
                    participant_text,
                )
                if conditional_review.reason == "module_not_applicable":
                    if conditional_review.closed:
                        self._closed = True
                        self.current_question_index = len(self._questions)
                    elif conditional_review.question_index is not None:
                        self.current_question_index = conditional_review.question_index
                        self._last_question_text = conditional_review.text
                    return conditional_review
            if self._closed or not self._questions:
                self.current_question_index = len(self._questions)
                return TurnReview(
                    text=self._closing,
                    marker=None,
                    decision=TurnDecision(action="close"),
                    replaced=str(text or "").strip() != str(self._closing).strip(),
                    reason="fixed_closing_enforced",
                    question_index=len(self._questions),
                    closed=True,
                )
            participant_kind = classify_participant_turn(
                participant_text, self._language
            )
            if (
                decision is not None
                and decision.action == "clarify"
                and participant_kind
                in {"procedural", "flow_repair", "clarification", "continuation"}
            ):
                visible = str(text or "").strip()
                if participant_kind == "clarification":
                    visible = clarification_pause_response(visible) or visible
                return TurnReview(
                    text=visible,
                    marker=None,
                    decision=decision,
                    question_index=self.current_question_index,
                )
            index = min(self.current_question_index, len(self._questions) - 1)
            visible = str(self._questions[index].get("text") or "").strip()
            exact_decision = TurnDecision(
                action="main",
                question=index + 1,
                intent=question_intents(self._questions[index], index)["main"],
            )
            changed = (
                str(text or "").strip() != visible
                or marker != index + 1
                or decision != exact_decision
            )
            self._last_question_text = visible
            self._question_indices_asked.add(index)
            return TurnReview(
                text=visible,
                marker=index + 1,
                decision=exact_decision,
                replaced=changed,
                reason="fixed_question_enforced" if changed else None,
                question_index=index,
            )
        participant_kind = classify_participant_turn(
            participant_text, self._language
        )
        original_decision = decision
        if participant_kind in {
            "procedural",
            "flow_repair",
            "clarification",
            "continuation",
        }:
            # A model-authored main/probe marker cannot override a participant
            # interruption. Keeping this guard in the controller protects
            # every caller, even if an API route forgets to intercept the turn.
            anchor = self.current_question_index
            safe_decision = TurnDecision(
                action="clarify",
                question=(anchor + 1) if anchor < len(self._questions) else None,
                intent=None,
            )
            reviewed_text = text
            replaced = False
            reason = None
            # A clarification cannot be used as a bridge into another guide
            # card. If the model emits valid clarification metadata, preserve
            # its declarative answer but replace any appended guide question
            # with a standard confirmation pause. If its metadata escapes the
            # suspended card entirely, resume with one bounded repair.
            clarification_escaped = (
                participant_kind == "clarification"
                and (
                    original_decision is None
                    or original_decision.action in {"main", "thread", "close"}
                    or original_decision.question != anchor + 1
                )
            )
            if clarification_escaped:
                resume = self._evidence.clarification_resume_text(anchor)
                reviewed_text = (
                    resume
                    if resume
                    else "Could you say a little more about what was unclear?"
                )
                replaced = True
                reason = "clarification_advance_blocked"
            elif participant_kind == "clarification":
                paused = clarification_pause_response(reviewed_text)
                if paused:
                    if paused != reviewed_text:
                        replaced = True
                        reason = "clarification_paused"
                    reviewed_text = paused
                else:
                    resume = self._evidence.clarification_resume_text(anchor)
                    reviewed_text = resume or (
                        "Could you say a little more about what was unclear?"
                    )
                    replaced = True
                    reason = "clarification_answer_missing"
            review = TurnReview(
                text=reviewed_text,
                marker=None,
                decision=safe_decision,
                replaced=replaced,
                reason=reason,
                question_index=anchor,
            )
        else:
            review = self._evidence.review(
                text, marker, decision, self.current_question_index, participant_text
            )
        if review.closed:
            self._closed = True
            self.current_question_index = len(self._questions)
            self._nudge_pending = False
        elif review.question_index is not None:
            self._question_indices_asked.add(review.question_index)
            self.sync_to_marker(review.question_index + 1)
        visible = (review.text or "").strip()
        if review.decision is None or review.decision.action in {
            "main",
            "probe",
            "thread",
        }:
            question_end = visible.find("?")
            if question_end >= 0:
                self._last_question_text = visible[: question_end + 1]
        return review

    def register_bot_turn(self, prompting_user_text: str) -> None:
        """Count one agent reply against the question's budget. Call after each
        LLM reply with the message that prompted it; replies to a clarification
        don't count, so those exchanges don't burn the budget.
        """
        if self._closed:
            return
        if is_participant_interruption(prompting_user_text, self._language):
            logger.debug(
                "[guide:text] q={} bot turn not counted (reply to "
                "clarification: {!r})",
                self.current_question_index + 1,
                (prompting_user_text or "")[:80],
            )
            return
        self._bot_turns_on_question += 1
        self._counted_bot_turns += 1
        budget = 1 + self._current_max_follow_ups()
        logger.debug(
            "[guide:text] q={} agent_turn={}/{}",
            self.current_question_index + 1,
            self._bot_turns_on_question,
            budget,
        )
        if self._bot_turns_on_question >= budget:
            self._nudge_pending = (
                self._orchestration["profile"] != "repair_only"
            )

    def stats(self) -> dict:
        reached = len(self._question_indices_asked)
        missing = [
            index + 1
            for index in range(self.total_questions)
            if index not in self._question_indices_asked
        ]
        return {
            "channel": "text",
            "total_questions": self.total_questions,
            "questions_reached": reached,
            "completed": self._closed,
            "time_constrained": self._time_constrained,
            "missing_question_numbers": missing,
            "forced_advances": self._advances,
            "nudges_injected": self._nudges_injected,
            "counted_bot_turns": self._counted_bot_turns,
            **self._evidence.stats(),
        }

    def snapshot(self) -> dict:
        return {
            "current_question_index": self.current_question_index,
            "bot_turns_on_question": self._bot_turns_on_question,
            "nudge_pending": self._nudge_pending,
            "closed": self._closed,
            "total_questions": self.total_questions,
        }


__all__ = [
    "build_structured_prompt",
    "control_marker_prefix",
    "format_turn_marker",
    "classify_participant_turn",
    "clarification_pause_response",
    "is_participant_interruption",
    "looks_like_clarification",
    "looks_like_more_procedural_help",
    "looks_like_procedural_resume",
    "participant_interruption_response",
    "procedural_resume_response",
    "parse_turn_decision",
    "question_intents",
    "strip_control_markers",
    "strip_progress_marker",
    "strip_turn_marker",
    "turn_decision_instruction",
    "enforce_one_question_per_turn",
    "build_protocol_guidance",
    "question_max_follow_ups",
    "InterviewGuideProcessor",
    "StructuredOutputFilter",
    "TextStructuredController",
    "TurnDecision",
    "TurnReview",
    "DEFAULT_MAX_FOLLOW_UPS",
    "DEFAULT_CLOSING",
]
