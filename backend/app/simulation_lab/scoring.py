from __future__ import annotations

import math
import re
from collections import Counter
from difflib import SequenceMatcher

from .models import SimilarityScores


TOKEN_RE = re.compile(r"[a-z0-9']+")
HEDGES = {"maybe", "guess", "probably", "kind", "like", "just", "really", "know"}
SELF_DISCLOSURE = {"i", "i'm", "i've", "me", "my"}


def _tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _cosine(left: list[str], right: list[str]) -> float:
    a, b = Counter(left), Counter(right)
    numerator = sum(a[token] * b[token] for token in a.keys() & b.keys())
    denom = math.sqrt(sum(v * v for v in a.values()) * sum(v * v for v in b.values()))
    return numerator / denom if denom else 0.0


def _length_match(a: list[str], b: list[str]) -> float:
    if not a or not b:
        return 0.0
    return min(len(a), len(b)) / max(len(a), len(b))


def _feature_vector(text: str, tokens: list[str]) -> list[bool]:
    return [
        "?" in text,
        any(token in HEDGES for token in tokens),
        any(token in SELF_DISCLOSURE for token in tokens),
        any(mark in text for mark in ["don't", "can't", "isn't", "won't", "didn't", "i'm", "i've"]),
        len(tokens) <= 25,
        len(re.findall(r"[.!?]+", text)) <= 2,
    ]


def score_response(candidate: str, reference: str) -> SimilarityScores:
    cand_tokens, ref_tokens = _tokens(candidate), _tokens(reference)
    union = set(cand_tokens) | set(ref_tokens)
    overlap = len(set(cand_tokens) & set(ref_tokens)) / len(union) if union else 0.0
    phrase = SequenceMatcher(None, candidate.lower(), reference.lower()).ratio()
    length = _length_match(cand_tokens, ref_tokens)
    cand_features, ref_features = _feature_vector(candidate.lower(), cand_tokens), _feature_vector(reference.lower(), ref_tokens)
    behavior = sum(a == b for a, b in zip(cand_features, ref_features, strict=True)) / len(ref_features)
    style = 0.55 * behavior + 0.45 * _cosine(cand_tokens, ref_tokens)
    composite = 0.20 * overlap + 0.20 * phrase + 0.15 * length + 0.20 * style + 0.25 * behavior
    values = [composite, overlap, phrase, length, style, behavior]
    rounded = [round(max(0.0, min(1.0, value)), 3) for value in values]
    return SimilarityScores(
        composite=rounded[0], token_overlap=rounded[1], phrase_similarity=rounded[2],
        length_match=rounded[3], style_match=rounded[4], behavior_match=rounded[5],
    )
