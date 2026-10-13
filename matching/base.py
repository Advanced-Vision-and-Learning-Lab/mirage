"""Matcher protocol, result type, and the name->result-key maps.

A ``Matcher`` decides whether a model prediction should count as correct for a
given ground-truth class string. Metrics accept a list of matchers and emit one
accuracy column per matcher, keyed by ``KEY_FOR[matcher.name]``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


def normalize(s: str) -> str:
    """Canonical form used by every matcher: strip + lowercase."""
    return s.strip().lower()


# Overall-accuracy JSON key each named matcher writes. Existing ROPE keys
# (overall_accuracy = lenient headline, overall_accuracy_strict) are preserved
# verbatim so previously-committed results stay byte-identical.
KEY_FOR = {
    "strict": "overall_accuracy_strict",
    "lenient": "overall_accuracy",
    "synonym": "overall_accuracy_synonym",
}

PER_POS_KEY_FOR = {
    "strict": "per_position_accuracy_strict",
    "lenient": "per_position_accuracy",
    "synonym": "per_position_accuracy_synonym",
}


@runtime_checkable
class Matcher(Protocol):
    name: str

    def match(self, pred: str, gt: str) -> bool:  # pragma: no cover - protocol
        ...


@dataclass
class MatchResult:
    """Outcome of a single match, with the tier that decided it (flip-audit)."""
    matched: bool
    tier: str
    votes: int = 0
