"""Exact and substring matchers — bodies moved verbatim from metrics/rope.py.

ExactMatcher   == the old ``_rope_match_strict`` (canonical ROPE, exact equality).
SubstringMatcher == the old ``_rope_match`` (lenient whole-word substring).

Keeping the logic identical is what guarantees the no-regression gate: the
default matcher set [SubstringMatcher, ExactMatcher] reproduces the committed
``overall_accuracy`` / ``overall_accuracy_strict`` byte-for-byte.
"""
from __future__ import annotations

import re

from .base import normalize


class ExactMatcher:
    """Strict ROPE match: exact (case-insensitive) equality."""
    name = "strict"

    def match(self, pred: str, gt: str) -> bool:
        return normalize(pred) == normalize(gt)


class SubstringMatcher:
    """Lenient ROPE match: GT class appears as a whole-word substring of the
    prediction (or vice versa). Tolerates "wooden cabinet" vs GT "cabinet"."""
    name = "lenient"

    def match(self, pred: str, gt: str) -> bool:
        p = normalize(pred)
        t = normalize(gt)
        if not p or not t:
            return False
        if p == t:
            return True
        t_pattern = r"\b" + re.escape(t) + r"\b"
        p_pattern = r"\b" + re.escape(p) + r"\b"
        return bool(re.search(t_pattern, p)) or bool(re.search(p_pattern, t))
