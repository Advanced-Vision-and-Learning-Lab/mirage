"""SynonymMatcher — tiered, add-on matcher for the ``overall_accuracy_synonym`` column.

Tiers (first hit wins), so ``synonym`` is a strict superset of ``lenient`` and the
``synonym >= lenient`` invariant holds by construction:

    1. base    (SubstringMatcher)  — free lenient check
    2. wordnet (WordNetMatcher)    — active default tier (offline, deterministic)
    3. judge   (LLMJudge|cache)    — DORMANT; only consulted if a judge or cache
                                     is provided (issue #27 ablation / --build-cache)

Default construction — ``SynonymMatcher()`` — uses Substring + WordNet only, which
is exactly the shipped, precision-1.0 config from the Phase-0 spike.
"""
from __future__ import annotations

from typing import Optional

from .base import MatchResult, Matcher
from .exact import SubstringMatcher
from .wordnet import WordNetMatcher


class MissingPairError(KeyError):
    """Offline judge mode: a residual pair is absent from the cache."""


class SynonymMatcher:
    name = "synonym"

    def __init__(
        self,
        base: Optional[Matcher] = None,
        wordnet: Optional[WordNetMatcher] = None,
        cache=None,
        judge=None,
        mode: str = "wordnet",
    ):
        # mode in {"wordnet", "trusting"}. "wordnet" (default) = base+WordNet,
        # no LLM. "trusting" additionally consults the judge tier when present.
        if mode not in ("wordnet", "trusting"):
            raise ValueError(f"unknown mode {mode!r}")
        self.base = base if base is not None else SubstringMatcher()
        self.wordnet = wordnet if wordnet is not None else WordNetMatcher()
        self.cache = cache
        self.judge = judge
        self.mode = mode

    def match(self, pred: str, gt: str) -> bool:
        return self.explain(pred, gt).matched

    def explain(self, pred: str, gt: str) -> MatchResult:
        if self.base.match(pred, gt):
            return MatchResult(True, "base")
        if self.wordnet is not None and self.wordnet.match(pred, gt):
            return MatchResult(True, "wordnet")
        # Dormant unless a judge or cache was supplied.
        if self.mode == "trusting" and (self.judge is not None or self.cache is not None):
            verdict = self._judge_verdict(pred, gt)
            if verdict is not None and verdict.get("yes"):
                return MatchResult(True, "judge", verdict.get("votes", 0))
        return MatchResult(False, "none")

    def _judge_verdict(self, pred: str, gt: str):
        if self.cache is not None:
            hit = self.cache.get(pred, gt)
            if hit is not None:
                return hit
        if self.judge is not None:  # build mode: compute now, cache if we have one
            verdict = self.judge.judge_batch([(pred, gt)])[0]
            if self.cache is not None:
                self.cache.put(pred, gt, verdict)
            return verdict
        # offline mode: cache present but pair unknown, no judge to fall back on
        raise MissingPairError(f"{pred!r} vs {gt!r} not in cache")
