"""Pluggable object-name matchers for grading VLM predictions against GT strings.

This package lives at the top level (not under ``metrics/``) on purpose:
``metrics/__init__`` eagerly imports torch (via throne/faithscore), so anything
that needs to run offline — the ROPE log-rescore, the synonym regrade — must be
importable without torch/PIL/GPU. Nothing here imports torch at module load;
the optional LLM judge lazy-imports it only when actually used.

Public objects:
    Matcher, MatchResult, KEY_FOR, PER_POS_KEY_FOR, normalize   (base)
    ExactMatcher, SubstringMatcher                              (exact)
    WordNetMatcher                                              (wordnet)
    SynonymMatcher, MissingPairError                            (synonym)
    PairCache                                                   (cache)
    LLMJudge, FlanJudge, DEFAULT_PROMPTS, PROMPT_VERSION        (judge, dormant)
"""
from .base import Matcher, MatchResult, KEY_FOR, PER_POS_KEY_FOR, normalize
from .exact import ExactMatcher, SubstringMatcher
from .wordnet import WordNetMatcher
from .synonym import SynonymMatcher, MissingPairError
from .cache import PairCache

__all__ = [
    "Matcher", "MatchResult", "KEY_FOR", "PER_POS_KEY_FOR", "normalize",
    "ExactMatcher", "SubstringMatcher", "WordNetMatcher",
    "SynonymMatcher", "MissingPairError", "PairCache",
]
