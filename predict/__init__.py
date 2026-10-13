"""Predict layer: unified prediction interface with transparent caching.

Architecture::

    Experiment Driver -> Predictor -> { SQLite cache | LocalModelBackend | future APIBackend }

Torch-free at import time so offline scripts can use the cache without a
GPU environment. Bump :data:`CACHE_KEY_VERSION` (in ``predict.identity``)
to invalidate cached responses after semantic changes to generation or
post-processing.
"""

from predict.backends import CallableBackend, LocalModelBackend
from predict.cache import SQLiteResponseCache, default_cache_path
from predict.factory import build_model, build_predictor
from predict.identity import CACHE_KEY_VERSION
from predict.predictor import Predictor

__all__ = [
    "CACHE_KEY_VERSION",
    "CallableBackend",
    "LocalModelBackend",
    "Predictor",
    "SQLiteResponseCache",
    "build_model",
    "build_predictor",
    "default_cache_path",
]
