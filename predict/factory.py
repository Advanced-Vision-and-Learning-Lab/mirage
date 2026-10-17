"""Factory helpers replacing the per-driver model-selection blocks.

Model classes are imported lazily inside :func:`build_model` so this
module (and the whole ``predict`` package) stays importable without torch.
"""

import importlib
import os
from typing import Any, Optional

from predict.backends import LocalModelBackend
from predict.cache import SQLiteResponseCache
from predict.predictor import Predictor

_REGISTRY = {
    "paligemma": ("models.paligemma", "PaligemmaModel",
                  "MIRAGE_PALIGEMMA_MODEL_PATH", "google/paligemma-3b-mix-224"),
    "qwen2vl": ("models.qwen2vl", "Qwen2VLModel",
                "MIRAGE_QWEN_MODEL_PATH", "Qwen/Qwen2-VL-2B-Instruct"),
    "gemma4-e4b": ("models.hf_chat", "HFChatVLM",
                   "MIRAGE_GEMMA4_E4B_MODEL_PATH", "google/gemma-4-E4B-it"),
    "gemma4-12b": ("models.hf_chat", "HFChatVLM",
                   "MIRAGE_GEMMA4_12B_MODEL_PATH", "google/gemma-4-12B-it"),
    "ministral3-8b": ("models.hf_chat", "HFChatVLM",
                      "MIRAGE_MINISTRAL3_8B_MODEL_PATH", "mistralai/Ministral-3-8B-Instruct-2512-BF16"),
    "ministral3-14b": ("models.hf_chat", "HFChatVLM",
                       "MIRAGE_MINISTRAL3_14B_MODEL_PATH", "mistralai/Ministral-3-14B-Instruct-2512-BF16"),
    "aya-vision-8b": ("models.hf_chat", "HFChatVLM",
                      "MIRAGE_AYA_VISION_8B_MODEL_PATH", "CohereLabs/aya-vision-8b"),
}


def _resolve_key(model_key: Optional[str]) -> str:
    key = (model_key or os.getenv("MIRAGE_VL_MODEL", "paligemma")).lower()
    if key not in _REGISTRY:
        raise ValueError(f"unknown model key {key!r}; expected one of {sorted(_REGISTRY)}")
    return key


def build_model(model_key: Optional[str] = None, **overrides) -> Any:
    """Construct the model selected by ``model_key`` or ``$MIRAGE_VL_MODEL``.

    Args:
        model_key: Registry key (see ``_REGISTRY``); None
            reads ``MIRAGE_VL_MODEL`` (default ``"paligemma"``).
        **overrides: Constructor kwargs; ``model_name`` defaults to the
            model's ``MIRAGE_*_MODEL_PATH`` environment variable.

    Returns:
        A constructed (not yet loaded) model instance.
    """
    key = _resolve_key(model_key)
    module_name, class_name, env_var, default_name = _REGISTRY[key]
    model_cls = getattr(importlib.import_module(module_name), class_name)
    overrides.setdefault("model_name", os.getenv(env_var, default_name))
    return model_cls(**overrides)


def build_predictor(model_key: Optional[str] = None,
                    cache: Optional[SQLiteResponseCache] = None,
                    **overrides) -> Predictor:
    """Construct a cached :class:`Predictor` around :func:`build_model`.

    Args:
        model_key: See :func:`build_model`.
        cache: Cache instance; None uses the default SQLite cache.
        **overrides: Forwarded to :func:`build_model`.

    Returns:
        A Predictor that is a drop-in replacement for the model object;
        its ``model_key`` attribute records which registry entry was used.
    """
    key = _resolve_key(model_key)
    predictor = Predictor(LocalModelBackend(build_model(key, **overrides)), cache=cache)
    predictor.model_key = key
    return predictor
