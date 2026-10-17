"""Model registry and the factory helpers drivers use to build models.

Model classes are imported lazily inside :func:`build_model` so this
module (and the whole ``predict`` package) stays importable without torch.
"""

import importlib
import os
from types import MappingProxyType
from typing import Any, Mapping, NamedTuple, Optional, Tuple, Union

from predict.backends import LocalModelBackend
from predict.cache import SQLiteResponseCache
from predict.identity import parse_model_name
from predict.predictor import Predictor


class Entry(NamedTuple):
    target: str                      # "module:Class"
    env: Union[str, Tuple[str, ...]]  # model path variable(s), first set one wins
    default: str                     # HF id used when no variable is set
    cache_name: str                  # mirror of the class's cache_name, so identity needs no torch
    dtype: str                       # mirror of str(model.torch_dtype)


_BF16 = "torch.bfloat16"
_QWEN = "MIRAGE_QWEN_MODEL_PATH"

REGISTRY: Mapping[str, Entry] = MappingProxyType({
    "paligemma": Entry("models.paligemma:PaligemmaModel", "MIRAGE_PALIGEMMA_MODEL_PATH",
                       "google/paligemma-3b-mix-224", "PaligemmaModel", _BF16),
    "qwen2vl": Entry("models.qwen2vl:Qwen2VLModel", _QWEN, "Qwen/Qwen2-VL-2B-Instruct", "Qwen2VLModel", _BF16),
    "qwen2vl-2b": Entry("models.qwen2vl:Qwen2VLModel", ("MIRAGE_QWEN2VL_2B_MODEL_PATH", _QWEN),
                        "Qwen/Qwen2-VL-2B-Instruct", "Qwen2VLModel", _BF16),
    "qwen2vl-7b": Entry("models.qwen2vl:Qwen2VLModel", ("MIRAGE_QWEN2VL_7B_MODEL_PATH", _QWEN),
                        "Qwen/Qwen2-VL-7B-Instruct", "Qwen2VLModel", _BF16),
    "gemma4-e4b": Entry("models.gemma4:Gemma4", "MIRAGE_GEMMA4_E4B_MODEL_PATH",
                        "google/gemma-4-E4B-it", "HFChatVLM", _BF16),
    "gemma4-12b": Entry("models.gemma4:Gemma4", "MIRAGE_GEMMA4_12B_MODEL_PATH",
                        "google/gemma-4-12B-it", "HFChatVLM", _BF16),
    "ministral3-8b": Entry("models.ministral3:Ministral3", "MIRAGE_MINISTRAL3_8B_MODEL_PATH",
                           "mistralai/Ministral-3-8B-Instruct-2512-BF16", "HFChatVLM", _BF16),
    "ministral3-14b": Entry("models.ministral3:Ministral3", "MIRAGE_MINISTRAL3_14B_MODEL_PATH",
                            "mistralai/Ministral-3-14B-Instruct-2512-BF16", "HFChatVLM", _BF16),
    "aya-vision-8b": Entry("models.aya_vision:AyaVision", "MIRAGE_AYA_VISION_8B_MODEL_PATH",
                           "CohereLabs/aya-vision-8b", "HFChatVLM", _BF16),
    "muse-glimmer-30b": Entry("models.muse_glimmer:MuseGlimmerVLM", "MIRAGE_MUSE_GLIMMER_30B_MODEL_PATH",
                              "meta-models/Muse-Glimmer-30B", "MuseGlimmerVLM", _BF16),
    # Thinking-on variants for the ablation (same checkpoints).
    "gemma4-e4b-think": Entry("models.gemma4:Gemma4Thinking", "MIRAGE_GEMMA4_E4B_MODEL_PATH",
                              "google/gemma-4-E4B-it", "HFChatThinkingVLM", _BF16),
    "gemma4-12b-think": Entry("models.gemma4:Gemma4Thinking", "MIRAGE_GEMMA4_12B_MODEL_PATH",
                              "google/gemma-4-12B-it", "HFChatThinkingVLM", _BF16),
})


def _resolve_key(model_key: Optional[str]) -> str:
    key = (model_key or os.getenv("MIRAGE_VL_MODEL", "paligemma")).lower()
    if key not in REGISTRY:
        raise ValueError(f"unknown model key {key!r}; expected one of {sorted(REGISTRY)}")
    return key


def _model_name(entry: Entry, env: Mapping[str, str]) -> str:
    names = (entry.env,) if isinstance(entry.env, str) else entry.env
    for i, var in enumerate(names):
        if var in env:
            # A shared fallback variable (e.g. MIRAGE_QWEN_MODEL_PATH for both Qwen sizes) must
            # point at this row's checkpoint; a dedicated variable may name any local dir.
            if i and parse_model_name(env[var])[0].rsplit("/", 1)[-1] != entry.default.rsplit("/", 1)[-1]:
                raise ValueError(f"${var}={env[var]!r} is not {entry.default}; set ${names[0]} instead")
            return env[var]
    return entry.default


def identity_for(key: str, env: Mapping[str, str] = os.environ) -> dict:
    """Cache identity of a registry model as :func:`build_model` would build it, without torch.

    Args:
        key: Registry key.
        env: Environment holding the ``MIRAGE_*_MODEL_PATH`` variables.

    Returns:
        The record ``LocalModelBackend.identity()`` returns for that model.
    """
    entry = REGISTRY[key]
    model_id, revision = parse_model_name(_model_name(entry, env))
    return {"provider": "local", "class_name": entry.cache_name, "model_id": model_id,
            "revision": revision, "quantization": "none", "dtype": entry.dtype}


def build_model(model_key: Optional[str] = None, **overrides) -> Any:
    """Construct the model selected by ``model_key`` or ``$MIRAGE_VL_MODEL``.

    Args:
        model_key: Registry key (see ``REGISTRY``); None
            reads ``MIRAGE_VL_MODEL`` (default ``"paligemma"``).
        **overrides: Constructor kwargs; ``model_name`` defaults to the
            model's ``MIRAGE_*_MODEL_PATH`` environment variable.

    Returns:
        A constructed (not yet loaded) model instance.
    """
    entry = REGISTRY[_resolve_key(model_key)]
    module_name, class_name = entry.target.split(":")
    model_cls = getattr(importlib.import_module(module_name), class_name)
    # The registry mirror must match the class pin, or identity_for and the cache disagree.
    if getattr(model_cls, "cache_name", None) != entry.cache_name:
        raise RuntimeError(f"{entry.target}.cache_name != registry cache_name {entry.cache_name!r}")
    overrides.setdefault("model_name", _model_name(entry, os.environ))
    model = model_cls(**overrides)
    if "torch_dtype" not in overrides and str(getattr(model, "torch_dtype", None)) != entry.dtype:
        raise RuntimeError(f"{entry.target} dtype {getattr(model, 'torch_dtype', None)} != registry dtype {entry.dtype}")
    return model


def build_predictor(model: Any = None, key: Optional[str] = None,
                    cache: Optional[SQLiteResponseCache] = None,
                    cache_only: bool = False, **overrides) -> Predictor:
    """Construct a cached :class:`Predictor` around a registry model or a model instance.

    Args:
        model: Registry key (see :func:`build_model`), None for
            ``key``/``$MIRAGE_VL_MODEL``, or a constructed model instance.
        key: Registry key when ``model`` is None. Required with an
            instance: it names the model in results and verdicts.
        cache: Cache instance; None uses the default SQLite cache.
        cache_only: Open the default cache read-only (ignored when
            ``cache`` is given).
        **overrides: Forwarded to :func:`build_model` (registry models only).

    Returns:
        A Predictor that is a drop-in replacement for the model object;
        its ``model_key`` attribute records the key.
    """
    if model is None or isinstance(model, str):
        if model is not None and key is not None and key.lower() != model.lower():
            raise ValueError(f"model {model!r} and key {key!r} disagree")
        key = _resolve_key(model or key)
        model = build_model(key, **overrides)
    elif key is None or overrides:
        raise ValueError("a model instance needs key= and takes no constructor overrides")
    if cache is None and cache_only:
        cache = SQLiteResponseCache(readonly=True)
    predictor = Predictor(LocalModelBackend(model), cache=cache)
    predictor.model_key = key
    return predictor
