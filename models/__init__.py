"""Model wrappers, one file per family; imported on first use so ``import models`` needs no torch.

The registry in ``predict.factory`` names each model as ``module:Class``.
"""
import importlib

_MODULES = {
    "BaseModel": "base",
    "PaligemmaModel": "paligemma",
    "Qwen2VLModel": "qwen2vl",
    "HFChatVLM": "hf_chat",
}

__all__ = list(_MODULES)


def __getattr__(name):
    if name not in _MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{_MODULES[name]}", __name__), name)
    globals()[name] = value
    return value
