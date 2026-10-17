"""Metric classes, imported on first use so ``import metrics`` does not pull in torch."""
import importlib

_MODULES = {
    "BaseMetric": "base_metric",
    "POPEMetric": "pope",
    "NOPEMetric": "nope",
    "ROPEMetric": "rope",
    "ChairMetric": "chair",
    "ThroneMetric": "throne",
    "FaithScoreMetric": "faithscore",
}

__all__ = ["BaseMetric", "POPEMetric", "ROPEMetric", "NOPEMetric", "ChairMetric", "ThroneMetric", "FaithScoreMetric"]


def __getattr__(name):
    if name not in _MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{_MODULES[name]}", __name__), name)
    globals()[name] = value
    return value
