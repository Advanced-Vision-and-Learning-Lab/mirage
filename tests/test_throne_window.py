"""THRONE evaluator window: trimming, generation limit, and cache identity follow it.

Uses a word-level fake tokenizer, so no checkpoint or GPU is needed (torch is imported).

Run:  python3 -m pytest tests/test_throne_window.py -q
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from metrics.throne import ThroneMetric


class FakeTokenizer:
    """One token per word, so caption lengths are easy to set."""

    def __init__(self):
        self.calls = []

    def __call__(self, text, add_special_tokens=True, **kwargs):
        if isinstance(text, list):
            self.calls.append(kwargs)
            return SimpleNamespace(to=lambda device: {"input_ids": torch.ones(len(text), 1, dtype=torch.long)})
        return SimpleNamespace(input_ids=list(range(len(text.split()))))

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"w{i}" for i in ids)

    def batch_decode(self, ids, skip_special_tokens=True):
        return ["yes"] * len(ids)


def _metric(**kwargs):
    metric = ThroneMetric(object_categories=["dog"], device="cpu", **kwargs)
    metric._evaluator_tokenizer = FakeTokenizer()
    return metric


def test_default_window_trims_long_captions():
    metric = _metric()
    assert metric._max_input_tokens == 512
    assert metric._fit_caption("word " * 600) != "word " * 600
    assert metric._truncated_inputs == 1


def test_wider_window_keeps_the_whole_caption():
    caption = "word " * 600
    metric = _metric(evaluator_max_tokens=1024)
    assert metric._fit_caption(caption) == caption
    assert metric._truncated_inputs == 0


def test_window_from_environment(monkeypatch):
    monkeypatch.setenv("MIRAGE_THRONE_EVALUATOR_MAX_TOKENS", "1024")
    assert _metric()._max_input_tokens == 1024


def test_window_reaches_generation_and_cache_identity(monkeypatch, tmp_path):
    metric = _metric(evaluator_max_tokens=1024, batch_size=4)
    metric._evaluator_model = SimpleNamespace(generate=lambda **kwargs: torch.ones(1, 1))
    metric._raw_generate_batch(["q1", "q2"])
    assert metric._evaluator_tokenizer.calls[0]["max_length"] == 1024

    monkeypatch.setenv("MIRAGE_EVALUATOR_CACHE_DIR", str(tmp_path))
    keys = []
    for window in (512, 1024):
        m = _metric(evaluator_max_tokens=window)
        m.load_evaluator()
        keys.append(m._predictor.backend._identity["max_input_tokens"])
    assert keys == [512, 1024]
