"""Check that model keys documented for runs remain accepted by the CLI."""

import sys
from pathlib import Path

import demo


def test_cli_accepts_each_registered_model(monkeypatch):
    from predict.factory import _REGISTRY

    for model_key in _REGISTRY:
        monkeypatch.setattr(sys, "argv", ["demo.py", "--model", model_key])
        assert demo._parse_args().model == model_key


def test_registered_models_are_named_in_user_docs():
    from predict.factory import _REGISTRY

    root = Path(__file__).resolve().parents[1]
    pages = [
        (root / "docs" / "metrics-and-models.mdx").read_text(encoding="utf-8"),
        (root / "docs" / "hpc.mdx").read_text(encoding="utf-8"),
    ]
    for model_key in _REGISTRY:
        assert all(f"`{model_key}`" in page for page in pages)
