"""Offline smoke test for a one-image run through the demo CLI."""

import json
import sys
import types
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import demo
import predict.factory as predictor_factory


def test_demo_runs_one_coco_image(tmp_path, monkeypatch, capsys):
    """Load one local image, call the model once, and write metric output."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.new("RGB", (2, 2), color="white").save(images_dir / "000000000001.jpg")

    annotations = tmp_path / "instances.json"
    annotations.write_text(json.dumps({
        "images": [{"id": 1, "file_name": "000000000001.jpg", "width": 2, "height": 2}],
        "annotations": [{
            "id": 1, "image_id": 1, "category_id": 1,
            "bbox": [0, 0, 1, 1], "area": 1, "iscrowd": 0,
        }],
        "categories": [{"id": 1, "name": "dog", "supercategory": "animal"}],
    }), encoding="utf-8")

    calls = []

    class FakeModel:
        model_name = "offline-test-model"
        model_key = "test"

        def load_model(self):
            pass

        def unload_model(self):
            pass

        def generate(self, *, image, prompt, **kwargs):
            assert image.size == (2, 2)
            calls.append(prompt)
            return "yes"

    monkeypatch.setattr(predictor_factory, "build_predictor", lambda model_key=None: FakeModel())

    # Bypass optional NLP/model imports while exercising demo's real data and CLI flow.
    metrics_pkg = types.ModuleType("metrics")
    metrics_pkg.__path__ = []
    fake_chair = types.ModuleType("metrics.chair")

    class FakeChairMetric:
        def __init__(self, **kwargs):
            self.captions = []

        def evaluate(self, captions, ground_truth):
            self.captions = captions
            assert ground_truth == [["dog"]]

        def format_results(self):
            return {"caption_count": len(self.captions), "caption": self.captions[0]}

    fake_chair.ChairMetric = FakeChairMetric
    monkeypatch.setitem(sys.modules, "metrics", metrics_pkg)
    monkeypatch.setitem(sys.modules, "metrics.chair", fake_chair)

    output_path = tmp_path / "results.json"
    monkeypatch.setattr(sys, "argv", [
        "demo.py", "--dataset", "coco", "--metric", "chair", "--model", "paligemma",
        "--samples", "1", "--annotations", str(annotations), "--images", str(images_dir),
        "--output", str(output_path),
    ])

    assert demo.main() == 0
    assert len(calls) == 1
    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "caption_count": 1,
        "caption": "yes",
    }
    assert '"caption_count": 1' in capsys.readouterr().out
