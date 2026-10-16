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
        "cache_stats": {},
    }
    assert '"caption_count": 1' in capsys.readouterr().out


def _write_vg(tmp_path):
    """Two VG images on disk plus one annotated image whose file is missing."""
    images_dir = tmp_path / "VG_100K"
    images_dir.mkdir()
    for image_id in (1, 2):
        Image.new("RGB", (2, 2), color="white").save(images_dir / f"{image_id}.jpg")
    annotations = tmp_path / "objects.json"
    annotations.write_text(json.dumps([
        {"image_id": 1, "objects": [
            {"names": ["dog"], "x": 0, "y": 0, "w": 1, "h": 1},
            {"names": ["the"], "x": 0, "y": 0, "w": 1, "h": 1},
            {"names": ["red"], "x": 0, "y": 0, "w": 1, "h": 1},
        ]},
        {"image_id": 2, "objects": [{"names": ["cat"], "x": 0, "y": 0, "w": 1, "h": 1}]},
        {"image_id": 3, "objects": [{"names": ["bus"], "x": 0, "y": 0, "w": 1, "h": 1}]},
    ]), encoding="utf-8")
    return annotations, images_dir


def _fake_predictor(monkeypatch, calls, loaded):
    class FakePredictor:
        model_name = "offline-test-model"
        model_key = "test"
        stats = {"hits": 0, "misses": 0, "bypassed": 0}

        def load_model(self):
            loaded.append(True)

        def unload_model(self):
            pass

        def generate(self, *, image, prompt, **kwargs):
            calls.append((prompt, kwargs))
            return "a dog"

    monkeypatch.setattr(predictor_factory, "build_predictor", lambda model_key=None: FakePredictor())


def _fake_metric_module(monkeypatch, name, cls_name, seen):
    metrics_pkg = types.ModuleType("metrics")
    metrics_pkg.__path__ = []
    module = types.ModuleType(f"metrics.{name}")

    class FakeMetric:
        def __init__(self, **kwargs):
            seen["init"] = kwargs

        def evaluate(self, captions, ground_truth):
            seen["captions"], seen["ground_truth"] = captions, ground_truth

        def format_results(self):
            return {"n": len(seen["captions"])}

    setattr(module, cls_name, FakeMetric)
    monkeypatch.setitem(sys.modules, "metrics", metrics_pkg)
    monkeypatch.setitem(sys.modules, f"metrics.{name}", module)


def test_vg_chair_uses_vg_prompt_filter_and_512_tokens(tmp_path, monkeypatch):
    """VG captions use the VG prompt and 512 tokens; CHAIR drops non-object strings."""
    annotations, images_dir = _write_vg(tmp_path)
    calls, loaded, seen = [], [], {}
    _fake_predictor(monkeypatch, calls, loaded)
    _fake_metric_module(monkeypatch, "chair", "ChairMetric", seen)
    stopwords = types.SimpleNamespace(words=lambda lang: ["the"])
    monkeypatch.setitem(sys.modules, "nltk.corpus", types.SimpleNamespace(stopwords=stopwords))

    output_path = tmp_path / "results.json"
    monkeypatch.setattr(sys, "argv", [
        "demo.py", "--dataset", "visual_genome", "--metric", "chair", "--model", "paligemma",
        "--samples", "5", "--annotations", str(annotations), "--images", str(images_dir),
        "--output", str(output_path),
    ])
    assert demo.main() == 0
    assert loaded
    assert [prompt for prompt, _ in calls] == [demo.VG_CAPTION_PROMPT] * 2
    assert all(kwargs["max_new_tokens"] == 512 for _, kwargs in calls)
    # The VG vocabulary comes from all of objects.json (as in the reference runs).
    assert sorted(seen["init"]["object_synsets"]) == ["bus", "cat", "dog"]
    results = json.loads(output_path.read_text(encoding="utf-8"))
    assert results["cache_stats"]["misses"] == 0


def test_throne_flags_and_cache_only(tmp_path, monkeypatch):
    """THRONE flags reach the metric, and --cache-only never requests a model load."""
    annotations, images_dir = _write_vg(tmp_path)
    calls, loaded, seen = [], [], {}
    _fake_predictor(monkeypatch, calls, loaded)
    _fake_metric_module(monkeypatch, "throne", "ThroneMetric", seen)

    monkeypatch.setattr(sys, "argv", [
        "demo.py", "--dataset", "visual_genome", "--metric", "throne", "--model", "paligemma",
        "--samples", "1", "--annotations", str(annotations), "--images", str(images_dir),
        "--throne-evaluator", "/models/flan-t5-xl", "--throne-distractor-ratio", "38",
        "--throne-evaluator-max-tokens", "1024", "--cache-only",
    ])
    assert demo.main() == 0
    assert not loaded
    assert len(calls) == 1
    assert seen["init"]["evaluator_model_name"] == "/models/flan-t5-xl"
    assert seen["init"]["distractor_ratio"] == 38.0
    assert seen["init"]["evaluator_max_tokens"] == 1024
    # VG THRONE categories come from the loaded pool only ('bus' is not on disk).
    assert seen["init"]["object_categories"] == ["cat", "dog", "red", "the"]
    assert seen["ground_truth"] == [["dog", "the", "red"]]


def test_default_pool_sizes(monkeypatch):
    """The candidate pool is fixed (2000; 3000 for COCO ROPE), not a multiple of --samples."""
    pools = []

    class FakeLoader:
        def __init__(self, **kwargs):
            pass

        def load_data(self, max_samples):
            pools.append(max_samples)

        def get_category_names(self):
            return []

    import dataloader.coco
    monkeypatch.setattr(dataloader.coco, "COCODataLoader", FakeLoader)
    for metric in ("pope", "rope"):
        args = types.SimpleNamespace(dataset="coco", metric=metric, samples=10, pool_size=None,
                                     annotations=None, images=None, split="val2017")
        demo._load_dataset(args)
    assert pools == [2000, 3000]


def test_cache_only_miss_raises(tmp_path):
    """Without load_model(), a prediction-cache miss raises instead of running the model."""
    from predict.cache import SQLiteResponseCache
    from predict.predictor import Predictor

    class StubModel:
        model_name = "stub/model"
        config = {"max_new_tokens": 10, "do_sample": False}
        is_loaded = False

        def load_model(self):
            raise AssertionError("cache-only runs must not load the model")

    predictor = Predictor(StubModel(), cache=SQLiteResponseCache(tmp_path / "predictions.sqlite"))
    try:
        predictor.generate(prompt="Is there a dog?", image=None, max_new_tokens=10)
    except RuntimeError as error:
        assert "not loaded" in str(error)
    else:
        raise AssertionError("a cache miss should raise")
