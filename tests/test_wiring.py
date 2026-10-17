"""run.py end to end on real data: dataloader -> metric -> prediction cache -> judge -> results.

tests/data/wiring (from tests/data/freeze.py) holds the first two annotated images and the first
two ROPE images of COCO and Visual Genome, their annotations, and gemma4-e4b's published cache
rows for them. Each cell replays them --cache-only with the frozen judgments: every request must
hit a published row (same prompts, generation settings, image decoding, ROPE box drawing and
cache identity as the reference runs), and the results must equal expected.json. Visual Genome
THRONE is left out: its question list comes from the full 2,000-image pool.

    python tests/test_wiring.py --freeze    # rewrite expected.json; only when an MR means to change it
"""

import gzip
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run
from predict.cache import SQLiteResponseCache
from predict.factory import REGISTRY, build_predictor

DATA = ROOT / "tests/data/wiring"
KEY = "gemma4-e4b"
EVALUATOR = ("google/flan-t5-xl", "7d6315df2c2fb742f0f5b556879d730926ca9001")  # the published THRONE evaluator
CELLS = ["coco-pope", "coco-rope", "coco-chair", "coco-throne",
         "visual_genome-pope", "visual_genome-rope", "visual_genome-chair"]
META = ("key_version", "provider", "model_class", "model_id", "revision", "quantization",
        "prompt", "image_hash", "gen_config")


class Replay:
    """The published gemma4-e4b's cache identity; it never generates, so a cache miss fails."""

    def __init__(self, model_id, revision):
        self.model_name = f"models--{model_id.replace('/', '--')}/snapshots/{revision}"
        self.cache_name, self.torch_dtype, self.config = REGISTRY[KEY].cache_name, REGISTRY[KEY].dtype, {}
        self.is_loaded = False

    def load_model(self):
        pass

    def unload_model(self):
        pass

    def generate(self, prompt, image=None, **gen):
        raise AssertionError(f"no published row for {prompt!r} {gen}")


def _evaluator_snapshot():
    """The evaluator's HF cache snapshot; only its tokenizer is read (it trims long captions)."""
    from huggingface_hub import try_to_load_from_cache

    path = try_to_load_from_cache(EVALUATOR[0], "tokenizer_config.json", revision=EVALUATOR[1])
    if not isinstance(path, str):
        raise FileNotFoundError(f"{EVALUATOR[0]} tokenizer at {EVALUATOR[1][:8]} is not in the HF cache")
    return str(Path(path).parent)


def run_cell(cell, tmp, monkeypatch):
    ds, metric = cell.split("-")
    rows = json.loads(gzip.decompress((DATA / "rows.json.gz").read_bytes()))
    for name in [k for k in os.environ if k.startswith("MIRAGE_")]:
        monkeypatch.delenv(name)
    monkeypatch.setenv("MIRAGE_EVALUATOR_CACHE_DIR", str(tmp))
    for db, part in ((tmp / "predictions.sqlite", [r for r in rows if r["model_class"] != "ThroneEvaluator"]),
                     (tmp / "evaluator.sqlite", [r for r in rows if r["model_class"] == "ThroneEvaluator"])):
        cache = SQLiteResponseCache(db)
        cache.put_many([(r["key"], r["response"], {k: r[k] for k in META}) for r in part])
        cache.close()
    vlm = next(r for r in rows if r["model_class"] == REGISTRY[KEY].cache_name)
    model = build_predictor(Replay(vlm["model_id"], vlm["revision"]), key=KEY,
                            cache=SQLiteResponseCache(tmp / "predictions.sqlite", readonly=True))
    annotations, images = (("coco/annotations/instances_val2017.json", "coco/val2017") if ds == "coco"
                           else ("visual_genome/objects.json", "visual_genome/VG_100K"))
    files = ["--annotations", str(DATA / annotations), "--images", str(DATA / images)]
    if metric == "throne":
        files += ["--throne-evaluator", _evaluator_snapshot()]
    out = tmp / "results.json"
    assert run.main(["--dataset", ds, "--metric", metric, "--model", KEY, "--samples", "2", "--cache-only",
                     "--output", str(out), *files], model=model) == 0
    results = json.loads(out.read_text())
    assert results["cache_stats"]["misses"] == 0
    if metric == "throne":  # a local path
        assert results["throne_results"].pop("evaluator_model") == files[-1]
    return results


@pytest.mark.parametrize("cell", CELLS)
def test_cell(cell, tmp_path, monkeypatch):
    if cell.endswith("throne"):
        try:
            _evaluator_snapshot()
        except (ImportError, FileNotFoundError) as e:
            pytest.skip(f"THRONE needs the FLAN-T5-XL tokenizer: {e}")
    expected = json.loads((DATA / "expected.json").read_text())[cell]
    assert run_cell(cell, tmp_path, monkeypatch) == expected


if __name__ == "__main__" and sys.argv[1:] == ["--freeze"]:
    import tempfile

    expected = {}
    with pytest.MonkeyPatch.context() as mp:
        for cell in CELLS:
            with tempfile.TemporaryDirectory() as tmp:
                expected[cell] = run_cell(cell, Path(tmp), mp)
    (DATA / "expected.json").write_text(json.dumps(expected, indent=1, sort_keys=True) + "\n")
