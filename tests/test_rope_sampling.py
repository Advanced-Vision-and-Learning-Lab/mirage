"""Acceptance tests for ROPE per-image box sampling in the COCO/VG dataloaders.

Runs fully offline on synthetic annotations (no GPU, no downloads, no torch).

Run:  python3 -m pytest tests/test_rope_sampling.py -q
"""
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# dataloader/__init__ pulls in pandas/torch; load the two loaders without it.
if "dataloader" not in sys.modules:
    pkg = types.ModuleType("dataloader")
    pkg.__path__ = [str(ROOT / "dataloader")]
    sys.modules["dataloader"] = pkg

from dataloader.coco import COCODataLoader
from dataloader.visual_genome import VisualGenomeDataLoader

NUM_IMAGES = 4
POOL = 10


def _coco_loader():
    loader = COCODataLoader(data_path="unused.json", images_path="unused")
    loader.images = [{"id": i} for i in range(1, NUM_IMAGES + 1)]
    loader.categories = {c: {"name": f"cat{c}"} for c in range(POOL)}
    loader.image_id_to_annotations = {
        img["id"]: [{"bbox": [c, c, 1, 1], "category_id": c} for c in range(POOL)]
        for img in loader.images
    }
    loader._loaded = True
    return loader


def _vg_loader():
    loader = VisualGenomeDataLoader(data_path="unused.json", images_path="unused")
    loader.images = [{"id": i} for i in range(1, NUM_IMAGES + 1)]
    loader.image_id_to_objects = {
        img["id"]: [{"x": c, "y": c, "w": 1, "h": 1, "names": [f"obj{c}"]} for c in range(POOL)]
        for img in loader.images
    }
    loader._loaded = True
    return loader


LOADERS = [pytest.param(_coco_loader, id="coco"), pytest.param(_vg_loader, id="visual_genome")]


def _selection(build, **kwargs):
    """Map image_id -> ordered box names drawn by extract_bboxes_for_rope."""
    bboxes = build().extract_bboxes_for_rope(num_objects=5, **kwargs)
    return {image_id: tuple(b["name"] for b in boxes) for image_id, boxes in bboxes.items()}


@pytest.mark.parametrize("build", LOADERS)
def test_same_seed_repeats_selection(build):
    assert _selection(build) == _selection(build)


@pytest.mark.parametrize("build", LOADERS)
def test_different_seed_changes_selection(build):
    assert _selection(build) != _selection(build, seed=7)


@pytest.mark.parametrize("build", LOADERS)
def test_images_sample_independently(build):
    selection = _selection(build)
    assert len(selection) == NUM_IMAGES
    assert len(set(selection.values())) == NUM_IMAGES


@pytest.mark.parametrize("build", LOADERS)
def test_selection_is_stable_under_batch_changes(build):
    full = _selection(build)
    partial = _selection(build, sample_indices=[2])
    assert partial and partial == {k: v for k, v in full.items() if k in partial}
