"""Judged metrics on hand-built records: one test per scoring rule.

No datasets, caches or NLTK data are needed (mentions are given pre-tagged).

Run:  python3 -m pytest tests/test_judged_metrics.py -q
"""
import sys
import types
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# metrics/__init__ imports torch; load metrics.judged without it.
if "metrics" not in sys.modules:
    pkg = types.ModuleType("metrics")
    pkg.__path__ = [str(ROOT / "metrics")]
    sys.modules["metrics"] = pkg

from metrics.judged import (MissingVerdicts, fbeta, load_drop_list, load_verdicts, mod_chair, mod_pope,
                            mod_rope, throne_classwise)


def test_mod_pope_drops_listed_no_questions_only():
    records = [("1", "dog", True, True), ("1", "cat", False, True), ("1", "car", False, False),
               ("2", "cup", True, False)]
    out = mod_pope(records, {("1", "cat"): "present", ("2", "cup"): "unsure"})
    assert out["raw"] == 0.5 and out["n_raw"] == 4
    assert out["n_head"] == 3  # the listed yes-question ("2", "cup") stays
    assert out["head"] == pytest.approx((1 / 2 + 1 / 1) / 2)  # balanced: yes recall 1/2, no recall 1/1
    assert out["yes_ratio"] == pytest.approx(1 / 3)


def _rope_verdicts():
    return {("coco_1", "1|dog"): "correct", ("coco_1", "2|car"): "related", ("coco_1", "3|cat"): "wrong",
            ("coco_1", "4|cup"): "unsure"}


def test_mod_rope_excludes_unsure_and_reports_related():
    records = [("coco_1", 1, "dog", True), ("coco_1", 2, "car", False), ("coco_1", 3, "cat", False),
               ("coco_1", 4, "cup", False)]
    out = mod_rope(records, _rope_verdicts())
    assert out["raw"] == 0.25
    assert out["head"] == pytest.approx(1 / 3) and out["n_head"] == 3
    assert out["related"] == pytest.approx(1 / 3)
    assert out["correct_incl_unsure_as_wrong"] == pytest.approx(1 / 4)


def test_missing_verdict_is_an_error():
    with pytest.raises(MissingVerdicts) as err:
        mod_rope([("coco_1", 5, "bus", False)], _rope_verdicts())
    assert err.value.items == [("coco_1", "5|bus")]


def _mention(word, hall=True, noun=True, rel=None):
    return {"word": word, "hall": hall, "noun": noun, "rel": rel}


def test_mod_chair_resolution_dedup_and_coverage():
    long = "a dog sits next to a red car on the street"
    images = [{"item": "vg_1", "models": {
        "a": {"caption": long, "mentions": [_mention("dog", hall=False), _mention("car"), _mention("car"),
                                             _mention("red", noun=False), _mention("pup", rel="synonym")]},
        "b": {"caption": "car", "mentions": [_mention("car")]},  # under 5 words: mentions nothing
    }}]
    verdicts = {("vg_1", "a|car"): "absent"}
    out = mod_chair(images, verdicts)
    a, b = out["a"], out["b"]
    assert a["raw"] == pytest.approx(4 / 5)  # raw CHAIR-I counts every non-annotated mention
    assert a["n_head"] == 3 and a["mod_chair_i"] == pytest.approx(1 / 3)  # dog, pup, car once; "red" not an object
    assert a["coverage"] == 1.0 and b["coverage"] == 0.0 and b["n_head"] == 0
    assert a["head"] == pytest.approx(fbeta(2 / 3, 1.0)) and a["mod_chair_s"] == 1.0


def test_mod_chair_reference_caption_objects_are_present():
    images = [{"item": "coco_1", "models": {"a": {"caption": "a man holds a red umbrella outside",
                                                   "mentions": [_mention("umbrella")]}}}]
    out = mod_chair(images, {}, refs={"coco_1": {"umbrella"}})
    assert out["a"]["mod_chair_i"] == 0.0
    with pytest.raises(MissingVerdicts):
        mod_chair(images, {})


def _detail(caption, pred, gt):
    return {"caption": caption, "predicted_objects": pred, "gt_objects": gt, "disqualified_objects": []}


def test_throne_minimum_response_completed_labels_and_support():
    details = [_detail("a dog and a cat on a sofa", ["dog", "sofa"], ["dog", "cat"]),
               _detail("dog", ["dog"], ["dog"])]  # under 5 words: predicts nothing
    support = Counter({"dog": 2, "cat": 1, "sofa": 0})
    out = throne_classwise(details, support)
    assert out["n_head"] == 3 and out["P"] == pytest.approx((1 + 0) / 2)  # dog 1/1, sofa 0/1
    assert out["R"] == pytest.approx((1 / 2 + 0) / 2)  # dog 1/2, cat 0/1
    completed = throne_classwise(details, support, is_present=lambda i, c: c == "sofa")
    assert completed["P"] == 1.0  # the judge saw the sofa: not a false positive
    assert throne_classwise(details, support, min_support=2)["n_head"] == 1  # dog only
    assert throne_classwise(details, support, min_words=None)["R"] == pytest.approx((1 + 0) / 2)


def test_committed_judgments_load():
    drop = load_drop_list()
    assert sum(len(v) for v in drop.values()) == 569
    assert len(load_verdicts("mentions")) == 43499 and len(load_verdicts("rope")) == 28897
