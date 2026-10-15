"""ROPE single-box answer extraction: formatting is normalized, matching stays exact.

Every input below is a real cached model answer. Runs offline (no GPU, no torch).

Run:  python3 -m pytest tests/test_rope_parse.py -q
"""
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# metrics/__init__ imports torch; load the torch-free metrics.rope without it.
if "metrics" not in sys.modules:
    pkg = types.ModuleType("metrics")
    pkg.__path__ = [str(ROOT / "metrics")]
    sys.modules["metrics"] = pkg

from metrics.rope import ROPEMetric, _rope_match_strict


@pytest.fixture(scope="module")
def parse():
    return ROPEMetric(num_objects=5, class_list=["person"]).parse_single_response


@pytest.mark.parametrize("response, expected", [
    # Gemma-4-E4B: markdown emphasis and parenthetical asides.
    ("The object inside box 4 is a **zebra**.", "zebra"),
    ("The object inside box 3 appears to be a **person** (a man).", "person"),
    ("The object inside box 3 appears to be a **person (a man)**.", "person"),
    ("The object inside box 5 is a **vehicle** (specifically, a light-colored van or small truck).", "vehicle"),
    ("The object inside **box 2** is a **boat** (specifically, a tour or excursion boat).", "boat"),
    ("The object inside box 1 appears to be a **dining table** with **chairs** around it.", "dining table"),
    ("Based on the image, the object inside **box 3** appears to be **durian**.", "durian"),
    ("Based on the image, **box 1** is pointing to the **swimming pool**.", "swimming pool"),
    # Ministral 3: "appears to be" before a later "It is a ..." sentence.
    ("The object inside **box 1** appears to be a **rolling pin** (also known as a dough roller). "
     "It is a cylindrical tool typically", "rolling pin"),
    ("The object inside **box 1** appears to be a **person**.", "person"),
    # Gemma-4-12B and Qwen2-VL: "is the".
    ("The object inside box 2 is the sky.", "sky"),
    ("The object inside box 5 is the water.", "water"),
    # Qwen2-VL-7B: "appears to be" without markdown.
    ("The object inside box 4 appears to be a book.", "book"),
    # Markdown list, header, code, and quoted answers.
    ("- **cup**", "cup"),
    ("### person", "person"),
    ("`laptop`", "laptop"),
    ('The object inside box 1 is a "bus".', "bus"),
])
def test_formatting_is_normalized(parse, response, expected):
    assert parse(response) == expected


@pytest.mark.parametrize("response, expected", [
    # Answers that already parsed cleanly parse identically.
    ("flour", "flour"),
    ("fireplace", "fireplace"),
    ("The object inside box 2 is a soccer ball.", "soccer ball"),
    ("The object inside box 1 is a wooden cabinet.", "wooden cabinet"),
    ("The object inside box 2 is a toaster oven. It is located on the countertop", "toaster oven"),
    ("The object inside box 1 is a person wearing a red shirt.", "person wearing a red shirt"),
    ("The object inside box 3 is a piece of lettuce.", "piece"),
    # Qwen2-VL-7B hedge: the echoed question is not an answer.
    ("I am not sure what is the object inside box 1. It can be a suitcase, a box, a bag",
     "i am not sure what is the object inside box 1. it can be a suitcase, a box, a bag"),
])
def test_clean_answers_unchanged(parse, response, expected):
    assert parse(response) == expected


@pytest.mark.parametrize("response, truth", [
    # Normalization is about format only: no synonyms, no picking one of
    # several alternatives, no dropping modifiers.
    ("The object inside box 5 is a **tent** or **canopy**.", "tent"),
    ("The object inside box 2 is a **black cow**.", "cow"),
    ("The object inside box 1 is a piece of **meat** (likely ham).", "meat"),
    ("Based on the image, **box 1** contains **bread** or **baked goods**.", "bread"),
])
def test_strict_stays_exact(parse, response, truth):
    assert not _rope_match_strict(parse(response), truth)
