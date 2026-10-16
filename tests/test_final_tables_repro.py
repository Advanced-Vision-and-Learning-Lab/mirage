"""The judged metrics reproduce tests/data/final_tables.json from cached predictions.

Needs the reference results tree (datasets, caches, result JSONs) and NLTK data,
so it is skipped unless MIRAGE_REPRO_ROOT points at that tree. Run it as a CPU
job, from the repository root:

    MIRAGE_REPRO_ROOT=/path/to/tree python -m pytest -q tests/test_final_tables_repro.py
"""
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
TREE = os.environ.get("MIRAGE_REPRO_ROOT")


def _leaves(node, path=()):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _leaves(v, path + (k,))
    else:
        yield path, node


@pytest.mark.skipif(not TREE, reason="set MIRAGE_REPRO_ROOT to the reference results tree")
def test_reproduces_reference_table():
    from metrics.judged import build_table

    expected = dict(_leaves(json.load(open(ROOT / "tests/data/final_tables.json"))["table"]))
    actual = dict(_leaves(build_table(Path(TREE))))
    assert actual.keys() == expected.keys()
    floats = [p for p, v in expected.items() if isinstance(v, float)]
    worst = max(abs(actual[p] - expected[p]) for p in floats)
    print(f"{len(expected)} values ({len(floats)} floats); max float difference {worst:.3g}")
    # Class-wise THRONE averages sum over a set, so the reference pins its summation
    # order only to the last bit; everything else is exact.
    assert worst <= 1e-12
    assert all(actual[p] == v for p, v in expected.items() if not isinstance(v, float))
