"""Grading reproduces the published table.

tests/data/records.json.gz holds every model reply the table scores (``repro.py --export-records``
on the reference tree, via tests/data/freeze.py). ``repro.py --check`` rescores them with the
frozen judgments and must give all 612 values of judgments/final_tables.json and
judgments/committee.json exactly. A change to answer parsing, mention tagging, the judge store,
a metric's ``score`` or the table assembly fails here.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import repro

RECORDS = ROOT / "tests/data/records.json.gz"


@pytest.mark.skipif(sys.version_info < (3, 12), reason="the exact THRONE sums need Python 3.12+ (see repro.py)")
def test_published_table(capsys):
    status = repro.main(["--records", str(RECORDS), "--check"])
    assert status == 0, capsys.readouterr().out
