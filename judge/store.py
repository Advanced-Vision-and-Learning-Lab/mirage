"""Stored judge verdicts: the frozen files in ``judgments/``."""

import csv
import gzip
from pathlib import Path
from typing import Dict, Tuple

JUDGMENTS = Path(__file__).resolve().parent.parent / "judgments"


def load_verdicts(name: str, directory: Path = JUDGMENTS) -> Dict[Tuple[str, str], str]:
    """``judgments/<name>.tsv.gz`` rows ``item, key, verdict`` as ``{(item, key): verdict}``."""
    with gzip.open(directory / f"{name}.tsv.gz", "rt", encoding="utf-8") as f:
        return {(item, key): verdict for item, key, verdict in csv.reader(f, delimiter="\t")}


def load_drop_list(directory: Path = JUDGMENTS) -> Dict[str, Dict[Tuple[str, str], str]]:
    """POPE no-questions to drop, ``{dataset: {(image_id, object): reason}}``."""
    drop: Dict[str, Dict[Tuple[str, str], str]] = {}
    with open(directory / "pope_drop_list.tsv", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            drop.setdefault(r["dataset"], {})[(r["image_id"], r["object"])] = r["reason"]
    return drop
