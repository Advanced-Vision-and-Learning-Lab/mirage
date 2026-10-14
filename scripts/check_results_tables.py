"""Check that displayed model scores match the canonical results CSV."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "docs" / "model-results.csv"
DISPLAY_PATHS = (ROOT / "docs" / "index.mdx", ROOT / "README.md")
METRICS = (
    "pope_accuracy",
    "rope_strict_accuracy",
    "chair_instance_hallucination_rate",
    "throne_f1",
)


def expected_rows() -> set[tuple[str, ...]]:
    with CSV_PATH.open(newline="", encoding="utf-8") as source:
        return {
            (
                row["model"],
                *(f"{float(row[metric]) * 100:.1f}%" for metric in METRICS),
            )
            for row in csv.DictReader(source)
        }


def displayed_rows(path: Path) -> set[tuple[str, ...]]:
    rows = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = tuple(cell.strip() for cell in line.strip("|").split("|"))
        if len(cells) == 5 and all(cell.endswith("%") for cell in cells[1:]):
            rows.add(cells)
        elif len(cells) == 6 and all(cell.endswith("%") for cell in cells[2:]):
            rows.add((cells[1], *cells[2:]))
    return rows


def main() -> int:
    expected = expected_rows()
    errors = []
    for path in DISPLAY_PATHS:
        actual = displayed_rows(path)
        missing = expected - actual
        extra = actual - expected
        if missing or extra:
            details = []
            if missing:
                details.append("missing or stale result rows: " + "; ".join(" | ".join(row) for row in sorted(missing)))
            if extra:
                details.append("rows not present in the CSV: " + "; ".join(" | ".join(row) for row in sorted(extra)))
            errors.append(
                f"{path.relative_to(ROOT)} " + "; ".join(details)
            )
    if errors:
        print("\n".join(errors))
        return 1
    print("README and homepage result tables match docs/model-results.csv.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
