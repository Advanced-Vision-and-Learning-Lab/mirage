"""Require documentation updates whenever project code or dependencies change."""

from __future__ import annotations

import subprocess
import sys
from pathlib import PurePosixPath


CODE_SUFFIXES = {".py", ".sh", ".slurm", ".ipynb"}
DEPENDENCY_FILES = {"requirements.txt", "pyproject.toml", "setup.py", "setup.cfg"}


def changed_files(base: str, head: str) -> list[str]:
    """Return changed paths between two fetched commits."""
    if base and set(base) == {"0"}:
        result = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", head],
            check=True,
            capture_output=True,
            text=True,
        )
    else:
        result = subprocess.run(
            ["git", "diff", "--name-only", "--diff-filter=ACMR", base, head],
            check=True,
            capture_output=True,
            text=True,
        )
    return [path for path in result.stdout.splitlines() if path]


def is_code_or_dependency(path: str) -> bool:
    candidate = PurePosixPath(path)
    return candidate.suffix.lower() in CODE_SUFFIXES or candidate.name in DEPENDENCY_FILES


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: check_docs_coverage.py BASE_SHA HEAD_SHA", file=sys.stderr)
        return 2

    base, head = sys.argv[1:]
    paths = changed_files(base, head)
    code_paths = [path for path in paths if is_code_or_dependency(path)]
    docs_changed = any(path.startswith("docs/") for path in paths)

    if code_paths and not docs_changed:
        print("Code or dependency changes require a documentation update under docs/:")
        print("\n".join(f"  {path}" for path in code_paths))
        return 1

    if code_paths:
        print(f"Documentation coverage passed for {len(code_paths)} code/dependency file(s).")
    else:
        print("No code or dependency changes require documentation coverage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
