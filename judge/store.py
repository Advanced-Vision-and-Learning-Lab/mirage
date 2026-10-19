"""Stored judge verdicts: the frozen files in ``judgments/`` plus an optional live shard."""

import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Tuple

JUDGMENTS = Path(__file__).resolve().parent.parent / "judgments"
MANIFEST = Path(__file__).resolve().parent / "manifest.json"
KINDS = ("mentions", "rope")


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


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def judgments_dir(path: Optional[Path] = None) -> Path:
    """``path``, else ``$MIRAGE_JUDGMENTS_DIR``, else the repository's ``judgments/``."""
    return Path(path or os.environ.get("MIRAGE_JUDGMENTS_DIR") or JUDGMENTS)


def _check(path: Path, digest: str) -> None:
    if sha256(path) != digest:
        raise ValueError(f"{path} does not match its sha256 in the judge manifest")


class VerdictStore:
    """Frozen verdicts (checked against the manifest's sha256s) and new ones in a live shard.

    Frozen rows always win: :meth:`verdicts` is ``{**live, **frozen}`` and :meth:`add`
    refuses keys that are already frozen, so a new model never moves a published row.
    ``committee.json`` (written by ``repro.py --export-committee``) is one of the manifest's
    "files", like the verdicts. ``final_tables.json`` is checked only when :meth:`reference_table`
    reads it.
    """

    def __init__(self, directory: Path, live_dir: Optional[Path] = None, reference_sha: Optional[str] = None):
        self.directory = Path(directory)
        self.live_dir = Path(live_dir) if live_dir else None
        self._reference_sha = reference_sha
        self._frozen = {kind: load_verdicts(kind, self.directory) for kind in KINDS}
        self._drop = load_drop_list(self.directory)
        self._live = {kind: self._read_live(kind) for kind in KINDS}
        self._committee = json.loads((self.directory / "committee.json").read_text())

    @classmethod
    def open(cls, path: Optional[Path] = None, manifest: Path = MANIFEST, live: bool = False,
             live_dir: Optional[Path] = None) -> "VerdictStore":
        """Open :func:`judgments_dir` ``(path)`` with the live shard in ``live_dir`` (with ``live=True``,
        default ``<directory>/live``).

        Raises:
            ValueError: A file the store loads (the manifest's "files") does not match its sha256.
        """
        directory, spec = judgments_dir(path), json.loads(Path(manifest).read_text())
        for name, digest in spec["files"].items():
            _check(directory / name, digest)
        live_dir = live_dir or (directory / "live" if live else None)
        return cls(directory, live_dir, spec.get("reference", {}).get("final_tables.json"))

    def verdicts(self, kind: str) -> Dict[Tuple[str, str], str]:
        """``{(item, key): verdict}`` for ``kind`` ("mentions" or "rope")."""
        return {**self._live[kind], **self._frozen[kind]}

    def drop_list(self, ds: str) -> Dict[Tuple[str, str], str]:
        """POPE no-questions to drop on ``ds``: ``{(image_id, object): reason}``."""
        return self._drop.get(ds, {})

    def committee(self, ds: str) -> Dict[str, Dict[str, Set[str]]]:
        """Per item of ``ds``: words the frozen pool resolved ``present`` and its ``verified`` present words."""
        return {item: {k: set(v) for k, v in row.items()} for item, row in self._committee.get(ds, {}).items()}

    def covered(self, ds: str) -> Set[str]:
        """Items with frozen POPE drop decisions and a committee (the frozen images of ``ds``)."""
        return set(self._committee.get(ds, {}))

    def reference_table(self) -> dict:
        """The published table (``final_tables.json``) that ``repro.py --check`` compares with.

        Raises:
            ValueError: The file does not match its sha256 in the manifest's "reference".
        """
        path = self.directory / "final_tables.json"
        if self._reference_sha:
            _check(path, self._reference_sha)
        return json.loads(path.read_text())["table"]

    def add(self, kind: str, rows: Iterable[Tuple[str, str, str]], judge_model: str, skill_sha: str) -> None:
        """Append new ``(item, key, verdict)`` rows to the live shard.

        Raises:
            ValueError: No live shard is open, or a row's key is already frozen.
        """
        rows = list(rows)
        if self.live_dir is None:
            raise ValueError("this store has no live shard; open it with live_dir")
        frozen = [(item, key) for item, key, _ in rows if (item, key) in self._frozen[kind]]
        if frozen:
            raise ValueError(f"{len(frozen)} {kind} keys are already frozen, e.g. {frozen[:5]}")
        self.live_dir.mkdir(parents=True, exist_ok=True)
        buf = io.StringIO()
        csv.writer(buf, delimiter="\t", lineterminator="\n").writerows(
            (item, key, verdict, judge_model, skill_sha) for item, key, verdict in rows)
        with open(self.live_dir / f"{kind}.tsv", "a", encoding="utf-8", newline="") as f:
            f.write(buf.getvalue())
        self._live[kind].update(((item, key), verdict) for item, key, verdict in rows)

    def _read_live(self, kind: str) -> Dict[Tuple[str, str], str]:
        path = self.live_dir / f"{kind}.tsv" if self.live_dir else None
        if path is None or not path.exists():
            return {}
        live = {}
        with open(path, encoding="utf-8", newline="") as f:
            for n, row in enumerate(csv.reader(f, delimiter="\t"), 1):
                if len(row) < 3:
                    raise ValueError(f"{path}:{n}: expected item, key, verdict[, judge model, skill sha], got {row}")
                live[row[0], row[1]] = row[2]
        return live
