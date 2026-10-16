"""Compressed SQLite store for model predictions.

The database is a queryable log of every cached generation: prompts,
generation configs, and model identity are stored in plaintext; only the
response text is zlib-compressed. The default location is ``cache/`` at
the repository root so the same file is usable locally and on HPC;
override with the ``MIRAGE_CACHE_DIR`` environment variable or an
explicit ``db_path``.
"""

import logging
import os
import sqlite3
import zlib
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    key           TEXT PRIMARY KEY,
    key_version   TEXT NOT NULL,
    provider      TEXT NOT NULL,
    model_class   TEXT,
    model_id      TEXT NOT NULL,
    revision      TEXT,
    quantization  TEXT,
    prompt        TEXT NOT NULL,
    image_hash    TEXT,
    gen_config    TEXT NOT NULL,
    response      BLOB NOT NULL,
    response_len  INTEGER NOT NULL,
    created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
CREATE TABLE IF NOT EXISTS cache_meta (k TEXT PRIMARY KEY, v TEXT);
"""

# Secondary indexes exist only for offline analysis (browse by model, find every
# question about one image); lookups use the primary key. On a multi-GB store
# they dominate insert cost: 12,990 rows took 83 s into a 3.7 GB evaluator DB
# on Lustre with them and 1.5 s without, so write-heavy stores opt out.
_SECONDARY_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_pred_model ON predictions(model_id, created_at);
CREATE INDEX IF NOT EXISTS idx_pred_image ON predictions(image_hash);
"""
_DROP_SECONDARY_INDEXES = """
DROP INDEX IF EXISTS idx_pred_model;
DROP INDEX IF EXISTS idx_pred_image;
"""

_META_COLUMNS = (
    "key_version", "provider", "model_class", "model_id",
    "revision", "quantization", "prompt", "image_hash", "gen_config",
)
_INSERT_SQL = "INSERT OR IGNORE INTO predictions ({}) VALUES ({})".format(
    ", ".join(("key",) + _META_COLUMNS + ("response", "response_len")),
    ", ".join("?" * (len(_META_COLUMNS) + 3)),
)


def default_cache_path() -> Path:
    """Resolve the default database path.

    Returns:
        ``$MIRAGE_CACHE_DIR/predictions.sqlite`` when the variable is set,
        else ``<repo_root>/cache/predictions.sqlite``. The repo-relative
        default assumes a source checkout; for pip-installed usage set
        ``MIRAGE_CACHE_DIR``.
    """
    override = os.environ.get("MIRAGE_CACHE_DIR")
    base = Path(override) if override else Path(__file__).resolve().parent.parent / "cache"
    return base / "predictions.sqlite"


class SQLiteResponseCache:
    """Key-value store of compressed model responses backed by SQLite.

    Read/write errors degrade to misses/dropped writes rather than
    crashing a long-running evaluation.

    Args:
        db_path: Database file location. None uses :func:`default_cache_path`.
        readonly: Open the database read-only; a missing file then acts as
            an always-miss cache, and :meth:`put` raises.
        secondary_indexes: Keep the analysis indexes (by model, by image).
            Pass False for write-heavy stores such as the THRONE evaluator
            cache; existing indexes are dropped on open.
    """

    def __init__(self, db_path: Optional[Union[str, Path]] = None, readonly: bool = False,
                 secondary_indexes: bool = True):
        self.db_path = Path(db_path) if db_path else default_cache_path()
        self.readonly = readonly
        if readonly:
            if not self.db_path.exists():
                logger.warning("read-only cache %s does not exist; every lookup will miss",
                               self.db_path)
                self._conn = None
                return
            self._conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
            self._conn.execute("PRAGMA busy_timeout=30000")
            return
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        # Rollback journal, not WAL: HPC nodes share one filesystem and
        # WAL's shared-memory index is unsafe across hosts.
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        with self._conn:
            self._conn.executescript(_SCHEMA)
            self._conn.executescript(_SECONDARY_INDEXES if secondary_indexes else _DROP_SECONDARY_INDEXES)
            self._conn.execute(
                "INSERT OR IGNORE INTO cache_meta (k, v) VALUES ('schema_version', '1')"
            )

    def get(self, key: str) -> Optional[str]:
        """Return the cached response for ``key``, or None on a miss."""
        if self._conn is None:
            return None
        try:
            row = self._conn.execute(
                "SELECT response FROM predictions WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                return None
            return zlib.decompress(row[0]).decode("utf-8")
        except (sqlite3.Error, zlib.error, UnicodeDecodeError) as exc:
            logger.warning("cache read failed, treating as miss: %s", exc)
            return None

    def get_many(self, keys: Iterable[str]) -> Dict[str, str]:
        """Return ``{key: response}`` for every cached key in ``keys``."""
        if self._conn is None:
            return {}
        keys = list(keys)
        found: Dict[str, str] = {}
        try:
            for i in range(0, len(keys), 500):
                chunk = keys[i:i + 500]
                marks = ",".join("?" * len(chunk))
                rows = self._conn.execute(
                    f"SELECT key, response FROM predictions WHERE key IN ({marks})", chunk
                ).fetchall()
                found.update({k: zlib.decompress(blob).decode("utf-8") for k, blob in rows})
        except (sqlite3.Error, zlib.error, UnicodeDecodeError) as exc:
            logger.warning("cache read failed, treating as miss: %s", exc)
            return {}
        return found

    def rows(self) -> Iterator[Tuple[str, Optional[str], str]]:
        """Yield ``(prompt, image_hash, response)`` for every stored row (offline rescoring)."""
        if self._conn is None:
            return
        for prompt, image_hash, blob in self._conn.execute("SELECT prompt, image_hash, response FROM predictions"):
            yield prompt, image_hash, zlib.decompress(blob).decode("utf-8")

    def put(self, key: str, response: str, *, meta: Dict[str, Any]) -> None:
        """Insert one response; existing rows for the same key are kept as-is.

        Args:
            key: Cache key from :func:`predict.identity.cache_key`.
            response: Generated text to store.
            meta: Values for the metadata columns (see ``_META_COLUMNS``).

        Raises:
            RuntimeError: When the cache was opened read-only.
        """
        self.put_many([(key, response, meta)])

    def put_many(self, rows: List[Tuple[str, str, Dict[str, Any]]]) -> None:
        """Insert many ``(key, response, meta)`` rows in one transaction."""
        if self.readonly:
            raise RuntimeError("cache is read-only")
        if not rows:
            return
        values = []
        for key, response, meta in rows:
            raw = response.encode("utf-8")
            values.append([key] + [meta.get(c) for c in _META_COLUMNS]
                          + [zlib.compress(raw, 6), len(raw)])
        try:
            with self._conn:
                self._conn.executemany(_INSERT_SQL, values)
        except sqlite3.Error as exc:
            logger.warning("cache write dropped: %s", exc)

    def close(self) -> None:
        """Close the underlying connection."""
        if self._conn is not None:
            self._conn.close()
