"""Order-independent verdict cache for the (dormant) LLM judge tier.

Only the judge path uses this — the default WordNet regrade is deterministic and
needs no cache. Kept so a future ``--build-cache`` run (issue #27) can judge the
residual once on GPU, commit the verdicts, and then re-grade fully offline.

The cache invalidates itself if the prompt version or judge model changes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


class PairCache:
    def __init__(self, path, prompt_version: str, judge_model: str):
        self.path = Path(path)
        self.prompt_version = prompt_version
        self.judge_model = judge_model
        self._store: dict = {}

    @staticmethod
    def _norm(s: str) -> str:
        return s.strip().lower()

    def _key(self, a: str, b: str) -> str:
        x, y = sorted((self._norm(a), self._norm(b)))
        return hashlib.sha1((x + "␟" + y).encode("utf-8")).hexdigest()

    def get(self, a: str, b: str):
        return self._store.get(self._key(a, b))

    def put(self, a: str, b: str, verdict: dict) -> None:
        self._store[self._key(a, b)] = verdict

    def missing(self, pairs):
        return [(a, b) for (a, b) in pairs if self._key(a, b) not in self._store]

    def load(self) -> None:
        if not self.path.exists():
            return
        data = json.loads(self.path.read_text())
        if (data.get("prompt_version") != self.prompt_version
                or data.get("judge_model") != self.judge_model):
            self._store = {}  # stale -> invalidate
            return
        self._store = data.get("pairs", {})

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "prompt_version": self.prompt_version,
            "judge_model": self.judge_model,
            "pairs": self._store,
        }
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True))
