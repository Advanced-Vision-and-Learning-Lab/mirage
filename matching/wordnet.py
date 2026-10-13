"""WordNet synonym matcher — the active synonym tier shipped in issue #24.

Phase-0 spike (tests/data/synonym_gold.jsonl, 200 pairs) measured this at
**precision 1.00 / recall 0.46 / 0 false positives** with ``max_hops=0`` — the
shipped default. It is deterministic, offline, and needs no GPU/SU.

Algorithm (head-noun, shared-synset):
    head = last alphabetic token of each phrase
    match if heads are equal (catches "soccer ball" / "sports ball" via "ball"),
    else if the two heads share a WordNet NOUN synset (catches bike/bicycle,
    rock/rocks, egg/eggs, sofa/couch — NLTK Morphy handles the plural lemmas).

``max_hops > 0`` additionally walks hypernyms, but that also merges hyponym
pairs the grading protocol wants kept apart (e.g. mug/cup, woman/person), which
measured lower precision (0.857 at hops=1). Left off by default on purpose.
"""
from __future__ import annotations


class WordNetMatcher:
    name = "wordnet"

    def __init__(self, max_hops: int = 0):
        self.max_hops = max_hops
        self._wn = None

    def _ensure(self):
        """Lazy-load the WordNet corpus. Raises LookupError if not downloaded
        (run: python -m nltk.downloader wordnet omw-1.4)."""
        if self._wn is None:
            from nltk.corpus import wordnet as wn
            wn.ensure_loaded()
            self._wn = wn
        return self._wn

    @staticmethod
    def _head(s: str) -> str:
        toks = [t for t in "".join(c if c.isalpha() or c == " " else " "
                                   for c in s.lower()).split()]
        return toks[-1] if toks else ""

    def _synsets(self, word: str):
        wn = self._wn
        return set(wn.synsets(word, pos=wn.NOUN))

    def match(self, pred: str, gt: str) -> bool:
        wn = self._ensure()
        ha, hb = self._head(pred), self._head(gt)
        if not ha or not hb:
            return False
        if ha == hb:
            return True
        sa, sb = self._synsets(ha), self._synsets(hb)
        if not sa or not sb:
            return False
        if sa & sb:
            return True
        if self.max_hops > 0:
            for src, dst in ((sa, sb), (sb, sa)):
                frontier = set(src)
                for _ in range(self.max_hops):
                    frontier = {h for s in frontier for h in s.hypernyms()} | frontier
                if frontier & dst:
                    return True
        return False
