"""Judge: verdicts for the judged metrics, from the frozen store first and a live judge for misses."""

import re
from pathlib import Path
from typing import Dict, List, Optional

from metrics.base_metric import MIN_WORDS, MissingVerdicts

from .store import VerdictStore


def mention_requests(images: List[Dict], verdicts: Dict, refs: Optional[Dict[str, set]] = None,
                     image_paths: Optional[Dict[str, str]] = None) -> List[Dict]:
    """judge-mentions inputs: per image, each (model, word) that ChairMetric.score would find missing."""
    from metrics.chair import _resolve

    refs, out = refs or {}, []
    for img in images:
        item, ref, seen, mentions = img["item"], refs.get(img["item"], ()), set(), []
        for m, c in img["models"].items():
            if len(c["caption"].split()) < MIN_WORDS:
                continue
            sents = re.split(r"(?<=[.!?])\s+", c["caption"])
            for x in c["mentions"]:
                if (m, x["word"]) in seen or _resolve(item, m, x, verdicts, ref) != "missing":
                    continue
                seen.add((m, x["word"]))
                ctx = next((s for s in sents if re.search(r"\b" + re.escape(x["tok"]) + r"\b", s, re.I)),
                           c["caption"][:300])
                mentions.append({"model": m, "word": x["word"], "context": ctx[:400]})
        if mentions:
            if image_paths is None:
                raise ValueError(f"{item} has mentions to judge; pass image_paths (item -> image file)")
            out.append({"image_id": item, "image": str(image_paths[item]), "mentions": mentions})
    return out


class Judge:
    """Verdicts for scoring: the frozen :class:`VerdictStore` answers first; with a live judge
    (:class:`judge.claude.ClaudeCLI`) missing mention verdicts are graded and kept in the live
    shard, otherwise the metric's ``score`` raises :class:`MissingVerdicts` for them. ROPE has no
    live path yet (judge-rope needs box-annotated images and raw replies), so its misses always
    raise; POPE drop decisions and the committee exist only for the frozen images, so they raise
    outside them.
    """

    def __init__(self, store: VerdictStore, live=None):
        self.store, self.live = store, live

    @classmethod
    def load(cls, path: Optional[Path] = None, live: bool = False, live_dir: Optional[Path] = None) -> "Judge":
        """The frozen store (see :meth:`VerdictStore.open`); ``live=True`` adds the Claude judge,
        writing to ``live_dir`` (default ``<store>/live``)."""
        store = VerdictStore.open(path, live=live, live_dir=live_dir if live else None)
        if not live:
            return cls(store)
        from .claude import ClaudeCLI

        return cls(store, ClaudeCLI())

    def __call__(self, kind: str, records, ds: Optional[str] = None, refs: Optional[Dict[str, set]] = None,
                 image_paths: Optional[Dict[str, str]] = None) -> Dict:
        """Verdicts for ``records`` of ``kind``.

        Args:
            kind: "pope" (POPEMetric records; needs ``ds``), "rope" (ROPEMetric records) or
                "mentions" (ChairMetric images, with ``refs`` on COCO).
            image_paths: item -> image file shown to the live judge (mentions only; required
                when it has mentions to grade).

        Returns:
            The POPE drop list of ``ds``, or ``{(item, key): verdict}``.
        """
        if kind == "pope":
            covered = self.store.covered(ds)
            missing = sorted({(f"{ds}_{iid}", "drop-list") for iid, *_ in records if f"{ds}_{iid}" not in covered})
            if missing:
                raise MissingVerdicts("POPE", missing)
            return self.store.drop_list(ds)
        verdicts = self.store.verdicts(kind)
        if self.live is not None and kind == "mentions":
            requests = mention_requests(records, verdicts, refs, image_paths)
            if requests:
                from .claude import skill_sha

                self.store.add(kind, self.live.grade(kind, requests), self.live.model, skill_sha(f"judge-{kind}"))
                verdicts = self.store.verdicts(kind)
        return verdicts

    def committee(self, ds: str, images: List[Dict]) -> Dict[str, Dict[str, set]]:
        """Committee rows (``present``, ``verified``) for the items of ``images``.

        Raises:
            MissingVerdicts: Some items are outside the frozen images.
        """
        com = self.store.committee(ds)
        missing = [(img["item"], "committee") for img in images if img["item"] not in com]
        if missing:
            raise MissingVerdicts("committee", missing)
        return {img["item"]: com[img["item"]] for img in images}
