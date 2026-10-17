"""Judged hallucination metrics: Mod-POPE, Mod-ROPE, Mod-CHAIR and THRONE class-wise F0.5.

Each metric is rescored from cached model outputs plus static judge files in
``judgments/`` (verdicts and the POPE drop list). Scoring never calls a judge:
a verdict missing for an item raises :class:`MissingVerdicts`, and the item is
then graded with ``tools/judge``. Definitions are in ``docs/methodology.mdx``.

    python -m metrics.judged --root /path/to/mirage-tree --out final_tables.json
    python -m metrics.judged --table tests/data/final_tables.json --csv docs/model-results.csv --docs-tables

``--root`` is a tree with ``datasets/``, ``cache/`` and ``results/`` laid out as
in the reference runs (see ``MODELS`` and the ``*_path`` helpers).
"""

import argparse
import csv
import gzip
import json
import re
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
JUDGMENTS = REPO / "judgments"
N_IMAGES, MIN_WORDS, MIN_SUPPORT = 500, 5, 5
DATASETS = {"coco": "COCO", "vg": "Visual Genome"}
# Key -> display name. Order fixes rank ties.
MODELS = {
    "muse-glimmer-30b": "Muse-Glimmer-30B", "gemma4-12b": "Gemma-4-12B", "gemma4-e4b": "Gemma-4-E4B",
    "aya-vision-8b": "Aya-Vision-8B", "ministral3-14b": "Ministral-3-14B", "ministral3-8b": "Ministral-3-8B",
    "qwen2vl-7b": "Qwen2-VL-7B", "qwen2vl-2b": "Qwen2-VL-2B", "paligemma": "PaliGemma-3B",
}
# Rows reused from the 1,000-image parity runs (cache/old/, results/parity/).
OLD = {"paligemma": "paligemma", "qwen2vl-2b": "Qwen2-VL-2B-Instruct", "qwen2vl-7b": "Qwen2-VL-7B-Instruct"}
# Visual Genome THRONE rows re-judged at the 1,024-token window; the rest had no trimmed captions.
VG_W1024 = {"paligemma", "gemma4-12b", "muse-glimmer-30b"}
_POPE_RE = re.compile(r"Is there an?\s+(.+?)\s+in the image", re.I)


class MissingVerdicts(RuntimeError):
    """Items with no stored verdict; grade them with tools/judge and add the rows."""

    def __init__(self, kind: str, items: List[Tuple[str, str]]):
        super().__init__(f"{len(items)} {kind} items have no verdict, e.g. {items[:5]}")
        self.items = items


# ----------------------------------------------------------------------
# Static judge files
# ----------------------------------------------------------------------

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


# ----------------------------------------------------------------------
# Scorers (pure functions of records and verdicts)
# ----------------------------------------------------------------------

def fbeta(p: float, r: float, beta2: float = 0.25) -> float:
    return (1 + beta2) * p * r / (beta2 * p + r) if p + r else 0.0


def _binary(rows: List[Tuple[bool, bool]]) -> Dict[str, float]:
    tp = sum(1 for l, y in rows if l and y); tn = sum(1 for l, y in rows if not l and not y)
    fp = sum(1 for l, y in rows if not l and y); fn = sum(1 for l, y in rows if l and not y)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    bal = ((tp / (tp + fn) if tp + fn else 0.0) + (tn / (tn + fp) if tn + fp else 0.0)) / 2
    return {"accuracy": (tp + tn) / len(rows), "balanced_accuracy": bal,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "yes_ratio": (tp + fp) / len(rows), "questions": len(rows)}


def mod_pope(records: Iterable[Tuple[str, str, bool, bool]], drop: Dict[Tuple[str, str], str]) -> Dict[str, float]:
    """POPE on ``(image_id, object, label, said_yes)``; Mod-POPE drops listed no-questions."""
    records = list(records)
    raw = _binary([(label, yes) for _, _, label, yes in records])
    mod = _binary([(label, yes) for iid, obj, label, yes in records if label or (iid, obj) not in drop])
    return {"raw": raw["accuracy"], "raw_balanced": raw["balanced_accuracy"], "head": mod["balanced_accuracy"],
            "f1": mod["f1"], "yes_ratio": mod["yes_ratio"], "n_raw": raw["questions"], "n_head": mod["questions"]}


def mod_rope(records: Iterable[Tuple[str, int, str, bool]], verdicts: Dict[Tuple[str, str], str]) -> Dict[str, float]:
    """ROPE on ``(item, box, answer_key, strict_correct)``: strict accuracy and the judged
    share of fair names (correct / graded; unsure excluded, related reported apart)."""
    records = list(records)
    missing = [(item, f"{box}|{ans}") for item, box, ans, _ in records if (item, f"{box}|{ans}") not in verdicts]
    if missing:
        raise MissingVerdicts("ROPE", missing)
    k = Counter(verdicts[(item, f"{box}|{ans}")] for item, box, ans, _ in records)
    boxes, graded = len(records), k["correct"] + k["related"] + k["wrong"]
    correct = k["correct"] / graded
    return {"raw": sum(strict for *_, strict in records) / boxes, "head": correct,
            "related": (k["correct"] + k["related"]) / graded - correct,
            "correct_incl_unsure_as_wrong": correct * (boxes - k["unsure"]) / boxes,
            "n_raw": boxes, "n_head": boxes - k["unsure"]}


def _resolve(item: str, model: str, m: Dict, verdicts: Dict, ref: Iterable[str]) -> str:
    """Mention verdict: annotated, a reference-caption object, not a noun, or a plural/phrase/
    synonym/hypernym of an annotation resolve without the judge."""
    if not m["hall"] or m["word"] in ref:
        return "present"
    if not m["noun"]:
        return "not_object"
    if m["rel"]:
        return "present"
    return verdicts.get((item, f"{model}|{m['word']}"), "missing")


def mod_chair(images: List[Dict], verdicts: Dict, refs: Optional[Dict[str, set]] = None) -> Dict[str, Dict]:
    """CHAIR-I on raw mentions, and Mod-CHAIR on judged ones: each object counted once per
    caption, captions under ``MIN_WORDS`` words mention nothing, coverage = present objects
    named / all present objects named by any model. ``images``: ``{"item", "models":
    {model: {"caption", "mentions"}}}``; ``refs``: item -> reference-caption objects (COCO)."""
    refs = refs or {}
    rows: Dict[str, List[Dict]] = {}
    for img in images:
        item, ref = img["item"], refs.get(img["item"], ())
        short = {m: len(c["caption"].split()) < MIN_WORDS for m, c in img["models"].items()}
        res = {m: [] if short[m] else [(x["word"], _resolve(item, m, x, verdicts, ref)) for x in c["mentions"]]
               for m, c in img["models"].items()}
        missing = [(item, f"{m}|{w}") for m, r in res.items() for w, v in r if v == "missing"]
        if missing:
            raise MissingVerdicts("mention", missing)
        present = {m: {w for w, v in r if v == "present"} for m, r in res.items()}
        pool = set().union(*present.values())
        for m, c in img["models"].items():
            uniq = {}
            for w, v in res[m]:
                uniq.setdefault(w, v)
            obj = [v for v in uniq.values() if v in ("present", "absent")]
            rows.setdefault(m, []).append({
                "raw_m": len(c["mentions"]), "raw_h": sum(x["hall"] for x in c["mentions"]),
                "m": len(obj), "h": obj.count("absent"), "found": len(present[m]), "pool": len(pool)})

    def ratio(rs, num, den):
        d = sum(r[den] for r in rs)
        return sum(r[num] for r in rs) / d if d else 0.0

    out = {}
    for m, rs in rows.items():
        chair_i, coverage = ratio(rs, "h", "m"), ratio(rs, "found", "pool")
        out[m] = {"raw": ratio(rs, "raw_h", "raw_m"), "head": fbeta(1 - chair_i, coverage), "mod_chair_i": chair_i,
                  "coverage": coverage, "mod_chair_s": sum(r["h"] > 0 for r in rs) / len(rs),
                  "n_raw": sum(r["raw_m"] for r in rs), "n_head": sum(r["m"] for r in rs)}
    return out


def throne_classwise(details: List[Dict], support: Counter, min_support: Optional[int] = None,
                     min_words: Optional[int] = MIN_WORDS,
                     is_present: Optional[Callable[[int, str], bool]] = None) -> Dict[str, float]:
    """THRONE class-wise P and R (macro over categories), then F0.5, on ``per_image_details``.

    ``min_words``: shorter captions predict nothing. ``is_present(i, category)``: a predicted
    category verified present in image ``i`` counts as true (completed labels). Categories
    with fewer than ``min_support`` annotated images (``support``) are left out of the average.
    """
    tp, fp, fn = Counter(), Counter(), Counter()
    for i, d in enumerate(details):
        dq = set(d["disqualified_objects"])
        short = min_words is not None and len(d["caption"].split()) < min_words
        pred = set() if short else set(d["predicted_objects"]) - dq
        truth = set(d["gt_objects"]) - dq
        if is_present is not None:
            truth |= {c for c in pred if is_present(i, c)}
        for c in pred & truth: tp[c] += 1
        for c in pred - truth: fp[c] += 1
        for c in truth - pred: fn[c] += 1
    cats = set(tp) | set(fp) | set(fn)
    if min_support is not None:
        cats &= {c for c, n in support.items() if n >= min_support}
    cats = sorted(cats)
    ps = [tp[c] / (tp[c] + fp[c]) for c in cats if tp[c] + fp[c]]
    rs = [tp[c] / (tp[c] + fn[c]) for c in cats if tp[c] + fn[c]]
    p = sum(ps) / len(ps) if ps else 0.0
    r = sum(rs) / len(rs) if rs else 0.0
    return {"head": fbeta(p, r), "P": p, "R": r, "n_head": len(cats)}


# ----------------------------------------------------------------------
# Records from a results tree (datasets, prediction caches, result JSONs)
# ----------------------------------------------------------------------

def _loader(root: Path, ds: str, max_samples: int):
    if ds == "vg":
        from dataloader.visual_genome import VisualGenomeDataLoader
        loader = VisualGenomeDataLoader(data_path=str(root / "datasets/visual_genome/objects.json"),
                                        images_path=str(root / "datasets/visual_genome/VG_100K"), shuffle=False)
    else:
        from dataloader.coco import COCODataLoader
        loader = COCODataLoader(data_path=str(root / "datasets/coco/annotations/instances_val2017.json"),
                                images_path=str(root / "datasets/coco/val2017"), shuffle=False)
    loader.load_data(max_samples=max_samples)
    return loader


def cache_path(root: Path, model: str, ds: str, stage: str) -> Path:
    cell = f"old/{OLD[model]}" if model in OLD else model
    return root / "cache" / f"{cell}_{ds}_{stage}" / "predictions.sqlite"


def captions_path(root: Path, model: str, ds: str) -> Path:
    return root / ("results/parity" if model in OLD else "results/grid") / f"{OLD.get(model, model)}_{ds}_captions.json"


def throne_path(root: Path, model: str, ds: str) -> Path:
    w1024 = ds == "coco" or model in VG_W1024
    return root / "results/grid" / f"{model}_{ds}_throne{'_w1024' if w1024 else ''}.json"


def _rows(path: Path):
    from predict.cache import SQLiteResponseCache

    if not path.exists():
        raise FileNotFoundError(path)
    return SQLiteResponseCache(path, readonly=True).rows()


def pope_records(root: Path, ds: str) -> Dict[str, List[Tuple[str, str, bool, bool]]]:
    from metrics.pope import POPEMetric
    from predict.identity import hash_image

    loader = _loader(root, ds, 2000)
    samples = [s for s in loader if s.get("categories")][:N_IMAGES]
    images = {hash_image(loader.load_image(s)): (str(s["image_id"]), set(s["categories"])) for s in samples}
    pope = POPEMetric(mode="random", num_questions_per_image=6)
    out = {}
    for model in MODELS:
        recs = []
        for prompt, image_hash, resp in _rows(cache_path(root, model, ds, "pope")):
            m = _POPE_RE.search(prompt)
            if image_hash in images and m:
                iid, cats = images[image_hash]
                recs.append((iid, m.group(1), m.group(1) in cats, pope.process_response(resp)))
        out[model] = recs
    return out


def rope_records(root: Path, ds: str) -> Dict[str, List[Tuple[str, int, str, bool]]]:
    from metrics.rope import ROPEMetric, _rope_match_strict, judged_answer
    from predict.identity import hash_image

    loader = _loader(root, ds, 3000 if ds == "coco" else 2000)
    names = loader.get_category_names() if ds == "coco" else loader.get_object_names()
    boxes = loader.extract_bboxes_for_rope(num_objects=5)
    rope = ROPEMetric(num_objects=5, class_list=names)
    questions = []
    for s in [s for s in loader if str(s["image_id"]) in boxes][:N_IMAGES]:
        bxs = boxes[str(s["image_id"])]
        h = hash_image(rope.draw_bounding_boxes(loader.load_image(s), bxs))
        questions += [(f"{ds}_{s['image_id']}", b, bx["name"], h) for b, bx in enumerate(bxs, start=1)]
    want = {q[3] for q in questions}
    out = {}
    for model in MODELS:
        got = {(h, p): r for p, h, r in _rows(cache_path(root, model, ds, "rope")) if h in want}
        recs = []
        for item, b, gt, h in questions:
            resp = got.get((h, rope.generate_prompt_for_box(b)))
            if resp is not None:
                pred = rope.parse_single_response(resp)
                recs.append((item, b, judged_answer(resp, pred), _rope_match_strict(pred, gt)))
        out[model] = recs
    return out


def mention_records(root: Path, ds: str) -> Tuple[List[Dict], Dict[str, set]]:
    """Tagged caption mentions per image and model, and COCO reference-caption objects."""
    from metrics.chair import ChairMetric

    loader = _loader(root, ds, 2000)
    if ds == "vg":
        from metrics.chair import vg_vocabulary
        vocab = vg_vocabulary(loader.get_object_names())
        chair = ChairMetric(object_synsets={n: [n] for n in vocab}, use_coco_synonyms=False)
    else:
        chair = ChairMetric(use_coco_synonyms=True)
    samples = [s for s in loader if s.get("categories")][:N_IMAGES]
    caps = {m: json.load(open(captions_path(root, m, ds)))["caption_details"][:N_IMAGES] for m in MODELS}
    images = []
    for i, s in enumerate(samples):
        gt = list(s["categories"])
        images.append({"item": f"{ds}_{s['image_id']}", "gt": gt, "models": {
            m: {"caption": c[i]["caption"], "mentions": chair.tag_mentions(c[i]["caption"], gt)} for m, c in caps.items()}})
    refs = {}
    if ds == "coco":  # CHAIR paper: objects named in the reference captions count as present
        by_image: Dict[str, List[str]] = {}
        for a in json.load(open(root / "datasets/coco/annotations/captions_val2017.json"))["annotations"]:
            by_image.setdefault(f"coco_{a['image_id']}", []).append(a["caption"])
        refs = {img["item"]: {w for cap in by_image.get(img["item"], []) for w in chair.caption_to_words(cap)[1]}
                for img in images}
    return images, refs


def _verified_present(images: List[Dict], verdicts: Dict) -> List[set]:
    """Per image: words any model mentioned that are annotated, related to an annotation, or judged present."""
    out = []
    for img in images:
        words = set()
        for m, c in img["models"].items():
            for x in c["mentions"]:
                if not x["hall"] or (x["noun"] and x["rel"]) or (
                        x["noun"] and verdicts.get((img["item"], f"{m}|{x['word']}")) == "present"):
                    words.add(x["word"])
        out.append(words)
    return out


def build_table(root: Path) -> Dict[str, Dict]:
    from metrics.chair import variant_of_gt

    drop, mentions, rope_v = load_drop_list(), load_verdicts("mentions"), load_verdicts("rope")
    table = {}
    for ds in DATASETS:
        pope, rope = pope_records(root, ds), rope_records(root, ds)
        images, refs = mention_records(root, ds)
        chair = mod_chair(images, mentions, refs)
        support = Counter(g.lower() for img in images for g in set(img["gt"]))
        vp = _verified_present(images, mentions) if ds == "vg" else None
        rows = {}
        for model in MODELS:
            details = json.load(open(throne_path(root, model, ds)))["per_image_details"][:N_IMAGES]
            if any(d["caption"] != img["models"][model]["caption"] for d, img in zip(details, images)):
                raise ValueError(f"{model} {ds}: THRONE captions are not aligned with the caption run")
            min_support = MIN_SUPPORT if ds == "vg" else None
            paper = throne_classwise(details, support, min_support, min_words=None)
            present = (lambda i, c: c in vp[i] or variant_of_gt(c, list(vp[i]))[0] in ("exact", "plural", "phrase")) \
                if vp else None
            head = throne_classwise(details, support, min_support, is_present=present)
            rows[model] = {"POPE": mod_pope(pope[model], drop.get(ds, {})), "ROPE": mod_rope(rope[model], rope_v),
                           "CHAIR": chair[model], "THRONE": {"raw": paper["head"], **head}}
        for metric in ("POPE", "ROPE", "CHAIR", "THRONE"):
            for kind in ("raw", "raw_balanced", "head") if metric == "POPE" else ("raw", "head"):
                low = metric == "CHAIR" and kind == "raw"  # raw CHAIR-I: lower is better
                order = sorted(MODELS, key=lambda m: rows[m][metric][kind] * (1 if low else -1))
                for m in MODELS:
                    rows[m][metric][f"rank_{kind}"] = order.index(m) + 1
        table[ds] = rows
    return table


# ----------------------------------------------------------------------
# Docs output
# ----------------------------------------------------------------------

def write_csv(table: Dict, path: Path) -> None:
    """Headline columns for the homepage and README tables."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["model", "dataset", "mod_pope_balanced_accuracy", "mod_rope_accuracy", "mod_chair_f05", "throne_f05"])
        for ds, label in DATASETS.items():
            for m, name in MODELS.items():
                r = table[ds][m]
                w.writerow([name, label, *(f"{r[k]['head']:.3f}" for k in ("POPE", "ROPE", "CHAIR", "THRONE"))])


def docs_tables(table: Dict) -> str:
    """Raw → Mod appendix tables (percent) for docs/methodology.mdx."""
    lines = []
    for ds, label in DATASETS.items():
        lines += [f"**{label}**", "", "| Model | POPE acc → bal. acc → Mod-POPE | ROPE strict → Mod-ROPE "
                  "| CHAIR-I ↓ → Mod-CHAIR F0.5 ↑ | THRONE paper-style → Mirage |", "| --- | --- | --- | --- | --- |"]
        for m, name in MODELS.items():
            r = table[ds][m]
            cells = [" → ".join(f"{r[k][x] * 100:.1f}" for x in ("raw", "raw_balanced", "head") if x in r[k])
                     for k in ("POPE", "ROPE", "CHAIR", "THRONE")]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path, help="Results tree to rescore")
    parser.add_argument("--table", type=Path, help="Existing table JSON instead of --root")
    parser.add_argument("--out", type=Path, help="Write the table JSON here")
    parser.add_argument("--csv", type=Path, help="Write the headline CSV here")
    parser.add_argument("--docs-tables", action="store_true", help="Print the raw → Mod appendix tables")
    args = parser.parse_args()
    if bool(args.root) == bool(args.table):
        parser.error("pass exactly one of --root and --table")
    table = build_table(args.root) if args.root else json.load(open(args.table))["table"]
    if args.out:
        json.dump({"table": table, "models": MODELS}, open(args.out, "w"), indent=1)
    if args.csv:
        write_csv(table, args.csv)
    if args.docs_tables:
        print(docs_tables(table))


if __name__ == "__main__":
    main()
