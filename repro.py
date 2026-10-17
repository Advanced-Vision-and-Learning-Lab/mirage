"""Rebuild the published judged table (Mod-POPE, Mod-ROPE, Mod-CHAIR, THRONE class-wise F0.5) from a results tree.

The manual check of all 612 published values. Each metric is rescored with its own
``score`` from cached model outputs plus the frozen judge files (``judge.VerdictStore``:
verdicts and the POPE drop list). Scoring never calls a judge: a verdict missing for an
item raises :class:`MissingVerdicts`. The literals below (500 images, pools of 2000 and
3000, MODELS, OLD, VG_W1024) are frozen to the reference runs and never follow the
metrics' defaults.

    python repro.py --root /path/to/mirage-tree --check            # Python 3.12+; exit 1 on any difference
    python repro.py --root /path/to/mirage-tree --export-records tests/data/records.json.gz
    python repro.py --records tests/data/records.json.gz --check   # the same check without the tree
    python repro.py --root /path/to/mirage-tree --export-committee judgments/committee.json
    python repro.py --table judgments/final_tables.json --csv model-results.csv

``--root`` is a tree with ``datasets/``, ``cache/`` and ``results/`` laid out as
in the reference runs (see ``MODELS`` and the ``*_path`` helpers).
"""

import argparse
import csv
import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from judge.store import VerdictStore

N_IMAGES = 500
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
    rows = list(SQLiteResponseCache(path, readonly=True).rows())
    if len({(p, h) for p, h, _ in rows}) != len(rows):  # the joins below assume one row per (prompt, image)
        raise ValueError(f"{path}: (prompt, image_hash) is not unique")
    return rows


def pope_records(root: Path, ds: str, samples: List[Dict], loader) -> Dict[str, List[Tuple[str, str, bool, str]]]:
    """Per model, ``(image_id, object, label, reply)`` for the POPE questions on ``samples``."""
    from metrics.pope import QUESTION_RE
    from predict.identity import hash_image

    images = {hash_image(loader.load_image(s)): (str(s["image_id"]), set(s["categories"])) for s in samples}
    out = {}
    for model in MODELS:
        recs = []
        for prompt, image_hash, resp in _rows(cache_path(root, model, ds, "pope")):
            m = QUESTION_RE.search(prompt)
            if image_hash in images and m:
                iid, cats = images[image_hash]
                recs.append((iid, m.group(1), m.group(1) in cats, resp))
        out[model] = recs
    return out


def rope_records(root: Path, ds: str) -> Dict[str, List[Tuple[str, int, str, str]]]:
    """Per model, ``(item, box, ground-truth name, reply)`` for the first ``N_IMAGES`` ROPE images."""
    from metrics.rope import ROPEMetric
    from predict.identity import hash_image

    loader = _loader(root, ds, 3000 if ds == "coco" else 2000)
    boxes = loader.extract_bboxes_for_rope(num_objects=5)
    rope = ROPEMetric(num_objects=5)
    questions = []
    for s in [s for s in loader if str(s["image_id"]) in boxes][:N_IMAGES]:
        bxs = boxes[str(s["image_id"])]
        h = hash_image(rope.draw_bounding_boxes(loader.load_image(s), bxs))
        questions += [(f"{ds}_{s['image_id']}", b, bx["name"], h) for b, bx in enumerate(bxs, start=1)]
    want = {q[3] for q in questions}
    out = {}
    for model in MODELS:
        got = {(h, p): r for p, h, r in _rows(cache_path(root, model, ds, "rope")) if h in want}
        out[model] = [(item, b, gt, got[(h, rope.generate_prompt_for_box(b))]) for item, b, gt, h in questions
                      if (h, rope.generate_prompt_for_box(b)) in got]
    return out


def load_records(root: Path) -> Dict[str, Dict]:
    """Everything scoring reads from ``root``, per dataset, as JSON-ready data: the model replies
    joined to their items (the only step that needs the images), the captions with their THRONE
    evaluator outputs, COCO's reference captions and the Visual Genome pool's object names.
    :func:`rebuild` scores it; ``--export-records`` writes it."""
    out = {}
    for ds in DATASETS:
        loader = _loader(root, ds, 2000)
        samples = [s for s in loader if s.get("categories")][:N_IMAGES]
        caps = {m: json.load(open(captions_path(root, m, ds)))["caption_details"][:N_IMAGES] for m in MODELS}
        images = [{"item": f"{ds}_{s['image_id']}", "gt": list(s["categories"]),
                   "captions": {m: c[i]["caption"] for m, c in caps.items()}} for i, s in enumerate(samples)]
        throne = {}
        for model in MODELS:
            details = json.load(open(throne_path(root, model, ds)))["per_image_details"][:N_IMAGES]
            if any(d["caption"] != img["captions"][model] for d, img in zip(details, images)):
                raise ValueError(f"{model} {ds}: THRONE captions are not aligned with the caption run")
            # rebuild() puts the caption back from "images"
            throne[model] = [{k: d[k] for k in ("gt_objects", "predicted_objects", "disqualified_objects")}
                             for d in details]
        rec = {"pope": pope_records(root, ds, samples, loader), "rope": rope_records(root, ds),
               "images": images, "throne": throne}
        if ds == "vg":
            rec["object_names"] = loader.get_object_names()
        else:
            items = {img["item"] for img in images}
            rec["references"] = [a for a in json.load(open(root / "datasets/coco/annotations/captions_val2017.json"))
                                 ["annotations"] if f"coco_{a['image_id']}" in items]
        out[ds] = rec
    return out


def rebuild(records: Dict[str, Dict], store: Optional[VerdictStore] = None) -> Tuple[Dict[str, Dict], Dict]:
    """The published table and committee.json, scored from ``records`` (see :func:`load_records`)
    with verdicts from ``store`` (default :meth:`VerdictStore.open`)."""
    from types import SimpleNamespace

    from metrics.base_metric import MIN_WORDS
    from metrics.chair import ChairMetric, _resolve
    from metrics.pope import POPEMetric
    from metrics.rope import ROPEMetric, judged_answer
    from metrics.throne import ThroneMetric, completed_labels

    store = store or VerdictStore.open()
    mentions, rope_v = store.verdicts("mentions"), store.verdicts("rope")
    pope, rope = POPEMetric(mode="random", num_questions_per_image=6), ROPEMetric(num_objects=5)
    table, committee = {}, {}
    for ds in DATASETS:
        rec = records[ds]
        chair = ChairMetric.for_dataset(ds, SimpleNamespace(get_object_names=lambda: rec["object_names"]))
        images = [{"item": img["item"], "gt": img["gt"], "models": {
            m: {"caption": c, "mentions": chair.tag_mentions(c, img["gt"])} for m, c in img["captions"].items()}}
            for img in rec["images"]]
        refs = chair.reference_objects({"annotations": rec["references"]}, images) if ds == "coco" else {}
        chair_scores = ChairMetric.score(images, mentions, refs)
        support = Counter(g.lower() for img in images for g in set(img["gt"]))
        verified = ChairMetric.verified_present(images, mentions)
        throne = ThroneMetric.for_dataset(ds)  # class-wise over categories with MIN_SUPPORT images on VG
        present = completed_labels(verified) if ds == "vg" else None
        rows = {}
        for model in MODELS:
            rope_recs = []
            for item, b, gt, resp in rec["rope"][model]:
                pred = rope.parse_single_response(resp)
                rope_recs.append((item, b, judged_answer(resp, pred), rope.strict(pred, gt)))
            details = [{**d, "caption": img["captions"][model]} for d, img in zip(rec["throne"][model], rec["images"])]
            pope_recs = [(iid, obj, label, pope.process_response(resp)) for iid, obj, label, resp in rec["pope"][model]]
            rows[model] = {"POPE": POPEMetric.score(pope_recs, store.drop_list(ds)),
                           "ROPE": ROPEMetric.score(rope_recs, rope_v),
                           "CHAIR": chair_scores[model], "THRONE": throne.score(details, support, present)}
        for metric in ("POPE", "ROPE", "CHAIR", "THRONE"):
            for kind in ("raw", "raw_balanced", "head") if metric == "POPE" else ("raw", "head"):
                low = metric == "CHAIR" and kind == "raw"  # raw CHAIR-I: lower is better
                order = sorted(MODELS, key=lambda m: rows[m][metric][kind] * (1 if low else -1))
                for m in MODELS:
                    rows[m][metric][f"rank_{kind}"] = order.index(m) + 1
        table[ds] = rows
        # committee.json: the words the frozen pool (MODELS) resolved present (Mod-CHAIR's coverage
        # pool) and its verified-present words (Visual Genome THRONE's completed labels)
        committee[ds] = {img["item"]: {
            "present": sorted({x["word"] for m, c in img["models"].items() if len(c["caption"].split()) >= MIN_WORDS
                               for x in c["mentions"]
                               if _resolve(img["item"], m, x, mentions, refs.get(img["item"], ())) == "present"}),
            "verified": sorted(vp)} for img, vp in zip(images, verified)}
    return table, committee


# ----------------------------------------------------------------------
# Outputs
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


def _leaves(node, path=()):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _leaves(v, path + (k,))
    else:
        yield path, node


def _preflight() -> None:
    """--check needs Python 3.12+ and the NLTK data, and never downloads it."""
    import sys

    import nltk

    if sys.version_info < (3, 12):
        # The reference sums class-wise THRONE P and R in another order; only 3.12's sum() agrees to the last bit.
        raise SystemExit(f"--check needs Python 3.12 or newer, not {sys.version.split()[0]}")
    for resource in ("tokenizers/punkt_tab", "taggers/averaged_perceptron_tagger_eng", "corpora/stopwords"):
        nltk.data.find(resource)
    from nltk.corpus import wordnet  # LookupError when missing (it is usually a zip)

    if wordnet.get_version() != "3.0":
        raise SystemExit(f"--check needs WordNet 3.0, not {wordnet.get_version()}")


def read_records(path: Path) -> Dict[str, Dict]:
    return json.loads(gzip.decompress(Path(path).read_bytes()))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path, help="Results tree to rescore")
    parser.add_argument("--records", type=Path, help="Records exported from a results tree (--export-records)")
    parser.add_argument("--table", type=Path, help="Existing table JSON instead of --root")
    parser.add_argument("--judgments", type=Path,
                        help="Judgments directory (default $MIRAGE_JUDGMENTS_DIR, else judgments/)")
    parser.add_argument("--out", type=Path, help="Write the table JSON here")
    parser.add_argument("--csv", type=Path, help="Write the headline CSV here")
    parser.add_argument("--check", nargs="?", const=True, metavar="REF",
                        help="Compare every value with REF (default: the store's final_tables.json) and "
                             "with the store's committee.json if it has one; exit 1 on any difference")
    parser.add_argument("--export-committee", type=Path, metavar="OUT", help="Write committee.json here")
    parser.add_argument("--export-records", type=Path, metavar="OUT", help="Write --root's records here (gzip JSON)")
    args = parser.parse_args(argv)
    if sum(map(bool, (args.root, args.records, args.table))) != 1:
        parser.error("pass exactly one of --root, --records and --table")
    if (args.check or args.export_committee) and args.table:
        parser.error("--check and --export-committee need --root or --records")
    if args.export_records and not args.root:
        parser.error("--export-records needs --root")
    if args.check:
        _preflight()
    store = com = None
    if args.table:
        table = json.load(open(args.table))["table"]
    else:
        store = VerdictStore.open(args.judgments)  # frozen only; checks the verdict files' sha256
        if args.check:  # read (and sha256-check) the reference before the long rebuild
            ref = store.reference_table() if args.check is True else json.load(open(args.check))["table"]
        records = read_records(args.records) if args.records else load_records(args.root)
        if args.export_records:
            args.export_records.write_bytes(gzip.compress(json.dumps(records).encode(), mtime=0))
        table, com = rebuild(records, store)
    if args.out:
        json.dump({"table": table, "models": MODELS}, open(args.out, "w"), indent=1)
    if args.csv:
        write_csv(table, args.csv)
    if args.export_committee:
        args.export_committee.write_text(json.dumps(com, indent=1) + "\n")
    status = 0
    if args.check:
        want, got = dict(_leaves(ref)), dict(_leaves(table))
        bad = sorted(p for p in want.keys() | got.keys() if want.get(p, "missing") != got.get(p, "missing"))
        print(f"{len(want)} reference values, {len(got)} rebuilt, {len(bad)} differ")
        for p in bad[:20]:
            print("  ", "/".join(map(str, p)), want.get(p, "missing"), got.get(p, "missing"))
        status = int(bool(bad))
        shipped = {ds: store.committee(ds) for ds in DATASETS}
        rebuilt = {ds: {i: {k: set(v) for k, v in r.items()} for i, r in com[ds].items()} for ds in DATASETS}
        if any(shipped.values()) and rebuilt != shipped:
            print("committee.json differs from the committee rebuilt from the records")
            status = 1
    return status


if __name__ == "__main__":
    raise SystemExit(main())
