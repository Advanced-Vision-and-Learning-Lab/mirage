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
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from judge.store import load_drop_list, load_verdicts
from metrics.throne import MIN_SUPPORT, classwise

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
    return SQLiteResponseCache(path, readonly=True).rows()


def pope_records(root: Path, ds: str) -> Dict[str, List[Tuple[str, str, bool, bool]]]:
    from metrics.pope import QUESTION_RE, POPEMetric
    from predict.identity import hash_image

    loader = _loader(root, ds, 2000)
    samples = [s for s in loader if s.get("categories")][:N_IMAGES]
    images = {hash_image(loader.load_image(s)): (str(s["image_id"]), set(s["categories"])) for s in samples}
    pope = POPEMetric(mode="random", num_questions_per_image=6)
    out = {}
    for model in MODELS:
        recs = []
        for prompt, image_hash, resp in _rows(cache_path(root, model, ds, "pope")):
            m = QUESTION_RE.search(prompt)
            if image_hash in images and m:
                iid, cats = images[image_hash]
                recs.append((iid, m.group(1), m.group(1) in cats, pope.process_response(resp)))
        out[model] = recs
    return out


def rope_records(root: Path, ds: str) -> Dict[str, List[Tuple[str, int, str, bool]]]:
    from metrics.rope import ROPEMetric, judged_answer
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
        recs = []
        for item, b, gt, h in questions:
            resp = got.get((h, rope.generate_prompt_for_box(b)))
            if resp is not None:
                pred = rope.parse_single_response(resp)
                recs.append((item, b, judged_answer(resp, pred), rope.strict(pred, gt)))
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


def build_table(root: Path) -> Dict[str, Dict]:
    from metrics.chair import ChairMetric, variant_of_gt
    from metrics.pope import POPEMetric
    from metrics.rope import ROPEMetric

    drop, mentions, rope_v = load_drop_list(), load_verdicts("mentions"), load_verdicts("rope")
    table = {}
    for ds in DATASETS:
        pope, rope = pope_records(root, ds), rope_records(root, ds)
        images, refs = mention_records(root, ds)
        chair = ChairMetric.score(images, mentions, refs)
        support = Counter(g.lower() for img in images for g in set(img["gt"]))
        vp = ChairMetric.verified_present(images, mentions) if ds == "vg" else None
        rows = {}
        for model in MODELS:
            details = json.load(open(throne_path(root, model, ds)))["per_image_details"][:N_IMAGES]
            if any(d["caption"] != img["models"][model]["caption"] for d, img in zip(details, images)):
                raise ValueError(f"{model} {ds}: THRONE captions are not aligned with the caption run")
            min_support = MIN_SUPPORT if ds == "vg" else None
            paper = classwise(details, support, min_support, min_words=None)
            present = (lambda i, c: c in vp[i] or variant_of_gt(c, list(vp[i]))[0] in ("exact", "plural", "phrase")) \
                if vp else None
            head = classwise(details, support, min_support, is_present=present)
            rows[model] = {"POPE": POPEMetric.score(pope[model], drop.get(ds, {})),
                           "ROPE": ROPEMetric.score(rope[model], rope_v),
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
