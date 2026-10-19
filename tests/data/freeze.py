"""Rebuild the test fixtures from a results tree laid out as in the reference runs (see repro.py).

    python tests/data/freeze.py --root /path/to/mirage-tree     # Python 3.12+, read-only on the tree

- records.json.gz: ``repro.py --export-records``, checked against final_tables.json and committee.json.
- wiring/: the first two annotated images and the first two ROPE images of COCO and Visual Genome,
  their annotations, and gemma4-e4b's cached replies and THRONE evaluator answers for them.

Only an MR that means to change these inputs re-runs this; test_wiring's expected.json is then
refreshed with ``python tests/test_wiring.py --freeze`` and the diff reviewed.
"""

import argparse
import gzip
import json
import shutil
import sqlite3
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent
sys.path.insert(0, str(DATA.parents[1]))

import repro  # noqa: E402

MODEL = "gemma4-e4b"
COLUMNS = ("key", "key_version", "provider", "model_class", "model_id", "revision", "quantization",
           "prompt", "image_hash", "gen_config", "response")


def _rows(db: Path, where: str, args=()):
    import zlib

    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    query = f"SELECT {', '.join(COLUMNS)} FROM predictions WHERE {where}"
    rows = [dict(zip(COLUMNS, r)) for r in con.execute(query, args)]
    for r in rows:
        r["response"] = zlib.decompress(r["response"]).decode("utf-8")
    return rows


def wiring(root: Path, out: Path) -> None:
    from metrics.rope import ROPEMetric
    from predict.identity import hash_image

    rope, rows = ROPEMetric(num_objects=5), []
    for ds in ("coco", "vg"):
        loader = repro._loader(root, ds, 2000)
        first = [s for s in loader if s.get("categories")][:2]
        rope_loader = repro._loader(root, ds, 3000) if ds == "coco" else loader
        boxes = rope_loader.extract_bboxes_for_rope(num_objects=5)
        rope_first = [s for s in rope_loader if str(s["image_id"]) in boxes][:2]
        hashes = ({hash_image(loader.load_image(s)) for s in first}
                  | {hash_image(rope.draw_bounding_boxes(rope_loader.load_image(s), boxes[str(s["image_id"])]))
                     for s in rope_first})
        ids = {s["image_id"] for s in first + rope_first}
        if ds == "coco":
            ann = root / "datasets/coco/annotations"
            inst = json.load(open(ann / "instances_val2017.json"))
            caps = json.load(open(ann / "captions_val2017.json"))
            images = [i for i in inst["images"] if i["id"] in ids]
            dst = out / "coco"
            (dst / "annotations").mkdir(parents=True, exist_ok=True)
            json.dump({"images": images, "annotations": [a for a in inst["annotations"] if a["image_id"] in ids],
                       "categories": inst["categories"]}, open(dst / "annotations/instances_val2017.json", "w"))
            json.dump({"images": images, "annotations": [a for a in caps["annotations"] if a["image_id"] in ids]},
                      open(dst / "annotations/captions_val2017.json", "w"))
            files = [(root / "datasets/coco/val2017" / i["file_name"], dst / "val2017" / i["file_name"])
                     for i in images]
        else:
            objects = json.load(open(root / "datasets/visual_genome/objects.json"))
            dst = out / "visual_genome"
            dst.mkdir(parents=True, exist_ok=True)
            # The last entry has no image file, so the loader skips it, but its names keep the
            # vocabulary (every object name in objects.json) as in the full file.
            names = [{"names": [n]} for n in loader.get_object_names()]
            json.dump([o for o in objects if o["image_id"] in ids] + [{"image_id": 0, "objects": names}],
                      open(dst / "objects.json", "w"))
            files = [(root / f"datasets/visual_genome/VG_100K/{i}.jpg", dst / f"VG_100K/{i}.jpg")
                     for i in sorted(ids)]
        for src, to in files:
            to.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, to)
        marks = ",".join("?" * len(hashes))
        got = {stage: _rows(repro.cache_path(root, MODEL, ds, stage), f"image_hash IN ({marks})", sorted(hashes))
               for stage in ("captions", "pope", "rope")}
        rows += [r for stage_rows in got.values() for r in stage_rows]
        if ds == "coco":  # THRONE as published: window 1024 on COCO
            captions = {r["response"] for r in got["captions"]}
            evaluator = root / f"cache/{MODEL}_coco_throne_w1024_evaluator/evaluator.sqlite"
            rows += [r for r in _rows(evaluator, "image_hash IS NULL")
                     if json.loads(r["prompt"])["caption"] in captions]
    (out / "rows.json.gz").write_bytes(gzip.compress(json.dumps(rows, indent=0).encode(), mtime=0))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DATA)
    args = parser.parse_args(argv)
    status = repro.main(["--root", str(args.root), "--export-records", str(args.out / "records.json.gz"), "--check"])
    wiring(args.root, args.out / "wiring")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
