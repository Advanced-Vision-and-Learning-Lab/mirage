"""Prepare, run, and compare an alternate judge on Mirage's frozen model outputs.

Typical flow::

    python -m judge.ablation prepare --root /path/to/mirage-data \
        --records tests/data/records.json.gz --out runs/ablate-llms
    python -m judge.ablation judge --backend codex --model gpt-6.1-sol --run runs/ablate-llms
    python -m judge.ablation score --backend codex --model gpt-6.1-sol \
        --records tests/data/records.json.gz --run runs/ablate-llms

The prepare step reuses Mirage's parser and item keys. Both judges see the same item list and local
images. The score step leaves ``judgments/`` untouched and writes the alternate store and ranking
comparison under the selected run directory.
"""

import argparse
import csv
import gzip
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

from .backend import expected_keys, export_rows, skill_sha
from .store import VerdictStore, sha256


def _read_records(path: Path):
    import repro

    return repro.read_records(path)


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.") or "judge"


def _write_chunks(directory: Path, requests, size: int):
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for start in range(0, len(requests), size):
        path = directory / f"in_{start // size:04d}.json"
        path.write_text(json.dumps(requests[start:start + size], ensure_ascii=False, indent=1) + "\n",
                        encoding="utf-8")
        paths.append(path)
    return paths


def _samples_by_id(loader):
    return {str(sample["image_id"]): sample for sample in loader}


def prepare(root: Path, records_path: Path, out: Path, chunk_size: int = 10):
    """Make matched mention, ROPE-answer, and POPE-no request chunks from records and local images."""
    import repro
    from dataloader.coco import COCODataLoader
    from dataloader.visual_genome import VisualGenomeDataLoader
    from judge.judge import mention_requests
    from metrics.chair import ChairMetric
    from metrics.rope import ROPEMetric, judged_answer

    if chunk_size < 1:
        raise ValueError("chunk size must be positive")
    records = _read_records(records_path)
    opus_store = VerdictStore.open()
    opus_mentions = opus_store.verdicts("mentions")
    opus_rope = opus_store.verdicts("rope")
    counts = {}
    source_hash = hashlib.sha256(records_path.read_bytes()).hexdigest()

    for ds in repro.DATASETS:
        if ds == "vg":
            loader = VisualGenomeDataLoader(root / "datasets/visual_genome/objects.json",
                                            root / "datasets/visual_genome/VG_100K")
            names = records[ds]["object_names"]
        else:
            loader = COCODataLoader(root / "datasets/coco/annotations/instances_val2017.json",
                                    root / "datasets/coco/val2017")
            names = ()
        loader.load_data(max_samples=3000 if ds == "coco" else 2000)
        sample_map = _samples_by_id(loader)
        rec = records[ds]

        # Caption mentions are extracted once by Mirage; alternate judges only label that fixed list.
        chair = ChairMetric.for_dataset(ds, SimpleNamespace(get_object_names=lambda: names))
        image_rows = [{"item": image["item"], "gt": image["gt"], "models": {
            model: {"caption": caption, "mentions": chair.tag_mentions(caption, image["gt"])}
            for model, caption in image["captions"].items()}}
            for image in rec["images"]]
        refs = chair.reference_objects({"annotations": rec["references"]}, image_rows) if ds == "coco" else {}
        mention_paths = {image["item"]: sample_map[image["item"].split("_", 1)[1]]["image_path"]
                         for image in image_rows}
        mentions_all = mention_requests(image_rows, {}, refs, mention_paths)
        current_mention_keys = {
            (request["image_id"], key)
            for request in mentions_all
            for key in expected_keys([request])[request["image_id"]]
        }
        expected_mention_keys = {key for key in opus_mentions if key[0].startswith(f"{ds}_")}
        unscored_keys = current_mention_keys - expected_mention_keys
        if unscored_keys:
            raise ValueError(f"{ds}: current Mirage mention candidates are missing frozen Opus verdicts; "
                             f"examples={list(unscored_keys)[:5]}")
        mentions = []
        for request in mentions_all:
            filtered = [mention for mention in request["mentions"]
                        if (request["image_id"], f'{mention["model"]}|{mention["word"]}') in opus_mentions]
            if filtered:
                mentions.append({**request, "mentions": filtered})
        mention_keys = {(request["image_id"], key)
                        for request in mentions for key in expected_keys([request])[request["image_id"]]}
        if mention_keys != current_mention_keys:
            raise ValueError(f"{ds}: candidate filtering changed the current Mirage judge inputs")
        mention_inputs = _write_chunks(out / "inputs/mentions" / ds, mentions, chunk_size)

        # POPE asks the same sampled negative questions for every model; judge each image/object once.
        pope_by_image = defaultdict(set)
        for model_rows in rec["pope"].values():
            for image_id, obj, label, _reply in model_rows:
                if not label:
                    pope_by_image[str(image_id)].add(obj)
        pope_requests = [{"image_id": image_id,
                          "image": sample_map[image_id]["image_path"],
                          "objects": sorted(objects)}
                         for image_id, objects in sorted(pope_by_image.items())]
        pope_inputs = _write_chunks(out / "inputs/pope" / ds, pope_requests, chunk_size)

        # ROPE answers are parsed by Mirage. Dedupe identical (image, box, answer) keys across models,
        # then recreate the exact numbered-box image that the judge must inspect.
        rope = ROPEMetric(num_objects=5)
        boxes = loader.extract_bboxes_for_rope(num_objects=5)
        answers = defaultdict(lambda: defaultdict(dict))
        for model_rows in rec["rope"].values():
            for item, box, _gt, response in model_rows:
                pred = rope.parse_single_response(response)
                key = judged_answer(response, pred)
                answers[item][box].setdefault(key, response)
        rope_requests = []
        image_dir = out / "images/rope"
        image_dir.mkdir(parents=True, exist_ok=True)
        for item, by_box in sorted(answers.items()):
            by_box = {box: {answer: example for answer, example in by_box[box].items()
                            if (item, f"{box}|{answer}") in opus_rope}
                      for box in by_box}
            by_box = {box: values for box, values in by_box.items() if values}
            if not by_box:
                continue
            image_id = item.split("_", 1)[1]
            if image_id not in sample_map or image_id not in boxes:
                raise ValueError(f"{item}: missing source image or ROPE boxes")
            annotated = rope.draw_bounding_boxes(loader.load_image(sample_map[image_id]), boxes[image_id])
            image_path = image_dir / f"{item}.jpg"
            annotated.save(image_path, format="JPEG", quality=95)
            rope_requests.append({"image_id": item, "image": str(image_path.resolve()),
                                  "boxes": [{"box": box, "answers": [
                                      {"answer": answer, "example": example}
                                      for answer, example in sorted(by_box[box].items())]}
                                      for box in sorted(by_box)]})
        rope_inputs = _write_chunks(out / "inputs/rope" / ds, rope_requests, chunk_size)

        rope_keys = {(request["image_id"], f"{box_row['box']}|{answer_row['answer']}")
                     for request in rope_requests for box_row in request["boxes"]
                     for answer_row in box_row["answers"]}
        expected_rope_keys = {key for key in opus_rope if key[0].startswith(f"{ds}_")}
        if rope_keys != expected_rope_keys:
            missing, extra = expected_rope_keys - rope_keys, rope_keys - expected_rope_keys
            raise ValueError(f"{ds}: ROPE judge candidates differ from frozen Opus keys; "
                             f"missing={list(missing)[:5]}, extra={list(extra)[:5]}")

        counts[ds] = {
            "mentions": {"images": len(mentions), "keys": sum(len(x["mentions"]) for x in mentions),
                         "chunks": len(mention_inputs)},
            "pope": {"images": len(pope_requests), "keys": sum(len(x["objects"]) for x in pope_requests),
                     "chunks": len(pope_inputs)},
            "rope": {"images": len(rope_requests),
                     "keys": sum(len(b["answers"]) for x in rope_requests for b in x["boxes"]),
                     "chunks": len(rope_inputs)},
        }

    image_hashes = {}
    for task in ("mentions", "pope", "rope"):
        for ds in repro.DATASETS:
            for input_path in (out / "inputs" / task / ds).glob("in_*.json"):
                for request in json.loads(input_path.read_text(encoding="utf-8")):
                    image_path = Path(request["image"])
                    key = request["image_id"]
                    image_hashes[f"{task}:{ds}:{key}"] = sha256(image_path)
    manifest = {"records": str(records_path.resolve()), "records_sha256": source_hash,
                "root": str(root.resolve()), "chunk_size": chunk_size, "counts": counts,
                "image_sha256": image_hashes,
                "rubrics": {name: skill_sha(name) for name in ("judge-mentions", "judge-rope", "judge-pope")}}
    out.mkdir(parents=True, exist_ok=True)
    (out / "prepared.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest


def run_judge(run_dir: Path, backend_name: str, model=None):
    """Run all prepared chunks with Claude (default comparison) or Codex."""
    from .claude import ClaudeCLI
    from .codex import CodexCLI

    if backend_name == "claude":
        backend = ClaudeCLI(model=model)
    elif backend_name == "codex":
        backend = CodexCLI(model=model)
    else:
        raise ValueError(f"unsupported CLI judge backend: {backend_name}")
    slug = _slug(model or backend.model)
    result_root = run_dir / "outputs" / backend_name / slug
    for skill in ("mentions", "rope", "pope"):
        for ds in ("coco", "vg"):
            inputs = sorted((run_dir / "inputs" / skill / ds).glob("in_*.json"))
            if inputs:
                backend.run_files(inputs, result_root / skill / ds, f"judge-{skill}")
    return result_root


def _output_rows(result_root: Path, run_dir: Path, skill: str, ds: str):
    input_files = sorted((run_dir / "inputs" / skill / ds).glob("in_*.json"))
    return export_rows(input_files, result_root / skill / ds)


def _write_tsv_gz(path: Path, rows):
    import io

    stream = io.StringIO()
    writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
    writer.writerows(rows)
    path.write_bytes(gzip.compress(stream.getvalue().encode("utf-8"), mtime=0))


def _headline_deltas(opus: dict, alternate: dict):
    rows = []
    correlations = {}
    metrics = ("POPE", "ROPE", "CHAIR", "THRONE")
    for ds in opus:
        for metric in metrics:
            opus_order = {m: opus[ds][m][metric]["rank_head"] for m in opus[ds]}
            alt_order = {m: alternate[ds][m][metric]["rank_head"] for m in alternate[ds]}
            n = len(opus_order)
            sum_sq = sum((opus_order[m] - alt_order[m]) ** 2 for m in opus_order)
            correlations[f"{ds}.{metric}"] = 1 - 6 * sum_sq / (n * (n * n - 1)) if n > 1 else 1.0
            for model in opus_order:
                before, after = opus[ds][model][metric], alternate[ds][model][metric]
                rows.append({"dataset": ds, "metric": metric, "model": model,
                             "opus_head": before["head"], "alternate_head": after["head"],
                             "score_delta": after["head"] - before["head"],
                             "opus_rank": before["rank_head"], "alternate_rank": after["rank_head"],
                             "rank_delta": after["rank_head"] - before["rank_head"]})
    return rows, correlations


def score(run_dir: Path, records_path: Path, backend_name: str, model: str):
    """Build a separate alternate store, recompute all scores/ranks, and compare to frozen Opus."""
    import repro
    from .backend import SKILLS

    records = _read_records(records_path)
    prepared_path = run_dir / "prepared.json"
    if not prepared_path.is_file():
        raise FileNotFoundError(f"run is not prepared: {prepared_path}")
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    if prepared["records_sha256"] != hashlib.sha256(records_path.read_bytes()).hexdigest():
        raise ValueError("score records do not match the records used to prepare judge inputs")
    slug = _slug(model)
    result_root = run_dir / "outputs" / backend_name / slug
    alt_dir = run_dir / "judgments" / backend_name / slug
    alt_dir.mkdir(parents=True, exist_ok=True)
    mentions, rope_rows = [], []
    pope_rows = []

    for ds in repro.DATASETS:
        mentions.extend(_output_rows(result_root, run_dir, "mentions", ds))
        rope_rows.extend(_output_rows(result_root, run_dir, "rope", ds))
        for image, key, verdict in _output_rows(result_root, run_dir, "pope", ds):
            if verdict in ("present", "unsure"):
                pope_rows.append({"dataset": ds, "image_id": image, "object": key, "reason": verdict})

    # Preserve Opus's explicit junk-name exclusions as a shared POPE preprocessing decision;
    # the alternate judge changes only the image-grounded present/absent/unsure calls.
    opus_store = VerdictStore.open()
    for ds in repro.DATASETS:
        for (image, obj), reason in opus_store.drop_list(ds).items():
            if reason == "junk":
                pope_rows.append({"dataset": ds, "image_id": image, "object": obj, "reason": "junk"})
    _write_tsv_gz(alt_dir / "mentions.tsv.gz", mentions)
    _write_tsv_gz(alt_dir / "rope.tsv.gz", rope_rows)
    with (alt_dir / "pope_drop_list.tsv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("dataset", "image_id", "object", "reason"),
                                delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(pope_rows)
    (alt_dir / "committee.json").write_text("{}\n", encoding="utf-8")

    alternate_store = VerdictStore(alt_dir)
    alternate, committee = repro.rebuild(records, alternate_store)
    (alt_dir / "committee.json").write_text(json.dumps(committee, indent=1) + "\n", encoding="utf-8")
    files = {name: sha256(alt_dir / name) for name in
             ("mentions.tsv.gz", "rope.tsv.gz", "pope_drop_list.tsv", "committee.json")}
    rubric_hashes = {skill: skill_sha(skill) for skill in SKILLS}
    (alt_dir / "manifest.json").write_text(json.dumps({"files": files, "judge_model": model,
                                                       "judge_backend": backend_name,
                                                       "skills": rubric_hashes}, indent=1) + "\n",
                                           encoding="utf-8")

    opus, _ = repro.rebuild(records, opus_store)
    deltas, rho = _headline_deltas(opus, alternate)
    (run_dir / "scores").mkdir(parents=True, exist_ok=True)
    (run_dir / "scores" / f"{backend_name}-{slug}.json").write_text(
        json.dumps({"table": alternate, "models": repro.MODELS}, indent=1) + "\n", encoding="utf-8")
    (run_dir / "scores" / f"{backend_name}-{slug}-rank-deltas.json").write_text(
        json.dumps({"rows": deltas, "spearman_rank_correlation": rho}, indent=1) + "\n", encoding="utf-8")
    with (run_dir / "scores" / f"{backend_name}-{slug}-rank-deltas.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(deltas[0]))
        writer.writeheader()
        writer.writerows(deltas)
    return deltas, rho, alt_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="prepare matched item chunks and image paths")
    prep.add_argument("--root", type=Path, required=True, help="dataset tree containing datasets/coco and datasets/visual_genome")
    prep.add_argument("--records", type=Path, required=True, help="records exported from the reference results")
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--chunk-size", type=int, default=10)
    run = commands.add_parser("judge", help="run all prepared chunks")
    run.add_argument("--backend", choices=("claude", "codex"), required=True)
    run.add_argument("--model")
    run.add_argument("--run", type=Path, required=True)
    rescore = commands.add_parser("score", help="score the alternate judgments and write rank deltas")
    rescore.add_argument("--backend", choices=("claude", "codex"), required=True)
    rescore.add_argument("--model", required=True)
    rescore.add_argument("--records", type=Path, required=True)
    rescore.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        manifest = prepare(args.root, args.records, args.out, args.chunk_size)
        print(json.dumps(manifest["counts"], indent=1))
    elif args.command == "judge":
        print(run_judge(args.run, args.backend, args.model))
    else:
        deltas, correlations, alt_dir = score(args.run, args.records, args.backend, args.model)
        print(f"alternate verdict store: {alt_dir}")
        print(f"headline rank correlations: {json.dumps(correlations, sort_keys=True)}")
        print(f"rank-delta rows: {len(deltas)}")


if __name__ == "__main__":
    main()
