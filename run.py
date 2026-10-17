"""Run Mirage hallucination metrics on COCO or Visual Genome.

    python run.py --dataset coco --metric pope --samples 10
    python run.py --dataset visual_genome --metric chair --samples 10

Datasets default to datasets/{coco,visual_genome}; model paths come from the
MIRAGE_*_MODEL_PATH variables.
"""

import argparse
import json
import logging
import os
from pathlib import Path

from PIL import Image

from metrics.base_metric import CAPTION_GEN, CAPTION_PROMPTS, SAMPLES, BaseMetric
from predict.factory import _REGISTRY as MODEL_REGISTRY

logger = logging.getLogger("mirage.run")


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=("coco", "visual_genome"), default="coco")
    p.add_argument("--metric", choices=("pope", "chair", "rope", "throne", "faithscore", "nope"), default="chair")
    p.add_argument("--model", choices=tuple(MODEL_REGISTRY), default=None)
    p.add_argument("--samples", type=int, default=SAMPLES, help="evaluated images")
    p.add_argument("--pool-size", type=int, default=None,
                   help="candidate images loaded before filtering (default 2000; 3000 for COCO ROPE)")
    p.add_argument("--annotations", type=Path, default=None)
    p.add_argument("--images", type=Path, default=None)
    p.add_argument("--split", default="val2017", help="COCO split")
    p.add_argument("--max-new-tokens", type=int, default=CAPTION_GEN["max_new_tokens"], help="caption length cap")
    p.add_argument("--pope-questions", type=int, default=6)
    p.add_argument("--output", type=Path, default=None, help="JSON results path")
    p.add_argument("--nope-turn2", type=Path, help="NOPE turn-two results JSON")
    p.add_argument("--nope-turn1", type=Path, help="NOPE turn-one results JSON")
    p.add_argument("--throne-evaluator", default=os.getenv("MIRAGE_THRONE_EVALUATOR_MODEL", "google/flan-t5-base"))
    p.add_argument("--throne-distractor-ratio", type=float, default=None,
                   help="distractors per ground-truth object (default: full vocabulary)")
    p.add_argument("--throne-evaluator-max-tokens", type=int, default=None,
                   help="evaluator input window; longer captions are trimmed (default 512)")
    p.add_argument("--cache-only", action="store_true", help="never load the model; fail on a cache miss")
    p.add_argument("--llama-model", default=os.getenv("MIRAGE_LLAMA_MODEL_PATH"))
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def _load_dataset(args):
    from dataloader.coco import COCODataLoader
    from dataloader.visual_genome import VisualGenomeDataLoader

    if args.dataset == "coco":
        root = Path("datasets/coco")
        loader = COCODataLoader(data_path=args.annotations or root / "annotations" / f"instances_{args.split}.json",
                                images_path=args.images or root / args.split, shuffle=False)
    else:
        root = Path("datasets/visual_genome")
        loader = VisualGenomeDataLoader(data_path=args.annotations or root / "objects.json",
                                        images_path=args.images or root / "VG_100K", shuffle=False)
    from metrics.rope import ROPEMetric

    pool = (ROPEMetric if args.metric == "rope" else BaseMetric).POOL[_key(args)]
    loader.load_data(max_samples=args.pool_size or pool)
    return loader, loader.get_category_names() if args.dataset == "coco" else loader.get_object_names()


def _key(args):
    return "coco" if args.dataset == "coco" else "vg"


def _ground_truth(sample):
    if "categories" in sample:
        return list(sample["categories"])
    return [obj.get("name", "") for obj in sample.get("objects", []) if obj.get("name")]


def _run_dataset_metric(args):
    from predict.factory import build_predictor

    loader, all_objects = _load_dataset(args)
    samples = list(loader)
    if not samples:
        raise RuntimeError("No images were loaded. Check the annotation and image paths.")
    model = build_predictor(args.model)
    if not args.cache_only:
        model.load_model()
    try:
        results = _evaluate(args, model, loader, samples, all_objects)
        results["cache_stats"] = dict(getattr(model, "stats", {}))
        return results
    finally:
        model.unload_model()


def _evaluate(args, model, loader, samples, all_objects):
    if args.metric == "pope":
        from dataloader.base import sample_for_image
        from metrics.pope import POPEMetric

        metric = POPEMetric(mode="random", num_questions_per_image=args.pope_questions)
        n_yes = int(metric.num_questions_per_image * metric.yes_ratio)
        responses, labels = [], []
        for sample in [s for s in samples if _ground_truth(s)][:args.samples]:
            objects, image = _ground_truth(sample), loader.load_image(sample)
            yes = sample_for_image(sorted(objects), min(n_yes, len(objects)), sample["image_id"], seed=42)
            questions = metric.generate_questions(objects, yes, all_objects)
            responses += [model.generate(image=image, prompt=q, **metric.GEN) for q, _ in questions]
            labels += questions
        metric.evaluate(responses, labels)
        return metric.format_results()

    if args.metric == "rope":
        from metrics.rope import ROPEMetric

        boxes = loader.extract_bboxes_for_rope(num_objects=5)
        metric = ROPEMetric(num_objects=5, class_list=all_objects)
        # The VG vocabulary is too large for one prompt: list the classes in the pool's boxes.
        classes = all_objects if args.dataset == "coco" else sorted(
            {b["name"] for image_boxes in boxes.values() for b in image_boxes if b["name"]})
        predictions, ground_truth = [], []
        for sample in [s for s in samples if str(s["image_id"]) in boxes][:args.samples]:
            image_boxes = boxes[str(sample["image_id"])]
            image = metric.draw_bounding_boxes(loader.load_image(sample), image_boxes)
            predictions.append([metric.parse_single_response(model.generate(
                image=image, prompt=metric.generate_prompt_for_box(i, class_list=classes),
                **metric.GEN)) for i in range(1, 6)])
            ground_truth.append([b["name"] for b in image_boxes])
        if not predictions:
            raise RuntimeError("No images had enough object boxes for ROPE.")
        metric.evaluate_predictions(predictions, ground_truth)
        return metric.format_results()

    if args.metric == "chair" and args.dataset == "visual_genome":
        from metrics.chair import vg_vocabulary

        vocabulary = vg_vocabulary(all_objects)  # before generation: needs NLTK data
    prompt = CAPTION_PROMPTS[_key(args)]
    captions, ground_truth, image_paths = [], [], []
    for sample in [s for s in samples if _ground_truth(s)][:args.samples]:
        captions.append(model.generate(image=loader.load_image(sample), prompt=prompt,
                                       **{**CAPTION_GEN, "max_new_tokens": args.max_new_tokens}))
        ground_truth.append(_ground_truth(sample))
        image_paths.append(str(sample.get("image_path") or loader.images_path / sample["image_filename"]))

    if args.metric == "chair":
        from metrics.chair import ChairMetric

        metric = ChairMetric(use_coco_synonyms=True) if args.dataset == "coco" else ChairMetric(
            object_synsets={name: [name] for name in vocabulary}, use_coco_synonyms=False)
        metric.evaluate(captions, ground_truth)
    elif args.metric == "throne":
        from metrics.throne import ThroneMetric

        # VG THRONE asks about the pool's object names, not all of objects.json.
        categories = all_objects if args.dataset == "coco" else sorted(
            {name for s in samples for name in _ground_truth(s)})
        metric = ThroneMetric(evaluator_model_name=args.throne_evaluator, object_categories=categories,
                              distractor_ratio=args.throne_distractor_ratio,
                              evaluator_max_tokens=args.throne_evaluator_max_tokens)
        metric.evaluate(captions, ground_truth)
    elif args.metric == "faithscore":
        from metrics.faithscore import FaithScoreMetric

        if not args.llama_model:
            raise ValueError("FaithScore needs --llama-model or MIRAGE_LLAMA_MODEL_PATH.")
        from models.paligemma import PaligemmaModel

        verifier = PaligemmaModel(model_name=os.getenv("MIRAGE_PALIGEMMA_MODEL_PATH", "google/paligemma-3b-mix-224"))
        verifier.load_model()
        try:
            metric = FaithScoreMetric(llama_path=args.llama_model, verification_model=verifier,
                                      caption_model_name=model.model_name)
            metric.evaluate(captions, image_paths)
        finally:
            verifier.unload_model()
    else:
        raise ValueError(f"Unsupported metric for image datasets: {args.metric}")
    return metric.format_results()


def _run_nope(args):
    from metrics.nope import NOPEMetric
    from predict.factory import build_predictor

    if not args.nope_turn2:
        raise ValueError("NOPE requires --nope-turn2 with generated Turn 2 model responses.")
    image_dir = args.images or Path("datasets/coco/val2017")
    metric = NOPEMetric(turn2_path=str(args.nope_turn2), image_dir=str(image_dir),
                        turn1_path=str(args.nope_turn1) if args.nope_turn1 else None)
    records = metric.records[:args.samples]
    model = build_predictor(args.model or "paligemma")
    model.load_model()
    outputs = []
    try:
        for record in records:
            image_id = record["image_id"]
            found = list(image_dir.glob(f"*{image_id.zfill(12)}*")) or list(image_dir.glob(f"*{image_id}*"))
            if not found:
                logger.warning("NOPE image %s is missing; scoring an empty response", image_id)
                outputs.append("")
                continue
            with Image.open(found[0]) as image:
                outputs.append(model.generate(image=image.convert("RGB"), prompt=record["question"],
                                              max_new_tokens=20, do_sample=False))
    finally:
        model.unload_model()
    metric.records = records
    metric.evaluate(outputs)
    return metric.format_results()


def main():
    args = _parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.samples < 1:
        raise SystemExit("--samples must be positive")
    results = _run_nope(args) if args.metric == "nope" else _run_dataset_metric(args)
    rendered = json.dumps(results, indent=2, default=str)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
