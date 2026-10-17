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

from metrics.base_metric import CAPTION_GEN, SAMPLES, BaseMetric, caption_images, object_names, pool_objects
from metrics.pope import QUESTIONS_PER_IMAGE
from predict.factory import REGISTRY as MODEL_REGISTRY

logger = logging.getLogger("mirage.run")
DS = {"coco": "coco", "visual_genome": "vg"}  # --dataset -> the key metrics use (POOL, CAPTION_PROMPTS)


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
    p.add_argument("--pope-questions", type=int, default=QUESTIONS_PER_IMAGE)
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
    if args.metric == "rope":
        from metrics.rope import ROPEMetric as metric_cls
    else:
        metric_cls = BaseMetric
    loader.load_data(max_samples=args.pool_size or metric_cls.POOL[DS[args.dataset]])
    return loader, object_names(loader)


def _run_dataset_metric(args):
    from predict.factory import build_predictor

    loader, all_objects = _load_dataset(args)
    if not len(loader):
        raise RuntimeError("No images were loaded. Check the annotation and image paths.")
    # --cache-only opens the prediction cache read-only: a replay never writes to it, and a miss raises.
    model = build_predictor(args.model, cache_only=args.cache_only)
    if not args.cache_only:
        model.load_model()
    try:
        results = _evaluate(args, model, loader, all_objects)
        results["cache_stats"] = dict(getattr(model, "stats", {}))
        return results
    finally:
        model.unload_model()


def _evaluate(args, model, loader, all_objects):
    # Each metric runs its own protocol (metric.run); run.py only picks the metric and its settings.
    if args.metric == "pope":
        from metrics.pope import POPEMetric

        metric = POPEMetric(mode="random", num_questions_per_image=args.pope_questions)
        metric.run(loader, model, n=args.samples)
        return metric.format_results()

    if args.metric == "rope":
        from metrics.rope import ROPEMetric

        metric = ROPEMetric(num_objects=5)
        metric.run(loader, model, n=args.samples)
        return metric.format_results()

    gen = {**CAPTION_GEN, "max_new_tokens": args.max_new_tokens}
    if args.metric == "chair":
        from metrics.chair import ChairMetric, vg_vocabulary

        # The VG vocabulary is built before generation (it needs NLTK data).
        metric = ChairMetric(use_coco_synonyms=True) if args.dataset == "coco" else ChairMetric(
            object_synsets={name: [name] for name in vg_vocabulary(all_objects)}, use_coco_synonyms=False)
        metric.run(loader, model, n=args.samples, gen=gen)
    elif args.metric == "throne":
        from metrics.throne import ThroneMetric

        # VG THRONE asks about the pool's object names, not all of objects.json.
        categories = all_objects if args.dataset == "coco" else pool_objects(loader)
        metric = ThroneMetric(evaluator_model_name=args.throne_evaluator, object_categories=categories,
                              distractor_ratio=args.throne_distractor_ratio,
                              evaluator_max_tokens=args.throne_evaluator_max_tokens)
        metric.run(loader, model, n=args.samples, gen=gen)
    elif args.metric == "faithscore":
        from metrics.faithscore import FaithScoreMetric

        if not args.llama_model:
            raise ValueError("FaithScore needs --llama-model or MIRAGE_LLAMA_MODEL_PATH.")
        from predict.factory import build_model

        scored, captions = caption_images(loader, model, args.samples, gen)
        image_paths = [str(s.get("image_path") or loader.images_path / s["image_filename"]) for s in scored]
        verifier = build_model("paligemma")
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
