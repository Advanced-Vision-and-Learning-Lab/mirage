"""Run Mirage hallucination metrics on COCO or Visual Genome.

Example:
    python demo.py --dataset coco --metric pope --samples 10
    python demo.py --dataset visual_genome --metric chair --samples 10

Dataset files and model weights are expected to be available locally. Paths can
be set with --annotations, --images, and the MIRAGE_*_MODEL_PATH variables.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image

from predict.factory import _REGISTRY as MODEL_REGISTRY

logger = logging.getLogger("mirage.demo")
CAPTION_PROMPT = (
    "Describe this image in detail. List every object you can see, their colors, "
    "positions, and what is happening in the scene. Be thorough and specific. "
    "Description: "
)
VG_CAPTION_PROMPT = (
    "You are tasked with generating a detailed caption for the given image. The caption "
    "should include all the objects present in the image. The objects should be described "
    "in a natural and coherent way. The caption should be written in a way that is easy to "
    "understand and should be free of any grammatical errors and at least 10 words long. "
    "Caption: "
)
# Visual Genome object strings that are not objects (stopwords are added from NLTK).
# Without this filter CHAIR counts every caption containing 'the' or 'red' as a mention.
VG_NON_OBJECTS = {
    "red", "white", "black", "blue", "green", "yellow", "brown", "gray",
    "grey", "pink", "purple", "orange", "gold", "silver", "tan", "beige",
    "large", "small", "wooden", "metal", "plastic", "old", "new", "tall",
    "short", "long", "wide", "narrow", "thick", "thin", "open", "closed",
    "cozy", "sunny", "shiny", "rough", "smooth", "clean", "dirty",
    "background", "foreground", "left", "right", "top", "bottom",
    "middle", "center", "side", "corner", "edge",
    "parked", "standing", "sitting", "walking", "lying", "running",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("coco", "visual_genome"), default="coco")
    parser.add_argument(
        "--metric",
        choices=("pope", "chair", "rope", "throne", "faithscore", "nope"),
        default="chair",
    )
    parser.add_argument("--model", choices=tuple(MODEL_REGISTRY), default=None)
    parser.add_argument("--samples", type=int, default=5, help="Number of evaluated images")
    parser.add_argument(
        "--pool-size", type=int, default=None,
        help="Candidate images loaded before filtering; also fixes the VG THRONE categories "
             "and ROPE prompt classes (default 2000, or 3000 for COCO ROPE)",
    )
    parser.add_argument("--annotations", type=Path, default=None)
    parser.add_argument("--images", type=Path, default=None)
    parser.add_argument("--split", default="val2017", help="COCO split; ignored for Visual Genome")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Caption length cap")
    parser.add_argument("--pope-questions", type=int, default=6)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON results path")
    parser.add_argument("--nope-turn2", type=Path, help="NOPE turn-two results JSON (metric=nope)")
    parser.add_argument("--nope-turn1", type=Path, help="Optional NOPE turn-one results JSON")
    parser.add_argument(
        "--throne-evaluator",
        default=os.getenv("MIRAGE_THRONE_EVALUATOR_MODEL", "google/flan-t5-base"),
        help="THRONE evaluator model id or local path",
    )
    parser.add_argument(
        "--throne-distractor-ratio", type=float, default=None,
        help="Ask about ground-truth objects plus this many distractors each "
             "(default: $MIRAGE_THRONE_DISTRACTOR_RATIO, else the full vocabulary)",
    )
    parser.add_argument(
        "--throne-evaluator-max-tokens", type=int, default=None,
        help="THRONE evaluator input window in tokens; longer captions are trimmed "
             "from the end (default: $MIRAGE_THRONE_EVALUATOR_MAX_TOKENS, else 512)",
    )
    parser.add_argument(
        "--cache-only", action="store_true",
        help="Never load the model; fail on the first prediction-cache miss",
    )
    parser.add_argument("--llama-model", default=os.getenv("MIRAGE_LLAMA_MODEL_PATH"))
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _dataset_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    if args.dataset == "coco":
        root = Path("datasets/coco")
        annotations = args.annotations or root / "annotations" / f"instances_{args.split}.json"
        images = args.images or root / args.split
    else:
        root = Path("datasets/visual_genome")
        annotations = args.annotations or root / "objects.json"
        images = args.images or root / "VG_100K"
    return annotations, images


def _load_dataset(args: argparse.Namespace):
    # Submodule imports keep working when tests stub the dataloader package.
    from dataloader.coco import COCODataLoader
    from dataloader.visual_genome import VisualGenomeDataLoader

    annotations, images = _dataset_paths(args)
    loader_type = COCODataLoader if args.dataset == "coco" else VisualGenomeDataLoader
    loader = loader_type(data_path=annotations, images_path=images, shuffle=False)
    pool_size = args.pool_size or (3000 if args.dataset == "coco" and args.metric == "rope" else 2000)
    loader.load_data(max_samples=pool_size)
    if args.dataset == "coco":
        all_objects = loader.get_category_names()
    else:
        all_objects = loader.get_object_names()
    return loader, all_objects


def _ground_truth(loader, sample: Dict[str, Any]) -> List[str]:
    if "categories" in sample:
        return list(sample["categories"])
    return [obj.get("name", "") for obj in sample.get("objects", []) if obj.get("name")]


def _run_dataset_metric(args: argparse.Namespace) -> Dict[str, Any]:
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


def _vg_chair_vocabulary(all_objects: List[str]) -> List[str]:
    from nltk.corpus import stopwords

    bad = set(stopwords.words("english")) | VG_NON_OBJECTS
    return [name for name in all_objects if name.lower().strip() not in bad and len(name.strip()) >= 2]


def _evaluate(args: argparse.Namespace, model, loader, samples, all_objects) -> Dict[str, Any]:
    if args.metric == "pope":
        from dataloader.base import sample_for_image
        from metrics.pope import POPEMetric

        metric = POPEMetric(mode="random", num_questions_per_image=args.pope_questions)
        responses: List[str] = []
        question_labels = []
        evaluated = 0
        for sample in samples:
            objects = _ground_truth(loader, sample)
            if not objects:
                continue
            image = loader.load_image(sample)
            n_yes = int(metric.num_questions_per_image * metric.yes_ratio)
            yes_objects = sample_for_image(sorted(objects), min(n_yes, len(objects)), sample["image_id"], seed=42)
            questions = metric.generate_questions(objects, yes_objects, all_objects)
            for question, _ in questions:
                responses.append(model.generate(image=image, prompt=question, max_new_tokens=10, do_sample=False))
            question_labels.extend(questions)
            evaluated += 1
            if evaluated >= args.samples:
                break
        metric.evaluate(responses, question_labels)
        return metric.format_results()

    if args.metric == "rope":
        from metrics.rope import ROPEMetric

        num_objects = 5
        boxes = loader.extract_bboxes_for_rope(num_objects=num_objects)
        metric = ROPEMetric(num_objects=num_objects, class_list=all_objects)
        # The VG vocabulary is too large for one prompt, so VG prompts list only
        # the classes present in the pool's ROPE boxes.
        prompt_classes = all_objects
        if args.dataset == "visual_genome":
            prompt_classes = sorted({box["name"] for image_boxes in boxes.values()
                                     for box in image_boxes if box["name"]})
        predictions, ground_truth = [], []
        for sample in samples:
            image_id = str(sample["image_id"])
            if image_id not in boxes:
                continue
            gt = [box["name"] for box in boxes[image_id]]
            annotated = metric.draw_bounding_boxes(loader.load_image(sample), boxes[image_id])
            predicted = []
            for index in range(1, num_objects + 1):
                prompt = metric.generate_prompt_for_box(index, class_list=prompt_classes)
                response = model.generate(image=annotated, prompt=prompt, max_new_tokens=30, do_sample=False)
                predicted.append(metric.parse_single_response(response))
            predictions.append(predicted)
            ground_truth.append(gt)
            if len(predictions) >= args.samples:
                break
        if not predictions:
            raise RuntimeError("No images had enough object boxes for ROPE.")
        metric.evaluate_predictions(predictions, ground_truth)
        return metric.format_results()

    captions: List[str] = []
    ground_truth: List[List[str]] = []
    image_paths: List[str] = []
    caption_prompt = CAPTION_PROMPT if args.dataset == "coco" else VG_CAPTION_PROMPT
    if args.metric == "chair" and args.dataset == "visual_genome":
        chair_vocabulary = _vg_chair_vocabulary(all_objects)  # before generation: needs NLTK data
    for sample in samples:
        objects = _ground_truth(loader, sample)
        if not objects:
            continue
        captions.append(model.generate(
            image=loader.load_image(sample), prompt=caption_prompt,
            max_new_tokens=args.max_new_tokens, do_sample=False,
        ))
        ground_truth.append(objects)
        if sample.get("image_path"):
            image_paths.append(str(sample["image_path"]))
        else:
            image_paths.append(str(loader.images_path / sample["image_filename"]))
        if len(captions) >= args.samples:
            break

    if args.metric == "chair":
        from metrics.chair import ChairMetric
        if args.dataset == "coco":
            metric = ChairMetric(use_coco_synonyms=True)
        else:
            metric = ChairMetric(
                object_synsets={name: [name] for name in chair_vocabulary},
                use_coco_synonyms=False,
            )
        metric.evaluate(captions, ground_truth)
    elif args.metric == "throne":
        from metrics.throne import ThroneMetric
        categories = all_objects
        if args.dataset == "visual_genome":
            # VG THRONE asks about the pool's object names, not all of objects.json.
            categories = sorted({name for sample in samples for name in _ground_truth(loader, sample)})
        metric = ThroneMetric(
            evaluator_model_name=args.throne_evaluator,
            object_categories=categories,
            distractor_ratio=args.throne_distractor_ratio,
            evaluator_max_tokens=args.throne_evaluator_max_tokens,
        )
        metric.evaluate(captions, ground_truth)
    elif args.metric == "faithscore":
        from metrics.faithscore import FaithScoreMetric
        if not args.llama_model:
            raise ValueError("FaithScore needs --llama-model or MIRAGE_LLAMA_MODEL_PATH.")
        from models.paligemma import PaligemmaModel
        verifier = PaligemmaModel(model_name=os.getenv("MIRAGE_PALIGEMMA_MODEL_PATH", "google/paligemma-3b-mix-224"))
        verifier.load_model()
        try:
            metric = FaithScoreMetric(
                llama_path=args.llama_model,
                verification_model=verifier,
                caption_model_name=model.model_name,
            )
            metric.evaluate(captions, image_paths)
        finally:
            verifier.unload_model()
    else:
        raise ValueError(f"Unsupported metric for image datasets: {args.metric}")
    return metric.format_results()


def _run_nope(args: argparse.Namespace) -> Dict[str, Any]:
    from metrics.nope import NOPEMetric
    from predict.factory import build_predictor

    if not args.nope_turn2:
        raise ValueError("NOPE requires --nope-turn2 with generated Turn 2 model responses.")
    if args.model is None:
        args.model = "paligemma"
    image_dir = args.images or Path("datasets/coco/val2017")
    metric = NOPEMetric(
        turn2_path=str(args.nope_turn2),
        turn1_path=str(args.nope_turn1) if args.nope_turn1 else None,
        image_dir=str(image_dir),
    )
    records = metric.records[:args.samples]
    model = build_predictor(args.model)
    model.load_model()
    try:
        outputs = []
        for record in records:
            candidates = list(image_dir.glob(f"*{record['image_id'].zfill(12)}*"))
            if not candidates:
                candidates = list(image_dir.glob(f"*{record['image_id']}*"))
            if not candidates:
                logger.warning("NOPE image %s is missing; scoring an empty response", record["image_id"])
                outputs.append("")
                continue
            with Image.open(candidates[0]) as image:
                outputs.append(model.generate(
                    image=image.convert("RGB"), prompt=record["question"],
                    max_new_tokens=20, do_sample=False,
                ))
    finally:
        model.unload_model()
    metric.records = records
    metric.evaluate(outputs)
    return metric.format_results()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
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
