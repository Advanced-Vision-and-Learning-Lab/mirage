"""Mirage on one page: load COCO and a model, then score POPE, ROPE and CHAIR on a few images.

Needs datasets/coco and $MIRAGE_PALIGEMMA_MODEL_PATH; answers are cached in $MIRAGE_CACHE_DIR.
"""
from dataloader import COCODataLoader
from dataloader.base import sample_for_image
from metrics import ChairMetric, POPEMetric, ROPEMetric
from metrics.base_metric import CAPTION_GEN, CAPTION_PROMPTS
from predict.factory import build_predictor
N = 5  # images per metric; run.py is the full CLI (python run.py --help)


def coco(metric):  # COCO val2017 in file order; each metric says how many candidate images it draws from
    loader = COCODataLoader("datasets/coco/annotations/instances_val2017.json", "datasets/coco/val2017")
    loader.load_data(max_samples=metric.POOL["coco"])
    return loader, [s for s in loader if s.get("categories")][:N]


def main():
    model = build_predictor("paligemma")  # any key in predict.factory._REGISTRY
    model.load_model()                    # lazy: weights load only on a cache miss
    pope, (loader, samples) = POPEMetric(), coco(POPEMetric)  # 6 yes/no questions per image, seed 42
    n_yes, responses, questions = int(pope.num_questions_per_image * pope.yes_ratio), [], []
    for s in samples:
        objects, image = list(s["categories"]), loader.load_image(s)
        yes = sample_for_image(sorted(objects), min(n_yes, len(objects)), s["image_id"], seed=42)
        asked = pope.generate_questions(objects, yes, loader.get_category_names())
        responses += [model.generate(image=image, prompt=q, **pope.GEN) for q, _ in asked]
        questions += asked
    pope.evaluate(responses, questions)
    print("POPE", pope.format_results()["pope_results"])
    rope, (loader, _) = ROPEMetric(), coco(ROPEMetric)  # 5 numbered boxes per image, one question per box
    boxes, predictions, truths = loader.extract_bboxes_for_rope(num_objects=5), [], []
    for s in [s for s in loader if str(s["image_id"]) in boxes][:N]:
        image = rope.draw_bounding_boxes(loader.load_image(s), boxes[str(s["image_id"])])
        predictions.append([rope.parse_single_response(model.generate(
            image=image, prompt=rope.generate_prompt_for_box(i), **rope.GEN)) for i in range(1, 6)])
        truths.append([b["name"] for b in boxes[str(s["image_id"])]])
    rope.evaluate_predictions(predictions, truths)
    print("ROPE", rope.format_results())
    chair, (loader, samples) = ChairMetric(use_coco_synonyms=True), coco(ChairMetric)  # captions, then CHAIR
    captions = [model.generate(image=loader.load_image(s), prompt=CAPTION_PROMPTS["coco"], **CAPTION_GEN)
                for s in samples]
    chair.evaluate(captions, [list(s["categories"]) for s in samples])
    print("CHAIR", chair.format_results()["summary"])
    model.unload_model()


if __name__ == "__main__":
    main()
