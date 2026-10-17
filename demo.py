"""Mirage on one page: load COCO and a model, then score POPE, ROPE and CHAIR on a few images.

    python demo.py

Needs datasets/coco and $MIRAGE_PALIGEMMA_MODEL_PATH; answers are cached in $MIRAGE_CACHE_DIR.
It takes no flags: run.py is the full CLI (python run.py --help).
"""
import sys

from dataloader import COCODataLoader
from metrics import ChairMetric, POPEMetric, ROPEMetric
from predict.factory import build_predictor

N = 5  # images per metric


def coco(metric):
    """COCO val2017 in file order; each metric says how many candidate images it draws from."""
    loader = COCODataLoader("datasets/coco/annotations/instances_val2017.json", "datasets/coco/val2017", shuffle=False)
    loader.load_data(max_samples=metric.POOL["coco"])
    return loader


def main():
    if sys.argv[1:]:
        raise SystemExit("demo.py takes no flags; use run.py (python run.py --help)")
    model = build_predictor("paligemma")  # any key in predict.factory._REGISTRY
    model.load_model()                    # lazy: weights load only on a cache miss

    pope = POPEMetric()                   # 6 yes/no questions per image, seed 42
    pope.run(coco(pope), model, n=N)
    print("POPE", pope.format_results()["pope_results"])

    rope = ROPEMetric()                   # 5 numbered boxes per image, one question per box
    rope.run(coco(rope), model, n=N)
    print("ROPE", rope.format_results())

    chair = ChairMetric(use_coco_synonyms=True)  # caption each image, then count hallucinated objects
    chair.run(coco(chair), model, n=N)
    print("CHAIR", chair.format_results()["summary"])
    model.unload_model()


if __name__ == "__main__":
    main()
