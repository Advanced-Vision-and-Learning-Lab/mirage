"""Mirage on one page: load COCO, a model, each metric and the judge, then score a few images.

Needs datasets/coco and $MIRAGE_PALIGEMMA_MODEL_PATH. Answers are cached in $MIRAGE_CACHE_DIR, verdicts come
from judgments/ (or $MIRAGE_JUDGMENTS_DIR). No flags: run.py is the full CLI (python run.py --help).
"""
import argparse

from dataloader import COCODataLoader
from judge import Judge
from metrics import ChairMetric, POPEMetric, ROPEMetric, ThroneMetric
from predict.factory import build_predictor

DS, N = "coco", 5  # images per metric


def coco(metric):  # COCO val2017 in file order; each metric says how many candidate images it draws from
    loader = COCODataLoader("datasets/coco/annotations/instances_val2017.json", "datasets/coco/val2017", shuffle=False)
    loader.load_data(max_samples=metric.POOL[DS])
    return loader


def main():
    argparse.ArgumentParser(description="Takes no flags; run.py is the full CLI.").parse_args()
    model = build_predictor("paligemma")  # any key in predict.factory.REGISTRY, or build_predictor(instance, key=...)
    model.load_model()                    # lazy: weights load only on a cache miss
    judge = Judge.load()                  # frozen verdicts; Judge.load(live=True) asks Claude about new items

    pope = POPEMetric()                   # 6 yes/no questions per image, seed 42
    records = pope.run(coco(pope), model, n=N)
    print("POPE", pope.score(records, judge("pope", records, ds=DS)))  # raw accuracy; head = Mod-POPE

    rope = ROPEMetric()                   # 5 numbered boxes per image, one question per box
    records = rope.run(coco(rope), model, n=N)
    print("ROPE", rope.score(records, judge("rope", records)))  # raw = strict exact match; head = judged

    loader = coco(ChairMetric)
    chair = ChairMetric.for_dataset(DS, loader)  # COCO synonyms (VG: names from the loader)
    images = chair.run(loader, model, n=N)       # caption each image, tag its object mentions
    refs = chair.reference_objects("datasets/coco/annotations/captions_val2017.json", images)
    scores = chair.score(images, judge("mentions", images, refs=refs), refs, judge.committee(DS, images))
    print("CHAIR", scores[model.model_key])      # Mod-CHAIR F0.5; coverage pooled with the 9 judged models

    throne = ThroneMetric.for_dataset(DS, loader)  # FLAN-T5-XL evaluator, ratio 38, window 1024
    throne.run(loader, model, n=N)                 # the same cached captions, then the evaluator
    print("THRONE", throne.score(throne.format_results()["per_image_details"]))  # class-wise F0.5
    model.unload_model()


if __name__ == "__main__":
    main()
