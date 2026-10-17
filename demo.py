"""Mirage on one page: load COCO, a model, each metric and the judge, then score a few images.
Needs datasets/coco and $MIRAGE_PALIGEMMA_MODEL_PATH; answers cache in $MIRAGE_CACHE_DIR, verdicts in judgments/."""
import argparse

from dataloader import COCODataLoader
from judge import Judge, MissingVerdicts
from metrics import ChairMetric, POPEMetric, ROPEMetric, ThroneMetric
from predict.factory import build_predictor

DS, N = "coco", 5  # images per metric


def coco(metric):  # COCO val2017 in file order; each metric says how many candidate images it draws from
    loader = COCODataLoader("datasets/coco/annotations/instances_val2017.json", "datasets/coco/val2017")
    loader.load_data(max_samples=metric.POOL[DS])
    return loader


def show(name, score):  # a judged score needs a stored verdict for every item, else it raises MissingVerdicts
    try:
        print(name, score())
    except MissingVerdicts as e:
        print(name, "has no judged score yet:", e)


def main():
    argparse.ArgumentParser(description="Takes no flags; run.py is the full CLI.").parse_args()
    model = build_predictor("paligemma")  # any key in predict.factory.REGISTRY, or build_predictor(instance, key=...)
    model.load_model()                    # lazy: weights load only on a cache miss
    judge = Judge.load()                  # frozen verdicts (live=True grades new mentions, given image_paths)

    pope = POPEMetric()                   # 6 yes/no questions per image, seed 42
    records = pope.run(coco(pope), model, n=N)
    show("POPE", lambda: pope.score(records, judge("pope", records, ds=DS)))  # raw accuracy; head = Mod-POPE
    rope = ROPEMetric()                   # 5 numbered boxes per image, one question per box
    records = rope.run(coco(rope), model, n=N)
    show("ROPE", lambda: rope.score(records, judge("rope", records)))  # raw = strict exact match; head = judged

    loader = coco(ChairMetric)
    chair = ChairMetric.for_dataset(DS, loader)  # COCO synonyms; coverage pooled with the 9 judged models
    images = chair.run(loader, model, n=N)       # caption each image, tag its object mentions
    refs = chair.reference_objects("datasets/coco/annotations/captions_val2017.json", images)  # count as present
    show("CHAIR", lambda: chair.score(images, judge("mentions", images, refs=refs), refs, judge.committee(DS, images)))
    throne = ThroneMetric.for_dataset(DS, loader)  # FLAN-T5-XL evaluator, ratio 38, window 1024; same cached captions
    print("THRONE", throne.score(throne.run(loader, model, n=N)))  # class-wise F0.5
    model.unload_model()


if __name__ == "__main__":
    main()
