from types import MappingProxyType
from typing import Dict, List, Any, Mapping, Tuple

SAMPLES = 500  # images scored per cell: run.py --samples default
BETA2 = 0.25  # F-beta weight: beta = 0.5 favours precision
MIN_WORDS = 5  # captions with fewer words name no objects in the judged scores
# Caption prompts by dataset key; their bytes are part of the prediction cache key.
# The protocol constants below are read-only: changing one in place would change cache keys.
CAPTION_PROMPTS = MappingProxyType({
    "coco": (
        "Describe this image in detail. List every object you can see, their colors, "
        "positions, and what is happening in the scene. Be thorough and specific. "
        "Description: "
    ),
    "vg": (
        "You are tasked with generating a detailed caption for the given image. The caption "
        "should include all the objects present in the image. The objects should be described "
        "in a natural and coherent way. The caption should be written in a way that is easy to "
        "understand and should be free of any grammatical errors and at least 10 words long. "
        "Caption: "
    ),
})
CAPTION_GEN = MappingProxyType({"max_new_tokens": 512, "do_sample": False})


def fbeta(p: float, r: float, beta2: float = BETA2) -> float:
    return (1 + beta2) * p * r / (beta2 * p + r) if p + r else 0.0


class MissingVerdicts(RuntimeError):
    """Items with no stored verdict; grade them with the judge (judge/claude.py) and add the rows."""

    def __init__(self, kind: str, items: List[Tuple[str, str]]):
        super().__init__(f"{len(items)} {kind} items have no verdict, e.g. {items[:5]}")
        self.items = items


def ground_truth(sample: Dict[str, Any]) -> List[str]:
    """Annotated object names of a loader sample: COCO categories or Visual Genome object names."""
    return list(sample["categories"])


def dataset_key(loader) -> str:
    """The POOL / CAPTION_PROMPTS key of a loader: its ``KEY``, "coco" or "vg"."""
    return loader.KEY


def verdict_key(model) -> str:
    """The name a model's mention verdicts are stored under: ``model.model_key`` (its registry
    key, set by predict.factory.build_predictor), else its ``model_name``."""
    return getattr(model, "model_key", None) or model.model_name


def object_names(loader) -> List[str]:
    """The dataset vocabulary: COCO category names, or every Visual Genome object name."""
    return loader.get_category_names() if dataset_key(loader) == "coco" else loader.get_object_names()


def pool_objects(loader) -> List[str]:
    """Sorted object names annotated anywhere in the loaded pool (Visual Genome THRONE asks about these)."""
    return sorted({name for sample in loader for name in ground_truth(sample)})


def annotated(loader, n: int = SAMPLES) -> List[Dict[str, Any]]:
    """The first ``n`` samples of ``loader`` with ground truth: the images a caption metric scores."""
    return [s for s in loader if ground_truth(s)][:n]


def caption_images(loader, model, n: int = SAMPLES,
                   gen: Mapping[str, Any] = CAPTION_GEN) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Caption the first ``n`` annotated images of ``loader`` with the dataset's prompt.

    Returns:
        ``(samples, captions)``, one caption per sample.
    """
    prompt = CAPTION_PROMPTS[dataset_key(loader)]
    samples = annotated(loader, n)
    captions = [model.generate(image=loader.load_image(s), prompt=prompt, **gen) for s in samples]
    return samples, captions


class BaseMetric:
    """
    Base class for all evaluation metrics in Mirage.

    Subclasses override the methods their protocol uses: ``run`` or
    ``evaluate``, and ``format_results``.
    """

    POOL = MappingProxyType({"coco": 2000, "vg": 2000})  # candidate images loaded before filtering

    def __init__(self, name: str):
        """
        Initialize the metric.

        Args:
            name: The name of the metric
        """
        self.name = name
        self.results = {}

    def evaluate(self, model_outputs: List[Any], ground_truth: List[Any], **kwargs) -> Dict[str, float]:
        """
        Evaluate model outputs against ground truth.

        Args:
            model_outputs: List of model outputs
            ground_truth: List of ground truth data
            **kwargs: Additional arguments specific to the metric

        Returns:
            Dictionary of metric scores
        """
        raise NotImplementedError(f"{type(self).__name__} has no evaluate()")

    def format_results(self) -> Dict[str, Any]:
        """
        Format the evaluation results for reporting.

        Returns:
            Dictionary containing formatted results
        """
        raise NotImplementedError(f"{type(self).__name__} has no format_results()")

    def run(self, loader, model, n: int = SAMPLES, gen: Mapping[str, Any] = CAPTION_GEN) -> List[Tuple[Any, str]]:
        """Caption the first ``n`` annotated images, then evaluate the captions.

        This is the caption protocol (CHAIR, THRONE); POPE and ROPE override it
        with their question loops. Read the scores with :meth:`format_results`.

        Args:
            loader: A loaded COCO or Visual Genome dataloader.
            model: Anything with ``generate(image=..., prompt=..., **gen)``.
            n: Images to score.
            gen: Caption generation settings (``max_new_tokens``, ``do_sample``).

        Returns:
            ``(image_id, caption)`` per scored image.
        """
        samples, captions = caption_images(loader, model, n, gen)
        self.evaluate(captions, [ground_truth(s) for s in samples])
        return [(s["image_id"], c) for s, c in zip(samples, captions)]

    def reset(self) -> None:
        """Reset the metric state."""
        self.results = {}

    def compute_accuracy(self, predictions: List[bool], labels: List[bool]) -> float:
        """Fraction of ``predictions`` equal to ``labels``; 0.0 when empty."""
        if len(predictions) == 0:
            return 0.0

        correct = sum(p == l for p, l in zip(predictions, labels))
        return correct / len(predictions)

    def compute_confusion_matrix(self, predictions: List[bool], labels: List[bool]) -> Dict[str, int]:
        """``true_positives``, ``true_negatives``, ``false_positives`` and ``false_negatives`` counts."""
        if len(predictions) != len(labels):
            raise ValueError("Predictions and labels must have the same length")

        tp = sum(p and l for p, l in zip(predictions, labels))
        tn = sum(not p and not l for p, l in zip(predictions, labels))
        fp = sum(p and not l for p, l in zip(predictions, labels))
        fn = sum(not p and l for p, l in zip(predictions, labels))

        return {
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn
        }

    def compute_precision(self, tp: int, fp: int) -> float:
        """TP / (TP + FP); 0.0 when nothing was predicted positive."""
        if tp + fp == 0:
            return 0.0
        return tp / (tp + fp)

    def compute_recall(self, tp: int, fn: int) -> float:
        """TP / (TP + FN); 0.0 when there are no positives."""
        if tp + fn == 0:
            return 0.0
        return tp / (tp + fn)

    def compute_f1(self, precision: float, recall: float) -> float:
        """Harmonic mean of precision and recall; 0.0 when both are 0."""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
