from abc import ABC
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
    """Annotated object names of a loader sample: COCO categories, else Visual Genome object names."""
    if "categories" in sample:
        return list(sample["categories"])
    return [obj.get("name", "") for obj in sample.get("objects", []) if obj.get("name")]


def dataset_key(loader) -> str:
    """The POOL / CAPTION_PROMPTS key of a loader: "coco" (it has category names), else "vg"."""
    return "coco" if hasattr(loader, "get_category_names") else "vg"


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


def caption_images(loader, model, n: int = SAMPLES,
                   gen: Mapping[str, Any] = CAPTION_GEN) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Caption the first ``n`` annotated images of ``loader`` with the dataset's prompt.

    Returns:
        ``(samples, captions)``, one caption per sample.
    """
    prompt = CAPTION_PROMPTS[dataset_key(loader)]
    samples = [s for s in loader if ground_truth(s)][:n]
    captions = [model.generate(image=loader.load_image(s), prompt=prompt, **gen) for s in samples]
    return samples, captions


class BaseMetric(ABC):
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
        """
        Compute accuracy from boolean predictions and labels.

        Args:
            predictions: List of boolean predictions
            labels: List of boolean labels

        Returns:
            Accuracy score
        """
        if len(predictions) == 0:
            return 0.0

        correct = sum(p == l for p, l in zip(predictions, labels))
        return correct / len(predictions)

    def compute_confusion_matrix(self, predictions: List[bool], labels: List[bool]) -> Dict[str, int]:
        """
        Compute confusion matrix components from boolean predictions and labels.

        Calculates the four components of a binary confusion matrix:
        - True Positives (TP): Correctly predicted positive cases
        - True Negatives (TN): Correctly predicted negative cases
        - False Positives (FP): Incorrectly predicted positive cases
        - False Negatives (FN): Incorrectly predicted negative cases

        Args:
            predictions: List of boolean predictions (True = positive, False = negative)
            labels: List of boolean ground truth labels (True = positive, False = negative)

        Returns:
            Dictionary containing confusion matrix components with keys:
            'true_positives', 'true_negatives', 'false_positives', 'false_negatives'

        Example:
            >>> predictions = [True, False, True, False]
            >>> labels = [True, False, False, False]
            >>> cm = metric.compute_confusion_matrix(predictions, labels)
            >>> print(cm)
            {'true_positives': 1, 'true_negatives': 2, 'false_positives': 1, 'false_negatives': 0}
        """
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
        """
        Compute precision from confusion matrix components.

        Precision measures the proportion of positive predictions that were correct.
        Formula: precision = TP / (TP + FP)

        Args:
            tp: Number of true positives
            fp: Number of false positives

        Returns:
            Precision score between 0 and 1. Returns 0.0 if no positive predictions were made.

        Example:
            >>> precision = metric.compute_precision(tp=80, fp=20)
            >>> print(precision)
            0.8
        """
        if tp + fp == 0:
            return 0.0
        return tp / (tp + fp)

    def compute_recall(self, tp: int, fn: int) -> float:
        """
        Compute recall (sensitivity) from confusion matrix components.

        Recall measures the proportion of actual positive cases that were correctly identified.
        Formula: recall = TP / (TP + FN)

        Args:
            tp: Number of true positives
            fn: Number of false negatives

        Returns:
            Recall score between 0 and 1. Returns 0.0 if no positive cases exist.

        Example:
            >>> recall = metric.compute_recall(tp=80, fn=20)
            >>> print(recall)
            0.8
        """
        if tp + fn == 0:
            return 0.0
        return tp / (tp + fn)

    def compute_f1(self, precision: float, recall: float) -> float:
        """
        Compute F1 score from precision and recall.

        F1 score is the harmonic mean of precision and recall, providing a balanced
        measure that accounts for both false positives and false negatives.
        Formula: F1 = 2 * (precision * recall) / (precision + recall)

        Args:
            precision: Precision score (between 0 and 1)
            recall: Recall score (between 0 and 1)

        Returns:
            F1 score between 0 and 1. Returns 0.0 if both precision and recall are 0.

        Example:
            >>> f1 = metric.compute_f1(precision=0.8, recall=0.9)
            >>> print(f1)
            0.8470588235294118
        """
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
