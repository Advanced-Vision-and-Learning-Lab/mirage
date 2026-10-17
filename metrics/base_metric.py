from abc import ABC, abstractmethod
from types import MappingProxyType
from typing import Dict, List, Any, Union, Optional
import numpy as np

SAMPLES = 500  # images scored per cell: run.py --samples default
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


class BaseMetric(ABC):
    """
    Abstract base class for all evaluation metrics in Mirage.

    All metric implementations must inherit from this class and implement
    the required abstract methods.
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

    @abstractmethod
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
        pass

    @abstractmethod
    def format_results(self) -> Dict[str, Any]:
        """
        Format the evaluation results for reporting.

        Returns:
            Dictionary containing formatted results
        """
        pass

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
