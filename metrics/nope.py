"""
NOPE (Negative Object Presence Evaluation) metric.

Evaluates whether VLLMs hallucinate objects by asking about objects that are
NOT present in the image and measuring how often the model correctly negates.
"""

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from .base_metric import BaseMetric

logger = logging.getLogger(__name__)

NEGP_ANSWERS = {"none", "nothing", "nowhere", "zero", "0", "no one", "nobody", "neither", "no"}

NEGP_PATTERN = re.compile(r"\b(?:%s)\b" % "|".join(re.escape(a) for a in sorted(NEGP_ANSWERS)))


def is_negp(response: str) -> bool:
    """
    Determine whether a model response contains a NegP (negative presence) word.

    Matches whole words only, so "10" is not read as "0" nor "snowman" as "no".

    Args:
        response: The model's raw response string.

    Returns:
        True if the response contains a negative-presence term, False otherwise.
    """
    return NEGP_PATTERN.search(response.lower()) is not None


class NOPEMetric(BaseMetric):
    """
    Negative Object Presence Evaluation (NOPE) metric.

    NOPE evaluates a model's resistance to hallucination by asking questions
    about objects that are known to be absent from the image. A correct answer
    must acknowledge the object is not present using a NegP expression.

    Initialised from the Turn 2 results JSON produced by NOPEGenerator, with an
    optional Turn 1 results JSON to enrich records with present_objects.

    Metrics computed:
        - negp_accuracy: Fraction of responses containing a NegP word.
        - standard_accuracy: Fraction of responses exactly matching ground truth.
        - hallucination_gap: standard_accuracy - negp_accuracy.
        - Per-relevance breakdowns (related / loosely_related / unrelated).

    Args:
        turn2_path: Path to nope_turn2_results.json from NOPEGenerator.
        turn1_path: Optional path to nope_turn1_results.json for present_objects enrichment.
        image_dir: Optional base directory prepended to build image_path per record.

    Raises:
        FileNotFoundError: If turn2_path does not exist.
    """

    def __init__(
        self,
        turn2_path: str,
        turn1_path: str = None,
        image_dir: str = None,
    ) -> None:
        """
        Initialize NOPEMetric from Turn 2 results JSON.

        Args:
            turn2_path: Path to nope_turn2_results.json.
            turn1_path: Optional path to nope_turn1_results.json.
            image_dir: Optional image directory for resolving image paths.

        Raises:
            FileNotFoundError: If turn2_path is not found.
        """
        super().__init__("nope")

        t2 = Path(turn2_path)
        if not t2.exists():
            raise FileNotFoundError(
                f"NOPE Turn 2 results not found at: {turn2_path}. "
                "Generate it first using NOPEGenerator."
            )

        with open(t2) as f:
            turn2_data: List[Dict[str, Any]] = json.load(f)

        # Load turn1 for present_objects if provided
        turn1_map: Dict[str, Dict] = {}
        if turn1_path:
            t1 = Path(turn1_path)
            if not t1.exists():
                raise FileNotFoundError(f"NOPE Turn 1 results not found at: {turn1_path}.")
            with open(t1) as f:
                turn1_map = json.load(f)

        self.records: List[Dict[str, Any]] = []
        for item in turn2_data:
            key = item["key"]
            image_id, absent_object = key.split("|", 1)
            t1_entry = turn1_map.get(image_id, {})
            present_objects = t1_entry.get("present_objects", [])

            image_path = ""
            if image_dir:
                image_path = str(Path(image_dir) / image_id)

            self.records.append({
                "image_id": image_id,
                "image_path": image_path,
                "question": item["question"],
                "answer": item["answer"],
                "absent_object": absent_object,
                "present_objects": present_objects,
                "object_scene_relevance": "related",
            })

        logger.info("Loaded %d NOPE records from %s.", len(self.records), turn2_path)

        self._per_record_results: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def evaluate(self, model_outputs: List[str], **kwargs) -> Dict[str, float]:
        """
        Evaluate model responses against NOPE ground-truth records.

        Args:
            model_outputs: List of model response strings, one per NOPE record.
            **kwargs: Unused additional arguments.

        Returns:
            Dict with negp_accuracy, standard_accuracy, hallucination_gap,
            and per-relevance category breakdowns.

        Raises:
            ValueError: If length of model_outputs does not match number of records.
        """
        if len(model_outputs) != len(self.records):
            raise ValueError(
                f"Length mismatch: {len(model_outputs)} model outputs vs "
                f"{len(self.records)} NOPE records."
            )

        negp_hits = 0
        exact_hits = 0
        relevance_buckets: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "negp": 0, "exact": 0})

        per_record: List[Dict[str, Any]] = []
        for record, response in zip(self.records, model_outputs):
            negp_match = is_negp(response)
            exact_match = response.strip().lower() == record["answer"].strip().lower()
            relevance = record.get("object_scene_relevance", "unknown")

            if negp_match:
                negp_hits += 1
            if exact_match:
                exact_hits += 1

            bucket = relevance_buckets[relevance]
            bucket["total"] += 1
            if negp_match:
                bucket["negp"] += 1
            if exact_match:
                bucket["exact"] += 1

            per_record.append({
                "image_id": record["image_id"],
                "absent_object": record["absent_object"],
                "response": response,
                "ground_truth": record["answer"],
                "negp_match": negp_match,
                "exact_match": exact_match,
                "relevance": relevance,
            })

        self._per_record_results.extend(per_record)

        n = len(self.records)
        negp_accuracy = negp_hits / n if n > 0 else 0.0
        standard_accuracy = exact_hits / n if n > 0 else 0.0
        hallucination_gap = standard_accuracy - negp_accuracy

        relevance_breakdown = {}
        for rel, counts in relevance_buckets.items():
            t = counts["total"]
            relevance_breakdown[rel] = {
                "negp_accuracy": counts["negp"] / t if t > 0 else 0.0,
                "standard_accuracy": counts["exact"] / t if t > 0 else 0.0,
                "count": t,
            }

        self.results = {
            "negp_accuracy": negp_accuracy,
            "standard_accuracy": standard_accuracy,
            "hallucination_gap": hallucination_gap,
            "num_questions": n,
            "relevance_breakdown": relevance_breakdown,
        }

        return self.results

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def format_results(self) -> Dict[str, Any]:
        """
        Format evaluation results for reporting.

        Returns:
            Nested dict with summary metrics and per-relevance category breakdown.
        """
        return {
            "nope_results": {
                "negp_accuracy": self.results.get("negp_accuracy", 0.0),
                "standard_accuracy": self.results.get("standard_accuracy", 0.0),
                "hallucination_gap": self.results.get("hallucination_gap", 0.0),
                "num_questions": self.results.get("num_questions", 0),
            },
            "relevance_breakdown": self.results.get("relevance_breakdown", {}),
            "summary": {
                "negp_accuracy": self.results.get("negp_accuracy", 0.0),
                "standard_accuracy": self.results.get("standard_accuracy", 0.0),
                "hallucination_gap": self.results.get("hallucination_gap", 0.0),
                "total_questions": self.results.get("num_questions", 0),
            },
        }

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset accumulated evaluation state for a new run."""
        super().reset()
        self._per_record_results = []
