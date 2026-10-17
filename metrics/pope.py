import random
import re
from types import MappingProxyType
from typing import Dict, Iterable, List, Any, Tuple, Optional
from collections import Counter

from .base_metric import SAMPLES, BaseMetric, ground_truth, object_names

QUESTIONS_PER_IMAGE = 6  # POPE questions per image (half yes): POPEMetric and run.py --pope-questions default
QUESTION_RE = re.compile(r"Is there an?\s+(.+?)\s+in the image", re.I)


def _binary(rows: List[Tuple[bool, bool]]) -> Dict[str, float]:
    tp = sum(1 for l, y in rows if l and y); tn = sum(1 for l, y in rows if not l and not y)
    fp = sum(1 for l, y in rows if not l and y); fn = sum(1 for l, y in rows if l and not y)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    bal = ((tp / (tp + fn) if tp + fn else 0.0) + (tn / (tn + fp) if tn + fp else 0.0)) / 2
    return {"accuracy": (tp + tn) / len(rows), "balanced_accuracy": bal,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "yes_ratio": (tp + fp) / len(rows), "questions": len(rows)}


class POPEMetric(BaseMetric):
    """
    Precision Of Perception Evaluation (POPE) metric.

    POPE evaluates a model's ability to accurately determine whether objects
    are present in an image by asking yes/no questions.
    """

    MODES = ["random", "popular"]
    GEN = MappingProxyType({"max_new_tokens": 10, "do_sample": False})

    def __init__(self,
                 mode: str = "random",
                 yes_ratio: float = 0.5,
                 num_questions_per_image: int = QUESTIONS_PER_IMAGE,
                 popular_objects_list: Optional[List[str]] = None,
                 seed: int = 42):
        """
        Initialize the POPE metric.

        Args:
            mode: Evaluation mode ("random" or "popular")
            yes_ratio: Ratio of questions expecting "yes" answers
            num_questions_per_image: Number of questions to generate per image
            popular_objects_list: List of popular objects to use in "popular" mode
            seed: RNG seed for negative-object sampling, so the generated
                question set is reproducible across runs (Li et al. 2023
                fixes the protocol). Set to None to use process-global random.
        """
        super().__init__("pope")

        if mode not in self.MODES:
            raise ValueError(f"Mode must be one of {self.MODES}")

        self.mode = mode
        self.yes_ratio = yes_ratio
        self.num_questions_per_image = num_questions_per_image
        self.popular_objects = popular_objects_list or []
        self.seed = seed
        self._rng = random.Random(seed) if seed is not None else random

        # Track performance
        self.predictions = []
        self.labels = []
        self.questions = []
        self.object_performance = {}

    def generate_questions(self,
                          image_objects: List[str],
                          yes_objects: List[str],
                          all_possible_objects: List[str]) -> List[Tuple[str, bool]]:
        """
        Generate yes/no questions about objects in the image.

        Args:
            image_objects: List of objects present in the image
            yes_objects: Objects asked about with a "yes" answer, in order
            all_possible_objects: List of all possible objects in the dataset

        Returns:
            List of (question, expected_answer) tuples
        """
        questions = []
        num_yes = int(self.num_questions_per_image * self.yes_ratio)
        num_no = self.num_questions_per_image - num_yes

        # Generate "yes" questions
        for obj in yes_objects:
            article = "an" if obj and obj[0].lower() in "aeiou" else "a"
            questions.append((f"You are required to answer the following question with either a yes or no. Is there {article} {obj} in the image?", True))

        # Generate "no" questions
        # Sorted so the seeded draw does not depend on candidate order —
        # callers often pass set-derived lists whose order changes per process.
        no_candidates = []
        if self.mode == "random":
            # Random mode: use random objects not in the image
            no_candidates = sorted(obj for obj in all_possible_objects if obj not in image_objects)
        elif self.mode == "popular":
            # Popular mode: use popular objects not in the image
            no_candidates = sorted(obj for obj in self.popular_objects if obj not in image_objects)

        if len(no_candidates) < num_no:
            # If we don't have enough candidates, sample with replacement
            no_objects = self._rng.choices(no_candidates, k=num_no) if no_candidates else []
        else:
            # Otherwise sample without replacement
            no_objects = self._rng.sample(no_candidates, num_no)

        for obj in no_objects:
            article = "an" if obj and obj[0].lower() in "aeiou" else "a"
            questions.append((f"You are required to answer the following question with either a yes or no. Is there {article} {obj} in the image?", False))

        return questions

    def process_response(self, response: str) -> bool:
        """
        Process model response to determine if it's a "yes" or "no".

        Args:
            response: Model's response to a yes/no question

        Returns:
            True if the response is affirmative, False otherwise
        """
        # Whole-word matching, negatives first: "incorrect" / "not right" must
        # not read as yes through their substrings (issue #30). Every response
        # in the September 2026 runs was a bare yes/no, so this changes no
        # committed number; it protects future runs.
        text = (response or "").lower().strip()
        words = set(re.findall(r"[a-z']+", text))
        negative = {"no", "nope", "not", "negative", "false", "incorrect", "wrong", "isn't", "aren't", "doesn't", "don't", "cannot", "can't"}
        affirmative = {"yes", "yeah", "yep", "correct", "true", "right", "affirmative", "indeed"}
        if words & negative:
            return False
        if words & affirmative:
            return True
        # Default to "no" if unclear
        return False

    def run(self, loader, model, n: int = SAMPLES) -> List[Tuple[str, str, bool, bool]]:
        """Ask the first ``n`` annotated images their yes/no questions, then evaluate.

        Per image, the "yes" objects are a seeded draw (seed 42, keyed by image
        id) from its sorted annotations, and the "no" objects come from the
        dataset vocabulary. Read the scores with :meth:`format_results`.

        Returns:
            ``(image_id, object, annotated, said_yes)`` per question, the records
            :meth:`score` takes (built as repro.py builds them from the cache).
        """
        from dataloader.base import sample_for_image

        self._rng = random.Random(self.seed) if self.seed is not None else random
        all_objects = object_names(loader)
        n_yes = int(self.num_questions_per_image * self.yes_ratio)
        asked = []
        for sample in [s for s in loader if ground_truth(s)][:n]:
            objects, image = ground_truth(sample), loader.load_image(sample)
            yes = sample_for_image(sorted(objects), min(n_yes, len(objects)), sample["image_id"], seed=42)
            questions = self.generate_questions(objects, yes, all_objects)
            asked += [(sample, q, expected, model.generate(image=image, prompt=q, **self.GEN))
                      for q, expected in questions]
        self.evaluate([r for *_, r in asked], [(q, expected) for _, q, expected, _ in asked])
        records = []
        for sample, q, _, response in asked:
            obj = QUESTION_RE.search(q).group(1)
            records.append((str(sample["image_id"]), obj, obj in set(ground_truth(sample)),
                            self.process_response(response)))
        return records

    def evaluate(self, model_outputs: List[str], questions_and_answers: List[Tuple[str, bool]], **kwargs) -> Dict[str, float]:
        """
        Evaluate model responses against expected answers.

        Args:
            model_outputs: List of model responses to questions
            questions_and_answers: List of (question, expected_answer) tuples

        Returns:
            Dictionary of metric scores
        """
        predictions = []
        labels = []
        object_counter = Counter()
        object_correct = Counter()

        for response, (question, expected) in zip(model_outputs, questions_and_answers):
            prediction = self.process_response(response)
            predictions.append(prediction)
            labels.append(expected)

            # Extract object from canonical "Is there a/an {obj} in the image?" prompt
            m = re.search(r"Is there an?\s+(.+?)\s+in the image", question, re.IGNORECASE)
            object_name = m.group(1).strip() if m else question
            object_counter[object_name] += 1
            if prediction == expected:
                object_correct[object_name] += 1

        self.predictions.extend(predictions)
        self.labels.extend(labels)
        self.questions.extend([q for q, _ in questions_and_answers])

        # Update object-level performance
        for obj in object_counter:
            if obj not in self.object_performance:
                self.object_performance[obj] = {"total": 0, "correct": 0}

            self.object_performance[obj]["total"] += object_counter[obj]
            self.object_performance[obj]["correct"] += object_correct[obj]

        # Calculate metrics
        accuracy = self.compute_accuracy(predictions, labels)

        # Calculate separate yes/no accuracies
        yes_preds = [p for p, l in zip(predictions, labels) if l]
        yes_labels = [l for l in labels if l]

        no_preds = [p for p, l in zip(predictions, labels) if not l]
        no_labels = [l for l in labels if not l]

        yes_accuracy = self.compute_accuracy(yes_preds, yes_labels)
        no_accuracy = self.compute_accuracy(no_preds, no_labels)

        self.results = {
            "accuracy": accuracy,
            "yes_accuracy": yes_accuracy,
            "no_accuracy": no_accuracy,
            "mode": self.mode,
        }

        return self.results

    def format_results(self) -> Dict[str, Any]:
        """
        Format the evaluation results for reporting.

        Computes comprehensive evaluation metrics including accuracy, confusion matrix,
        precision, recall, and F1 score for the POPE evaluation.

        Returns:
            Dictionary containing formatted results with the following structure:
            - pope_results: Main metrics including accuracy, precision, recall, F1, and confusion matrix
            - object_accuracy: Per-object accuracy scores
            - overall_accuracy: Overall accuracy across all predictions
            - num_questions: Total number of questions asked
            - mode: Evaluation mode used (random/popular)
        """
        # Calculate per-object accuracy
        object_accuracy = {}
        for obj, counts in self.object_performance.items():
            if counts["total"] > 0:
                object_accuracy[obj] = counts["correct"] / counts["total"]

        # Compute overall metrics for the entire run
        overall_accuracy = self.compute_accuracy(self.predictions, self.labels)
        yes_labels = [l for l in self.labels if l]
        yes_preds = [p for p, l in zip(self.predictions, self.labels) if l]
        no_labels = [l for l in self.labels if not l]
        no_preds = [p for p, l in zip(self.predictions, self.labels) if not l]
        yes_accuracy = self.compute_accuracy(yes_preds, yes_labels)
        no_accuracy = self.compute_accuracy(no_preds, no_labels)

        # Compute confusion matrix and derived metrics
        confusion_matrix = self.compute_confusion_matrix(self.predictions, self.labels)
        precision = self.compute_precision(
            confusion_matrix['true_positives'],
            confusion_matrix['false_positives']
        )
        recall = self.compute_recall(
            confusion_matrix['true_positives'],
            confusion_matrix['false_negatives']
        )
        f1_score = self.compute_f1(precision, recall)

        # Calculate yes ratio (percentage of "yes" predictions)
        yes_ratio = sum(self.predictions) / len(self.predictions) if self.predictions else 0.0

        return {
            "pope_results": {
                "accuracy": overall_accuracy,
                "yes_accuracy": yes_accuracy,
                "no_accuracy": no_accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "confusion_matrix": confusion_matrix,
                "mode": self.mode,
                "yes_ratio": yes_ratio
            },
            "object_accuracy": object_accuracy,
            "overall_accuracy": overall_accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "yes_ratio": yes_ratio,
            "num_questions": len(self.questions),
            "mode": self.mode
        }

    @staticmethod
    def score(records: Iterable[Tuple[str, str, bool, bool]], drop: Dict[Tuple[str, str], str]) -> Dict[str, float]:
        """POPE on ``(image_id, object, label, said_yes)``; Mod-POPE drops listed no-questions."""
        records = list(records)
        raw = _binary([(label, yes) for _, _, label, yes in records])
        mod = _binary([(label, yes) for iid, obj, label, yes in records if label or (iid, obj) not in drop])
        return {"raw": raw["accuracy"], "raw_balanced": raw["balanced_accuracy"], "head": mod["balanced_accuracy"],
                "f1": mod["f1"], "yes_ratio": mod["yes_ratio"], "n_raw": raw["questions"], "n_head": mod["questions"]}

