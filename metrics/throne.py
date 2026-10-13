"""
THRONE (Text-based Hallucination Rate ON Evaluation) metric.

An LLM-based hallucination metric that uses a language model evaluator (FLAN-T5)
to semantically determine which objects a VLM claims to see in its generated caption,
then compares against ground-truth annotations.

Unlike CHAIR (text-matching), THRONE uses semantic understanding via an evaluator LM,
halving the misjudgement rate (4.3% vs 8.8%).

Reference:
    Kaul, P., Haydarov, K., Dang-Nguyen, D. T., & Kalan, F. S. (2024).
    THRONE: An Object-based Hallucination Benchmark for the Free-form
    Generations of Large Vision-Language Models. CVPR 2024.
    https://arxiv.org/abs/2405.05256
    https://github.com/amazon-science/THRONE (CC-BY-NC-4.0)
"""

import hashlib
import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter

import torch

from predict.backends import CallableBackend
from predict.cache import SQLiteResponseCache, default_cache_path
from predict.identity import parse_model_name
from predict.predictor import Predictor

from .base_metric import BaseMetric

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants from the THRONE paper (throne_constants.py)
# ---------------------------------------------------------------------------

THRONE_INSTRUCTIONS = [
    "Read the text and answer the question.",
    "Read the text about an image and answer the question.",
    "Now answer this question.",
]

# 8 question-format pairs: (singular_template, plural_template)
# {0} = article (a/an), {1} = object name for singular; {0} = plural name for plural
THRONE_QUESTION_FORMATS: List[Tuple[str, str]] = [
    ("Please answer yes or no. Is there {0} {1} in this image?",
     "Please answer yes or no. Are there {0} in this image?"),
    ("Is there {0} {1} in this image?",
     "Are there {0} in this image?"),
    ("Please answer yes or no. Does the text imply {0} {1} is in the image?",
     "Please answer yes or no. Does the text imply {0} are in the image?"),
    ("Does the text imply {0} {1} is in the image?",
     "Does the text imply {0} are in the image?"),
    ("Please answer yes or no. Does the text give evidence for {0} {1} being present?",
     "Please answer yes or no. Does the text give evidence for {0} being present?"),
    ("Does the text give evidence for {0} {1} being present?",
     "Does the text give evidence for {0} being present?"),
    ("Please answer yes or no. Does the text explicitly mention {0} {1} is in the image?",
     "Please answer yes or no. Does the text explicitly mention {0} are in the image?"),
    ("Does the text explicitly mention {0} {1} is in the image?",
     "Does the text explicitly mention {0} are in the image?"),
]

# COCO 80 category names (default object categories for THRONE evaluation)
COCO_80_CATEGORIES: List[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

# Categories that are inherently plural (use plural question templates)
_PLURAL_CATEGORIES = {"skis", "scissors"}

# Single source of truth for evaluator generation settings: used both by the
# raw generate call and as the CallableBackend defaults that enter cache keys.
_EVALUATOR_GEN_CONFIG = {"max_new_tokens": 5, "do_sample": False}

# FLAN-T5 was trained at 512 tokens. The evaluator prompt is
# "Text: {caption}\n\n{instruction}\n\n{question}"; if it is longer than this
# only the caption is trimmed (from its end) so the question always survives.
_EVALUATOR_MAX_TOKENS = 512

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def evaluator_cache_path() -> Path:
    """Where the evaluator's own cache DB lives.

    One row per (caption, category list, formats) bundle holding every
    evaluator answer for that caption (80x3 on COCO, ~4330x3 on VG). Storing
    one row per question instead was measured at 13M rows / 10 GB per VG
    cell, and random inserts into a multi-GB SQLite file on Lustre ran at
    ~80 s per caption; the bundle store is ~1000 rows per cell. Still kept
    in its own file, never merged into ``cache/predictions.sqlite`` and never
    committed: ``$MIRAGE_EVALUATOR_CACHE_DIR/evaluator.sqlite`` when set,
    else ``evaluator.sqlite`` next to the default caption cache.
    """
    override = os.environ.get("MIRAGE_EVALUATOR_CACHE_DIR")
    if override:
        return Path(override) / "evaluator.sqlite"
    return default_cache_path().with_name("evaluator.sqlite")


def _get_article(name: str) -> str:
    """Return 'a' or 'an' for a given object name."""
    if name and name[0].lower() in "aeiou":
        return "an"
    return "a"


class ThroneMetric(BaseMetric):
    """
    THRONE hallucination metric using LLM-based abstractive question answering.

    The evaluator LM reads the VLM-generated caption (the model output being
    evaluated) and answers yes/no questions about whether each object class is
    mentioned. The ground-truth object lists are only used at scoring time to
    compute precision/recall.

    Pipeline:
        1. VLM generates a free-form caption (input to this metric)
        2. Evaluator LM (FLAN-T5) reads the caption and answers per-object questions
        3. Majority voting determines which objects the VLM claimed to see
        4. Compare predicted set vs ground-truth for precision/recall/F-scores

    Args:
        evaluator_model_name: HuggingFace model name for the evaluator LM.
        object_categories: List of object category names to check against.
            Defaults to COCO 80 categories.
        num_question_formats: Number of question format pairs to use (1-8).
            Default 3 as in the paper.
        voting_threshold: Fraction of votes needed to mark an object as
            "present". None means unanimous voting. Default None.
        device: Device for evaluator model ('cuda', 'cpu', or None for auto).
        batch_size: Batch size for evaluator inference.
        evaluator_dtype: Weight dtype name for the evaluator ('bfloat16',
            'float16', 'float32'). Part of the cache identity, so it must not
            depend on the device: a CPU replay of a GPU run has to produce the
            same keys. Default ``$MIRAGE_THRONE_EVALUATOR_DTYPE`` or bfloat16.
        evaluator_cache: Cache for evaluator answers; None uses
            :func:`evaluator_cache_path`.
        distractor_ratio: When set, ask about every ground-truth category plus
            ``round(ratio * len(gt))`` randomly chosen others, instead of the
            whole category set. Precision is governed by the distractor-to-
            positive ratio, so a fixed ratio is what makes the number
            comparable between a closed 80-class set and an open 4330-class
            one; COCO's native ratio is ~38. Recall is unaffected because
            every ground-truth category is still asked. The draw is seeded
            from the caption, so it is reproducible without an image id.
    """

    def __init__(
        self,
        evaluator_model_name: str = "google/flan-t5-base",
        object_categories: Optional[List[str]] = None,
        num_question_formats: int = 3,
        voting_threshold: Optional[float] = None,
        device: Optional[str] = None,
        batch_size: int = 32,
        evaluator_dtype: Optional[str] = None,
        evaluator_cache: Optional[SQLiteResponseCache] = None,
        distractor_ratio: Optional[float] = None,
    ):
        super().__init__("throne")

        self._evaluator_model_name = evaluator_model_name
        # Sorted: callers often pass set-derived lists whose order changes per
        # process, and predicted/disqualified lists follow category order, so
        # an unsorted list makes the results JSON differ between identical runs.
        self._object_categories = sorted(
            {c.lower() for c in (object_categories or COCO_80_CATEGORIES)}
        )
        dtype_name = (evaluator_dtype or os.environ.get("MIRAGE_THRONE_EVALUATOR_DTYPE")
                      or "bfloat16").lower()
        if dtype_name not in _DTYPES:
            raise ValueError(f"evaluator_dtype must be one of {sorted(_DTYPES)}, got {dtype_name!r}")
        self._evaluator_dtype_name = dtype_name
        self._evaluator_cache = evaluator_cache
        self._truncated_inputs = 0
        env_ratio = os.environ.get("MIRAGE_THRONE_DISTRACTOR_RATIO")
        if distractor_ratio is None and env_ratio:
            distractor_ratio = float(env_ratio)
        self._distractor_ratio = distractor_ratio
        self._questions_per_image: List[int] = []
        self._num_question_formats = min(
            max(num_question_formats, 1),
            len(THRONE_QUESTION_FORMATS),
        )
        self._voting_threshold = voting_threshold
        self._batch_size = batch_size

        if device is not None:
            self._device = device
        elif torch.cuda.is_available():
            self._device = "cuda"
        else:
            self._device = "cpu"

        self._evaluator_model = None
        self._evaluator_tokenizer = None
        self._predictor = None

        self._image_precisions: List[float] = []
        self._image_recalls: List[float] = []
        self._image_details: List[Dict[str, Any]] = []
        self._class_tp: Counter = Counter()
        self._class_fp: Counter = Counter()
        self._class_fn: Counter = Counter()

    # ------------------------------------------------------------------
    # Evaluator model lifecycle
    # ------------------------------------------------------------------

    def load_evaluator(self) -> None:
        """Set up the cached evaluator; model weights load on the first cache miss."""
        model_id, revision = parse_model_name(self._evaluator_model_name)
        cache = self._evaluator_cache
        if cache is None and os.environ.get("MIRAGE_CACHE_DISABLE") != "1":
            cache = SQLiteResponseCache(evaluator_cache_path(), secondary_indexes=False)
        self._predictor = Predictor(CallableBackend(
            fn=self._answer_bundles,
            identity={
                "provider": "local-callable",
                "class_name": "ThroneEvaluator",
                "model_id": model_id,
                "revision": revision,
                "dtype": str(self._evaluator_dtype()),
                "max_input_tokens": _EVALUATOR_MAX_TOKENS,
                "granularity": "caption-bundle",
            },
            defaults=dict(_EVALUATOR_GEN_CONFIG),
            loader=self._load_evaluator_weights,
            unloader=self._unload_evaluator_weights,
        ), cache=cache)
        self._predictor.load_model()

    def _evaluator_dtype(self):
        return _DTYPES[self._evaluator_dtype_name]

    def _load_evaluator_weights(self) -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        logger.info("Loading THRONE evaluator model: %s on %s",
                     self._evaluator_model_name, self._device)
        self._evaluator_tokenizer = self._get_tokenizer()
        self._evaluator_model = AutoModelForSeq2SeqLM.from_pretrained(
            self._evaluator_model_name,
            torch_dtype=self._evaluator_dtype(),
        ).to(self._device).eval()
        logger.info("THRONE evaluator model loaded successfully")

    def _unload_evaluator_weights(self) -> None:
        if self._evaluator_model is not None:
            del self._evaluator_model
        if self._evaluator_tokenizer is not None:
            del self._evaluator_tokenizer
        self._evaluator_model = None
        self._evaluator_tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def unload_evaluator(self) -> None:
        """Unload the evaluator model and free resources."""
        if self._predictor is not None:
            self._predictor.unload_model()
            self._predictor.close()
            self._predictor = None
        else:
            self._unload_evaluator_weights()
        logger.info("THRONE evaluator model unloaded")

    def _ensure_evaluator_loaded(self) -> None:
        """Set up the evaluator predictor if not already done."""
        if self._predictor is None:
            self.load_evaluator()

    # ------------------------------------------------------------------
    # Question generation
    # ------------------------------------------------------------------

    def _fit_caption(self, caption: str) -> str:
        """Trim the caption so every evaluator prompt fits in the model's window.

        Trims from the end of the caption only; instruction and question are
        never cut. Done before prompts are built so the cache key sees the
        text the evaluator actually reads.
        """
        tok = self._get_tokenizer()
        longest_q = max(
            (tmpl[0].format("an", "x") for tmpl in THRONE_QUESTION_FORMATS[:self._num_question_formats]),
            key=len,
        )
        overhead = len(tok(f"Text: \n\n{THRONE_INSTRUCTIONS[0]}\n\n{longest_q}").input_ids) + 8
        budget = _EVALUATOR_MAX_TOKENS - overhead
        ids = tok(caption, add_special_tokens=False).input_ids
        if len(ids) <= budget:
            return caption
        self._truncated_inputs += 1
        return tok.decode(ids[:budget], skip_special_tokens=True)

    def _get_tokenizer(self):
        if self._evaluator_tokenizer is None:
            from transformers import AutoTokenizer
            self._evaluator_tokenizer = AutoTokenizer.from_pretrained(self._evaluator_model_name)
        return self._evaluator_tokenizer

    def _generate_questions_for_category(
        self,
        vlm_caption: str,
        category: str,
    ) -> List[str]:
        """Generate M question prompts for one object category against one VLM caption.

        Args:
            vlm_caption: The VLM-generated caption (model output being evaluated).
            category: Object category name to ask about.

        Returns:
            List of formatted prompts for the evaluator LM.
        """
        is_plural = category.lower() in _PLURAL_CATEGORIES
        article = _get_article(category)
        instruction = THRONE_INSTRUCTIONS[0]

        prompts = []
        for i in range(self._num_question_formats):
            singular_tmpl, plural_tmpl = THRONE_QUESTION_FORMATS[i]

            if is_plural:
                question = plural_tmpl.format(category)
            else:
                question = singular_tmpl.format(article, category)

            # Match the THRONE paper prompt format:
            # "Text: {context}\n\n{instruction}\n\n{question}"
            prompt = f"Text: {vlm_caption}\n\n{instruction}\n\n{question}"
            prompts.append(prompt)

        return prompts

    # ------------------------------------------------------------------
    # Evaluator inference
    # ------------------------------------------------------------------

    def _categories_for(self, caption: str, gt_objects: List[str]) -> List[str]:
        """Category set to ask about for one image.

        Without ``distractor_ratio`` this is the full set. With it, every
        ground-truth category plus a seeded random sample of others, so the
        distractor-to-positive ratio is constant across images and datasets.
        """
        if not self._distractor_ratio:
            return self._object_categories
        gt = {o.lower() for o in gt_objects}
        present = [c for c in self._object_categories if c in gt]
        distractors = [c for c in self._object_categories if c not in gt]
        k = min(int(round(self._distractor_ratio * len(gt))), len(distractors))
        seed = hashlib.sha256(caption[:200].encode("utf-8")).hexdigest()
        drawn = random.Random(seed).sample(distractors, k) if k else []
        return sorted(set(present) | set(drawn))

    def _bundle(self, fitted_caption: str, categories: List[str]) -> str:
        """Serialize one caption's whole question set as a single cache-able request."""
        return json.dumps({
            "caption": fitted_caption,
            "categories": categories,
            "instruction": THRONE_INSTRUCTIONS[0],
            "formats": self._num_question_formats,
        }, ensure_ascii=False, sort_keys=True)

    def _query_evaluator(self, fitted_caption: str, categories: List[str]) -> List[bool]:
        """Answer every (category x format) question for one caption via the predict cache.

        Returns:
            One boolean per prompt, in ``categories`` x format order
            (True = "yes").
        """
        text = self._predictor.predict(self._bundle(fitted_caption, categories))
        answers = json.loads(text or "[]")
        expected = len(categories) * self._num_question_formats
        if len(answers) != expected:
            raise RuntimeError(f"evaluator returned {len(answers)} answers for {expected} questions")
        return [bool(a) for a in answers]

    def _answer_bundles(self, bundles: List[str]) -> List[str]:
        """CallableBackend fn: expand each bundle to prompts, run the LM, pack yes/no bits."""
        out: List[str] = []
        for b in bundles:
            req = json.loads(b)
            prompts: List[str] = []
            for cat in req["categories"]:
                prompts.extend(self._generate_questions_for_category(req["caption"], cat))
            texts = self._raw_generate_batch(prompts)
            out.append(json.dumps([int((t or "").strip().lower().startswith("yes")) for t in texts]))
        return out

    @torch.inference_mode()
    def _raw_generate_batch(self, prompts: List[str]) -> List[str]:
        """Tokenize, generate, and decode with the raw evaluator model."""
        texts: List[str] = []
        for i in range(0, len(prompts), self._batch_size):
            batch = prompts[i:i + self._batch_size]
            inputs = self._evaluator_tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self._device)

            output_ids = self._evaluator_model.generate(
                **inputs,
                **_EVALUATOR_GEN_CONFIG,
            )

            texts.extend(self._evaluator_tokenizer.batch_decode(
                output_ids, skip_special_tokens=True,
            ))
        return texts

    # ------------------------------------------------------------------
    # Voting
    # ------------------------------------------------------------------

    def _vote(self, votes: List[bool]) -> str:
        """Apply voting to determine object presence.

        Args:
            votes: List of yes/no votes from different question formats.

        Returns:
            "present", "absent", or "disqualified".
        """
        if not votes:
            return "absent"

        yes_count = sum(votes)
        total = len(votes)

        if self._voting_threshold is not None:
            threshold = self._voting_threshold * total
            complement = (1.0 - self._voting_threshold) * total
            if yes_count >= threshold:
                return "present"
            elif yes_count <= complement:
                return "absent"
            else:
                return "disqualified"
        else:
            # Unanimous voting (default)
            if yes_count == total:
                return "present"
            elif yes_count == 0:
                return "absent"
            else:
                return "disqualified"

    # ------------------------------------------------------------------
    # F-beta score helper
    # ------------------------------------------------------------------

    @staticmethod
    def compute_f_beta(precision: float, recall: float, beta: float) -> float:
        """Compute F-beta score.

        Args:
            precision: Precision value (0-1).
            recall: Recall value (0-1).
            beta: Beta parameter. beta < 1 weights precision more;
                beta > 1 weights recall more.

        Returns:
            F-beta score (0-1).
        """
        if precision + recall == 0:
            return 0.0
        beta_sq = beta ** 2
        return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        model_outputs: List[str],
        ground_truth: List[List[str]],
        **kwargs,
    ) -> Dict[str, float]:
        """Evaluate VLM-generated captions for hallucinations using THRONE.

        The evaluator LM reads each VLM-generated caption and determines which
        objects the VLM claimed to see. Ground-truth object lists are only used
        at the scoring step to compute precision/recall.

        Args:
            model_outputs: List of VLM-generated captions to evaluate.
            ground_truth: List of ground-truth object lists per image.
            **kwargs: Additional arguments (unused).

        Returns:
            Dictionary of metric scores including precision, recall, f1, f0.5.
        """
        if len(model_outputs) != len(ground_truth):
            raise ValueError(
                f"Length mismatch: {len(model_outputs)} captions vs "
                f"{len(ground_truth)} ground truth lists"
            )

        self._ensure_evaluator_loaded()

        categories = self._object_categories
        num_categories = len(categories)
        M = self._num_question_formats

        logger.info(
            "THRONE evaluation: %d captions, %d categories, %d question formats",
            len(model_outputs), num_categories, M,
        )

        for idx, (caption, gt_objects) in enumerate(
            zip(model_outputs, ground_truth)
        ):
            # Ask the evaluator every (category x format) question for this
            # caption; one cache row per caption.
            fitted = self._fit_caption(caption or "")
            cats = self._categories_for(caption or "", gt_objects)
            self._questions_per_image.append(len(cats) * M)
            all_answers = self._query_evaluator(fitted, cats)

            # Reshape: (num_categories, M)
            predicted_set: List[str] = []
            disqualified_set: List[str] = []

            for cat_idx, cat in enumerate(cats):
                start = cat_idx * M
                votes = all_answers[start:start + M]
                verdict = self._vote(votes)
                if verdict == "present":
                    predicted_set.append(cat)
                elif verdict == "disqualified":
                    disqualified_set.append(cat)

            gt_set = set(obj.lower() for obj in gt_objects)
            disqualified_set_lower = set(disqualified_set)
            gt_filtered = gt_set - disqualified_set_lower
            pred_filtered = set(predicted_set) - disqualified_set_lower

            # Compute per-image precision and recall
            if len(pred_filtered) > 0 and len(gt_filtered) > 0:
                intersection = pred_filtered & gt_filtered
                precision = len(intersection) / len(pred_filtered)
                recall = len(intersection) / len(gt_filtered)
            else:
                precision = 0.0
                recall = 0.0

            self._image_precisions.append(precision)
            self._image_recalls.append(recall)

            # Class-wise tracking
            for cat in pred_filtered:
                if cat in gt_filtered:
                    self._class_tp[cat] += 1
                else:
                    self._class_fp[cat] += 1
            for cat in gt_filtered:
                if cat not in pred_filtered:
                    self._class_fn[cat] += 1

            # Store per-image details (sorted: set order is per-process)
            self._image_details.append({
                "caption": caption,
                "gt_objects": sorted(gt_set),
                "predicted_objects": predicted_set,
                "disqualified_objects": disqualified_set,
                "precision": precision,
                "recall": recall,
            })

            logger.info(
                "Image %d/%d: P=%.3f R=%.3f predicted=%d gt=%d disqualified=%d",
                idx + 1, len(model_outputs), precision, recall,
                len(predicted_set), len(gt_set), len(disqualified_set),
            )

        # Aggregate metrics (micro-averaging at image level)
        num_images = len(self._image_precisions)
        if num_images > 0:
            avg_precision = sum(self._image_precisions) / num_images
            avg_recall = sum(self._image_recalls) / num_images
        else:
            avg_precision = 0.0
            avg_recall = 0.0

        f1 = self.compute_f1(avg_precision, avg_recall)
        f05 = self.compute_f_beta(avg_precision, avg_recall, beta=0.5)

        self.results = {
            "precision": avg_precision,
            "recall": avg_recall,
            "f1": f1,
            "f0.5": f05,
            "num_images": num_images,
            "num_categories_checked": num_categories,
            "evaluator_model": self._evaluator_model_name,
            "evaluator_dtype": self._evaluator_dtype_name,
            "num_question_formats": M,
            "num_truncated_evaluator_inputs": self._truncated_inputs,
            "distractor_ratio": self._distractor_ratio,
            "median_questions_per_image": (
                sorted(self._questions_per_image)[len(self._questions_per_image) // 2]
                if self._questions_per_image else 0
            ),
        }
        if self._truncated_inputs:
            logger.warning("THRONE: %d/%d captions were trimmed to fit the evaluator's %d-token window",
                           self._truncated_inputs, num_images, _EVALUATOR_MAX_TOKENS)

        return self.results

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def format_results(self) -> Dict[str, Any]:
        """Format evaluation results for reporting.

        Returns:
            Dictionary with throne_results, per_image_details,
            class_wise_results, and summary.
        """
        # Build class-wise results
        all_classes = set(self._class_tp.keys()) | set(self._class_fp.keys()) | set(self._class_fn.keys())
        class_wise: Dict[str, Dict[str, Any]] = {}
        for cls in sorted(all_classes):
            tp = self._class_tp[cls]
            fp = self._class_fp[cls]
            fn = self._class_fn[cls]
            p = self.compute_precision(tp, fp)
            r = self.compute_recall(tp, fn)
            class_wise[cls] = {
                "tp": tp, "fp": fp, "fn": fn,
                "precision": p, "recall": r,
                "f1": self.compute_f1(p, r),
            }

        return {
            "throne_results": self.results,
            "per_image_details": self._image_details,
            "class_wise_results": class_wise,
            "summary": {
                "precision": self.results.get("precision", 0.0),
                "recall": self.results.get("recall", 0.0),
                "f1": self.results.get("f1", 0.0),
                "f0.5": self.results.get("f0.5", 0.0),
                "total_images": len(self._image_precisions),
            },
        }

    def reset(self) -> None:
        """Reset metric state for a new evaluation.

        Note: Does NOT unload the evaluator model. Call unload_evaluator()
        explicitly for full cleanup.
        """
        super().reset()
        self._truncated_inputs = 0
        self._questions_per_image = []
        self._image_precisions = []
        self._image_recalls = []
        self._image_details = []
        self._class_tp = Counter()
        self._class_fp = Counter()
        self._class_fn = Counter()
