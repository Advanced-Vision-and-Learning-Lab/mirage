"""
ROPE (Recognition-based Object Probing Evaluation) metric.

Adapted from the sled-group/moh repository:
  https://github.com/sled-group/moh

ROPE evaluates multi-object hallucination in Vision-Language Models by drawing
red bounding boxes around N objects in an image and asking the model to identify
all N objects in a single prompt. This contrasts with POPE's binary yes/no
questioning and tests whether VLMs can correctly ground and classify multiple
objects simultaneously.

Reference:
    Zou et al., "Look, Compare, Decide: Alleviating Hallucination in Large
    Vision-Language Models via Multi-View Multi-Path Reasoning"
    https://github.com/sled-group/moh
"""

import re
import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from .base_metric import BaseMetric

# Matchers live in the top-level ``matching`` package (torch-free) so offline
# rescoring can import them without pulling in metrics/__init__ (torch). Ensure
# the repo root is importable regardless of the current working directory.
import os as _os
import sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
from matching.exact import ExactMatcher, SubstringMatcher  # noqa: E402
from matching.base import KEY_FOR, PER_POS_KEY_FOR  # noqa: E402

logger = logging.getLogger(__name__)

# Shared singletons; the module-level helpers delegate so behavior is identical
# to the previous inline implementations (no-regression on lenient/strict).
_SUBSTRING = SubstringMatcher()
_EXACT = ExactMatcher()


# parse_single_response normalizes answer formatting before exact matching:
# markdown marks, a leading header or list marker, and surrounding quotes.
_MARKUP_RE = re.compile(r"[*`]+")
_LINE_PREFIX_RE = re.compile(r"^(?:#+\s*|[-\u2022]\s+|\d+[.)]\s+)")
_QUOTES = "\"'\u2018\u2019\u201c\u201d"
# "... is a/an/the X", "... appears to be (a) X", "... is pointing to (the) X";
# not an echoed question ("I am not sure what is the object inside box 1").
_ANSWER_RE = re.compile(
    r"\b(?<!what )(?:(?:is|appears\s+to\s+be|seems\s+to\s+be)\s+(?:pointing\s+to\s+)?(?:an?|the)"
    r"|appears\s+to\s+be|seems\s+to\s+be|is\s+pointing\s+to)\s+(.+?)(?:[.,;:(]|$)"
)


def _rects_overlap(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    """True when two (x0, y0, x1, y1) rectangles share any area."""
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _rope_match(pred: str, truth: str) -> bool:
    """Lenient ROPE class-name match (delegates to matching.SubstringMatcher).

    Counts a prediction as correct if the ground-truth class appears as a
    whole-word substring of the prediction (or vice versa).
    """
    return _SUBSTRING.match(pred, truth)


def _rope_match_strict(pred: str, truth: str) -> bool:
    """Strict ROPE class-name match (delegates to matching.ExactMatcher)."""
    return _EXACT.match(pred, truth)


class ROPEMetric(BaseMetric):
    """
    Recognition-based Object Probing Evaluation (ROPE) metric.

    Presents the model with an image annotated with red bounding boxes and asks
    it to classify each boxed object from a provided class list in one prompt.
    Tracks per-position and overall classification accuracy.

    Adapted from: https://github.com/sled-group/moh
    """

    def __init__(
        self,
        num_objects: int = 5,
        class_list: Optional[List[str]] = None,
        bbox_color: str = "red",
        bbox_width: int = 3,
        matchers: Optional[List[Any]] = None,
    ):
        """
        Initialize the ROPE metric.

        Args:
            num_objects: Number of bounding boxes / objects per image.
            class_list: List of possible class names to identify from.
            bbox_color: Color of drawn bounding boxes (default red).
            bbox_width: Line width for bounding box outlines.
            matchers: Optional list of matching.Matcher objects. Defaults to
                [SubstringMatcher (lenient), ExactMatcher (strict)] which
                reproduces the committed headline keys exactly. Add a
                SynonymMatcher to also emit ``overall_accuracy_synonym``.
        """
        super().__init__("rope")
        self.num_objects = num_objects
        self.class_list = class_list or []
        self.bbox_color = bbox_color
        self.bbox_width = bbox_width
        self.matchers = matchers or [SubstringMatcher(), ExactMatcher()]

        # Accumulated state across evaluate() calls
        self._all_predictions: List[List[str]] = []
        self._all_ground_truth: List[List[str]] = []

    def draw_bounding_boxes(
        self,
        image: Image.Image,
        bboxes: List[Dict[str, Any]],
    ) -> Image.Image:
        """
        Draw numbered red bounding boxes on a copy of the image.

        Each box is labeled with its 1-based index (1, 2, 3, ...) so the model
        can match the visual box number to the obj1/obj2/... output format.

        Adapted from bbox rendering logic in sled-group/moh evaluation pipeline.

        Args:
            image: Original PIL image.
            bboxes: List of bbox dicts with keys xmin, ymin, xmax, ymax (and
                    optionally 'name' for the object label).

        Returns:
            New PIL image with numbered bounding boxes drawn; original is not modified.
        """
        annotated = deepcopy(image).convert("RGBA")

        overlay = Image.new("RGBA", annotated.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        draw = ImageDraw.Draw(annotated)

        font_size = max(12, min(annotated.width, annotated.height) // 40)
        # Pillow's bundled default face at the computed size. The old macOS
        # Helvetica path fell through to a fixed 10px bitmap font on Linux
        # clusters, and the label pixels are part of the predict-cache key, so
        # the rendering must be identical on every machine that replays a run.
        try:
            font = ImageFont.load_default(size=font_size)  # Pillow >= 10.1
        except TypeError:
            font = ImageFont.load_default()

        placed: List[Tuple[int, int, int, int]] = []
        for i, bbox in enumerate(bboxes):
            x0 = int(bbox["xmin"])
            y0 = int(bbox["ymin"])
            x1 = int(bbox["xmax"])
            y1 = int(bbox["ymax"])

            draw.rectangle([x0, y0, x1, y1], outline=self.bbox_color, width=self.bbox_width)

            label = f"box {i + 1}"
            bbox_text = font.getbbox(label)
            text_w = bbox_text[2] - bbox_text[0]
            text_h = bbox_text[3] - bbox_text[1]
            pad = 3
            box_w = text_w + pad * 2
            box_h = text_h + pad * 2

            label_x, label_y = self._place_label(
                (x0, y0, x1, y1), box_w, box_h, placed
            )
            placed.append((label_x, label_y, label_x + box_w, label_y + box_h))

            overlay_draw.rectangle(
                [label_x, label_y, label_x + box_w, label_y + box_h],
                fill=(0, 0, 0, 180),
            )
            overlay_draw.text(
                (label_x + pad, label_y + pad),
                label,
                fill=(255, 255, 255, 255),
                font=font,
            )

        annotated = Image.alpha_composite(annotated, overlay).convert("RGB")
        return annotated


    @staticmethod
    def _place_label(
        box: Tuple[int, int, int, int],
        box_w: int,
        box_h: int,
        placed: List[Tuple[int, int, int, int]],
    ) -> Tuple[int, int]:
        """Choose a non-overlapping position for one box's number label.

        Candidates are tried in a fixed order and the first one that clears
        every label already placed on this image wins. Candidate 0 is the
        historical position (just above the box's top-left corner), so an
        image whose labels never collided renders byte-identically to before
        the fix — and therefore keeps its prediction-cache entries.

        Every candidate touches its own box, so which label belongs to which
        box stays unambiguous. If all candidates collide (possible when boxes
        are stacked), candidate 0 is used, which is the old behaviour.

        Args:
            box: (x0, y0, x1, y1) of the bounding box in pixels.
            box_w: Width of the label plate in pixels.
            box_h: Height of the label plate in pixels.
            placed: Label plates already drawn on this image.

        Returns:
            (x, y) top-left corner for the label plate.
        """
        x0, y0, x1, y1 = box
        candidates = [
            (x0, y0 - box_h),          # 0: above, left-aligned (historical)
            (x0, y0),                  # 1: inside, top-left
            (x1 - box_w, y0 - box_h),  # 2: above, right-aligned
            (x1 - box_w, y0),          # 3: inside, top-right
            (x0, y1 - box_h),          # 4: inside, bottom-left
            (x1 - box_w, y1 - box_h),  # 5: inside, bottom-right
            (x0, y1),                  # 6: below, left-aligned
            (x1 - box_w, y1),          # 7: below, right-aligned
            (x0 - box_w, y0),          # 8: outside left
            (x1, y0),                  # 9: outside right
            (x0 - box_w, y1 - box_h),  # 10: outside left, bottom
            (x1, y1 - box_h),          # 11: outside right, bottom
        ]
        # Boxes stacked closer together than a label is tall exhaust the
        # anchored positions; step away from the box edge in label-height
        # increments, still column-aligned to the box so the association holds.
        for step in range(2, 6):
            candidates.append((x0, y0 - step * box_h))
            candidates.append((x0, y1 + (step - 2) * box_h))
        fallback = None
        for cx, cy in candidates:
            lx, ly = max(int(cx), 0), max(int(cy), 0)
            if fallback is None:
                fallback = (lx, ly)
            rect = (lx, ly, lx + box_w, ly + box_h)
            if not any(_rects_overlap(rect, other) for other in placed):
                return lx, ly
        return fallback

    def generate_prompt_for_box(
        self,
        box_index: int,
        class_list: Optional[List[str]] = None,
    ) -> str:
        """
        Generate a single-box identification prompt for PaliGemma with few-shot examples.

        Keeps the prompt short and direct. PaliGemma performs better with a
        simple, focused question than with long class lists or few-shot examples
        that can anchor completions to the example answers.

        Args:
            box_index: 1-based index of the bounding box to ask about.
            class_list: Unused (kept for API compatibility).

        Returns:
            Prompt string for a single box.
        """
        return f"What is the object inside box {box_index}?"

    def generate_prompt(
        self,
        class_list: Optional[List[str]] = None,
        num_objects: Optional[int] = None,
    ) -> str:
        """
        Generate the multi-box ROPE prompt (kept for reference / non-PaliGemma models).

        For PaliGemma use generate_prompt_for_box() instead, which queries
        one box at a time to match PaliGemma's pretrained completion style.

        Prompt template adapted from sled-group/moh evaluation.py.

        Args:
            class_list: Class names the model must choose from.
            num_objects: Number of bounding boxes in the image.

        Returns:
            Formatted prompt string.
        """
        classes = class_list if class_list is not None else self.class_list
        n = num_objects if num_objects is not None else self.num_objects

        classes_str = ", ".join(sorted(classes))
        obj_format = ", ".join(f"obj{i+1}: <class>" for i in range(n))

        return (
            f"There are {n} numbered red bounding boxes in this image, labeled 1 through {n}. "
            f"Choose the class of each numbered object from this list: {classes_str}. "
            f"Respond using ONLY this exact format with no extra words: {obj_format}."
        )

    def parse_response(
        self,
        response: str,
        num_objects: Optional[int] = None,
    ) -> List[str]:
        """
        Parse a multi-box model response into per-position class predictions.

        Expects the format: 'obj1: class1, obj2: class2, ...'
        Returns a list of length num_objects. Positions that cannot be parsed
        are filled with an empty string.

        For single-box responses (from generate_prompt_for_box), use
        parse_single_response() instead.

        Args:
            response: Raw model output string.
            num_objects: Expected number of objects. Falls back to self.num_objects.

        Returns:
            List of predicted class names, one per bounding box position.
        """
        n = num_objects if num_objects is not None else self.num_objects
        predictions = [""] * n

        # Match patterns like "obj1: cat" or "obj1:cat"
        matches = re.findall(r"obj(\d+)\s*:\s*([^,\n]+)", response, re.IGNORECASE)
        for idx_str, class_name in matches:
            idx = int(idx_str) - 1  # convert to 0-based
            if 0 <= idx < n:
                predictions[idx] = class_name.strip().lower()

        return predictions

    def parse_single_response(self, response: str) -> str:
        """
        Parse a single-box model response into a class name.

        Handles both terse outputs (PaliGemma: "spoon") and verbose
        outputs (Qwen: "The object inside box 1 is a wooden cabinet"), with
        markdown, quotes, and "appears to be" / "is pointing to" lead-ins
        removed ("Based on the image, **box 1** is pointing to a **cup**").

        Args:
            response: Raw model output string for a single-box query.

        Returns:
            Predicted class name string.
        """
        text = response.strip().lower().split("\n")[0].strip()
        # Formatting only: drop markdown emphasis/code marks and a leading
        # header or list marker. Matching itself stays exact.
        text = _LINE_PREFIX_RE.sub("", _MARKUP_RE.sub("", text)).strip()
        # Extract the noun phrase after echoed boilerplate ("is a/an/the",
        # "appears to be", "is pointing to"); a parenthetical aside ends it.
        m = _ANSWER_RE.search(text)
        if m:
            phrase = m.group(1).strip()
            phrase = re.sub(r"\s+(of|on|in|at|with|that|which|for)(\b.*)?$", "", phrase).strip()
            phrase = phrase.strip(_QUOTES)
            if phrase:
                return phrase
        return text.strip(" .,;:()" + _QUOTES)

    def evaluate_predictions(
        self,
        predictions: List[List[str]],
        ground_truth: List[List[str]],
        **kwargs,
    ) -> Dict[str, float]:
        """
        Evaluate pre-parsed predictions against ground truth.

        Use this when querying boxes one at a time (generate_prompt_for_box +
        parse_single_response), where you already have a list of predicted class
        names and don't need to re-parse a raw model string.

        Args:
            predictions: List of predicted class name lists, one per image.
                         Each inner list has length num_objects.
            ground_truth: List of ground truth class name lists, one per image.

        Returns:
            Dict with 'overall_accuracy' and per-position accuracies.
        """
        self._all_predictions.extend(predictions)
        self._all_ground_truth.extend(ground_truth)

        pairs = [
            (p, t)
            for preds, truths in zip(predictions, ground_truth)
            for p, t in zip(preds, truths)
        ]
        correct = sum(_rope_match(p, t) for p, t in pairs)
        correct_strict = sum(_rope_match_strict(p, t) for p, t in pairs)
        total = sum(len(p) for p in predictions)
        accuracy = correct / total if total > 0 else 0.0
        accuracy_strict = correct_strict / total if total > 0 else 0.0
        self.results = {
            "overall_accuracy": accuracy,
            "overall_accuracy_strict": accuracy_strict,
        }
        return self.results

    def evaluate(
        self,
        model_outputs: List[str],
        ground_truth: List[List[str]],
        **kwargs,
    ) -> Dict[str, float]:
        """
        Evaluate a batch of model responses against ground truth class lists.

        Parses each response string from the multi-box format (obj1: x, obj2: y, ...).
        For single-box querying use evaluate_predictions() instead.

        Args:
            model_outputs: List of raw model response strings, one per image.
            ground_truth: List of ground truth class name lists, one per image.
                          Each inner list has length num_objects.

        Returns:
            Dict with 'overall_accuracy' and per-position accuracies.
        """
        batch_predictions = [self.parse_response(r) for r in model_outputs]

        self._all_predictions.extend(batch_predictions)
        self._all_ground_truth.extend(ground_truth)

        # Compute batch-level accuracy (lenient headline + strict alongside)
        correct = 0
        correct_strict = 0
        total = 0
        for preds, truths in zip(batch_predictions, ground_truth):
            for p, t in zip(preds, truths):
                if _rope_match(p, t):
                    correct += 1
                if _rope_match_strict(p, t):
                    correct_strict += 1
                total += 1

        accuracy = correct / total if total > 0 else 0.0
        accuracy_strict = correct_strict / total if total > 0 else 0.0
        self.results = {
            "overall_accuracy": accuracy,
            "overall_accuracy_strict": accuracy_strict,
        }
        return self.results

    def format_results(self) -> Dict[str, Any]:
        """
        Compute and return comprehensive ROPE evaluation results.

        Returns:
            Dict containing:
                - overall_accuracy: Accuracy across all positions and images.
                - per_position_accuracy: Dict mapping 'obj1' ... 'objN' to accuracy.
                - precision: Macro-averaged precision (correct / predicted non-empty).
                - recall: Macro-averaged recall (correct / total ground truth).
                - f1_score: Harmonic mean of precision and recall.
                - num_images: Number of images evaluated.
                - num_objects_per_image: Configured num_objects.
        """
        if not self._all_predictions:
            return {
                "overall_accuracy": 0.0,
                "per_position_accuracy": {},
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "num_images": 0,
                "num_objects_per_image": self.num_objects,
            }

        n = self.num_objects
        per_position_correct = [0] * n
        per_position_correct_strict = [0] * n
        per_position_total = [0] * n
        total_correct = 0
        total_correct_strict = 0
        total_predicted_nonempty = 0
        total_gt = 0

        for preds, truths in zip(self._all_predictions, self._all_ground_truth):
            for pos, (p, t) in enumerate(zip(preds, truths)):
                if pos < n:
                    match = _rope_match(p, t)
                    match_strict = _rope_match_strict(p, t)
                    per_position_correct[pos] += int(match)
                    per_position_correct_strict[pos] += int(match_strict)
                    per_position_total[pos] += 1
                    total_correct += int(match)
                    total_correct_strict += int(match_strict)
                    total_gt += 1
                    if p.strip():
                        total_predicted_nonempty += 1

        overall_accuracy = total_correct / total_gt if total_gt > 0 else 0.0
        overall_accuracy_strict = total_correct_strict / total_gt if total_gt > 0 else 0.0
        precision = total_correct / total_predicted_nonempty if total_predicted_nonempty > 0 else 0.0
        recall = total_correct / total_gt if total_gt > 0 else 0.0
        f1 = self.compute_f1(precision, recall)

        per_position_accuracy = {}
        per_position_accuracy_strict = {}
        for pos in range(n):
            key = f"obj{pos + 1}"
            denom = per_position_total[pos]
            per_position_accuracy[key] = (
                per_position_correct[pos] / denom if denom > 0 else 0.0
            )
            per_position_accuracy_strict[key] = (
                per_position_correct_strict[pos] / denom if denom > 0 else 0.0
            )

        out = {
            # Headline metric uses lenient matching (substring-tolerant);
            # overall_accuracy_strict reports the canonical exact-match ROPE
            # number alongside it. See _rope_match / _rope_match_strict.
            "match_mode": "lenient",
            "overall_accuracy": overall_accuracy,
            "overall_accuracy_strict": overall_accuracy_strict,
            "per_position_accuracy": per_position_accuracy,
            "per_position_accuracy_strict": per_position_accuracy_strict,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "num_images": len(self._all_predictions),
            "num_objects_per_image": self.num_objects,
        }

        # Emit one extra accuracy column per additional matcher (e.g. a
        # SynonymMatcher). lenient/strict are already reported above, so only
        # matchers beyond those add keys; the default set adds nothing, keeping
        # the committed output byte-identical.
        extra = [m for m in getattr(self, "matchers", [])
                 if getattr(m, "name", None) not in ("lenient", "strict")]
        for m in extra:
            m_correct = [0] * n
            m_total = [0] * n
            m_overall_correct = 0
            m_overall_total = 0
            for preds, truths in zip(self._all_predictions, self._all_ground_truth):
                for pos, (p, t) in enumerate(zip(preds, truths)):
                    if pos < n:
                        hit = int(m.match(p, t))
                        m_correct[pos] += hit
                        m_total[pos] += 1
                        m_overall_correct += hit
                        m_overall_total += 1
            out[KEY_FOR.get(m.name, f"overall_accuracy_{m.name}")] = (
                m_overall_correct / m_overall_total if m_overall_total else 0.0
            )
            out[PER_POS_KEY_FOR.get(m.name, f"per_position_accuracy_{m.name}")] = {
                f"obj{i + 1}": (m_correct[i] / m_total[i] if m_total[i] else 0.0)
                for i in range(n)
            }
        return out

    def reset(self) -> None:
        """Reset accumulated prediction state."""
        super().reset()
        self._all_predictions = []
        self._all_ground_truth = []
