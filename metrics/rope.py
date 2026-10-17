"""
ROPE (Recognition-based Object Probing Evaluation) metric.

Adapted from the sled-group/moh repository:
  https://github.com/sled-group/moh

ROPE evaluates multi-object hallucination in Vision-Language Models by drawing
red bounding boxes around N objects in an image and asking the model to name
the object in each numbered box, one question per box. This contrasts with
POPE's binary yes/no questioning and tests whether VLMs can correctly ground
and classify multiple objects in the same image.

Reference:
    Zou et al., "Look, Compare, Decide: Alleviating Hallucination in Large
    Vision-Language Models via Multi-View Multi-Path Reasoning"
    https://github.com/sled-group/moh
"""

import re
import logging
from copy import deepcopy
from types import MappingProxyType
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont

from .base_metric import SAMPLES, BaseMetric

logger = logging.getLogger(__name__)


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


# Judged ROPE: the answer key the judge graded. A first noun phrase, not the
# strict extractor above; changing it orphans the stored verdicts.
_JUDGE_QUANT = (r"(?:a\s+|an\s+|the\s+)?(?:small\s+|large\s+)?(?:piece|pieces|slice|slices|part|parts|pair|pairs|"
                r"group|groups|stack|stacks|section|set|bunch|collection|pile|row|rows|cluster|portion|close-up|"
                r"closeup|corner|edge|side|top|bottom|view|image|picture|photo)\s+of\s+(?:the\s+|a\s+|an\s+|some\s+)?")
_JUDGE_LEAD = re.compile(r"(?:box\s*\d+\s*(?:contains|shows|has)?|is|appears\s+to\s+be|seems\s+to\s+be|looks\s+like|"
                         r"is\s+likely|is\s+pointing\s+to)\s+(?:an?\s+|the\s+|some\s+)?(.+)")
_JUDGE_ARTICLES = {"a", "an", "the", "some", "two", "three", "several"}


def _judge_norm(name: str) -> str:
    from .chair import _singularize

    toks = re.findall(r"[a-z]+", name.lower())
    while toks and toks[0] in _JUDGE_ARTICLES:
        toks = toks[1:]
    if not toks:
        return ""
    toks[-1] = _singularize(toks[-1])
    return " ".join(toks)


def judged_answer(response: str, pred: str) -> str:
    """Answer key for a judged ROPE verdict: the first noun run of the first
    alternative ("X or Y" -> X) after lead-ins and partitives are removed,
    normalized; ``pred`` (the strict extraction) when no noun is found."""
    import nltk

    line = re.sub(r"[*_`#>\"“”]", "", response.strip()[:400].strip().split("\n")[0].lower()).strip()
    m = next(_JUDGE_LEAD.finditer(line), None)
    text = re.split(r"[.;:(—]|\s-\s", m.group(1) if m else line)[0]
    alts = []
    for a in re.split(r"\s+or\s+|,\s*|\s+and\s+", text):
        a = re.sub("^" + _JUDGE_QUANT, "", a.strip())
        a = re.sub(r"\s+(?:of|on|in|at|with|that|which|for|from|near|behind|under)\b.*$", "", a)
        noun = []
        for w, t in (nltk.pos_tag(nltk.word_tokenize(a)) if a else []):
            if t.startswith("NN"):
                noun.append(w)
            elif noun:
                break
        if noun:
            alts.append(" ".join(noun))
    return _judge_norm((alts or [pred])[0]) or pred

def _rects_overlap(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    """True when two (x0, y0, x1, y1) rectangles share any area."""
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


class ROPEMetric(BaseMetric):
    """
    Recognition-based Object Probing Evaluation (ROPE) metric.

    Presents the model with an image annotated with red bounding boxes and asks
    it to name the object in each box, one question per box. Tracks
    per-position and overall exact-match accuracy (plain ROPE).

    Adapted from: https://github.com/sled-group/moh
    """

    POOL = MappingProxyType({"coco": 3000, "vg": 2000})
    GEN = MappingProxyType({"max_new_tokens": 30, "do_sample": False})

    def __init__(
        self,
        num_objects: int = 5,
        bbox_color: str = "red",
        bbox_width: int = 3,
    ):
        """
        Initialize the ROPE metric.

        Args:
            num_objects: Number of bounding boxes / objects per image.
            bbox_color: Color of drawn bounding boxes (default red).
            bbox_width: Line width for bounding box outlines.
        """
        super().__init__("rope")
        self.num_objects = num_objects
        self.bbox_color = bbox_color
        self.bbox_width = bbox_width

        # Accumulated state across evaluate_predictions() calls
        self._all_predictions: List[List[str]] = []
        self._all_ground_truth: List[List[str]] = []

    def draw_bounding_boxes(
        self,
        image: Image.Image,
        bboxes: List[Dict[str, Any]],
    ) -> Image.Image:
        """
        Draw numbered red bounding boxes on a copy of the image.

        Each box is labeled with its 1-based index (1, 2, 3, ...), matching
        the "box i" in the per-box prompts.

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

    def generate_prompt_for_box(self, box_index: int) -> str:
        """
        The single-box ROPE question; its bytes are part of the cache key.

        Args:
            box_index: 1-based index of the bounding box to ask about.

        Returns:
            Prompt string for a single box.
        """
        return f"What is the object inside box {box_index}?"

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

    def run(self, loader, model, n: int = SAMPLES) -> List[Tuple[Any, int, str, str]]:
        """Ask about each numbered box of the first ``n`` images that have enough boxes, then evaluate.

        Each image gets ``num_objects`` boxes drawn on it and one question per
        box. Read the scores with :meth:`format_results`.

        Returns:
            ``(image_id, box, response, ground_truth)`` per box (box is 1-based).
        """
        boxes = loader.extract_bboxes_for_rope(num_objects=self.num_objects)
        predictions, ground_truth, records = [], [], []
        for sample in [s for s in loader if str(s["image_id"]) in boxes][:n]:
            image_boxes = boxes[str(sample["image_id"])]
            image = self.draw_bounding_boxes(loader.load_image(sample), image_boxes)
            responses = [model.generate(image=image, prompt=self.generate_prompt_for_box(i),
                                        **self.GEN) for i in range(1, self.num_objects + 1)]
            predictions.append([self.parse_single_response(r) for r in responses])
            ground_truth.append([b["name"] for b in image_boxes])
            records += [(sample["image_id"], i, r, gt) for i, (r, gt) in enumerate(zip(responses, ground_truth[-1]), 1)]
        if not predictions:
            raise RuntimeError("No images had enough object boxes for ROPE.")
        self.evaluate_predictions(predictions, ground_truth)
        return records

    @staticmethod
    def strict(pred: str, gt: str) -> bool:
        """Plain ROPE: the parsed answer equals the label, ignoring case and surrounding whitespace."""
        return pred.strip().lower() == gt.strip().lower()

    def evaluate_predictions(
        self,
        predictions: List[List[str]],
        ground_truth: List[List[str]],
        **kwargs,
    ) -> Dict[str, float]:
        """
        Score parsed answers (from parse_single_response) against ground truth with :meth:`strict`.

        Args:
            predictions: List of predicted class name lists, one per image.
                         Each inner list has length num_objects.
            ground_truth: List of ground truth class name lists, one per image.

        Returns:
            Dict with 'overall_accuracy_strict' for this batch.
        """
        self._all_predictions.extend(predictions)
        self._all_ground_truth.extend(ground_truth)

        pairs = [
            (p, t)
            for preds, truths in zip(predictions, ground_truth)
            for p, t in zip(preds, truths)
        ]
        correct_strict = sum(self.strict(p, t) for p, t in pairs)
        total = sum(len(p) for p in predictions)
        self.results = {"overall_accuracy_strict": correct_strict / total if total > 0 else 0.0}
        return self.results

    def format_results(self) -> Dict[str, Any]:
        """
        Compute and return the plain ROPE results over every evaluated image.

        Returns:
            Dict containing:
                - overall_accuracy_strict: Exact-match accuracy across all positions and images.
                - per_position_accuracy_strict: Dict mapping 'obj1' ... 'objN' to accuracy.
                - num_images: Number of images evaluated.
                - num_objects_per_image: Configured num_objects.
        """
        n = self.num_objects
        if not self._all_predictions:
            return {
                "overall_accuracy_strict": 0.0,
                "per_position_accuracy_strict": {},
                "num_images": 0,
                "num_objects_per_image": n,
            }

        per_position_correct_strict = [0] * n
        per_position_total = [0] * n
        for preds, truths in zip(self._all_predictions, self._all_ground_truth):
            for pos, (p, t) in enumerate(zip(preds, truths)):
                if pos < n:
                    per_position_correct_strict[pos] += int(self.strict(p, t))
                    per_position_total[pos] += 1
        total_gt = sum(per_position_total)

        return {
            "overall_accuracy_strict": sum(per_position_correct_strict) / total_gt if total_gt > 0 else 0.0,
            "per_position_accuracy_strict": {
                f"obj{pos + 1}": per_position_correct_strict[pos] / denom if denom > 0 else 0.0
                for pos, denom in enumerate(per_position_total)
            },
            "num_images": len(self._all_predictions),
            "num_objects_per_image": n,
        }

    def reset(self) -> None:
        """Reset accumulated prediction state."""
        super().reset()
        self._all_predictions = []
        self._all_ground_truth = []
