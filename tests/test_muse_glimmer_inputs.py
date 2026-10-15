"""Muse Glimmer prompt construction: answer channel prefilled, template kwargs pinned.

Uses a fake processor, so no checkpoint or GPU is needed (torch is imported).

Run:  python3 -m pytest tests/test_muse_glimmer_inputs.py -q
"""
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.hf_chat_vlm import MuseGlimmerVLM
from predict.factory import _REGISTRY


class FakeImageProcessor:
    patch_size, merge_size, max_image_tokens = 14, 2, 4096


class FakeProcessor:
    image_processor = FakeImageProcessor()

    def __init__(self):
        self.template_calls, self.calls = [], []

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append((messages, kwargs))
        return "<|start|>user<|message|><|patch|>Q<|eot|><|start|>assistant"

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return kwargs


def test_answer_channel_is_prefilled():
    model = MuseGlimmerVLM("unused", device="cpu")
    model.processor = FakeProcessor()
    image = Image.new("RGB", (28, 28))
    model._build_inputs([{"type": "image", "image": image}, {"type": "text", "text": "Q"}])

    messages, kwargs = model.processor.template_calls[0]
    assert kwargs["tokenize"] is False and kwargs["add_generation_prompt"] is True
    assert kwargs["reasoning_strength"] == "low" and kwargs["current_date"] == "2026-10-15"
    assert messages[0]["content"] == [{"type": "image"}, {"type": "text", "text": "Q"}]

    call = model.processor.calls[0]
    assert call["text"] == ["<|start|>user<|message|><|patch|>Q<|eot|><|start|>assistant to=user<|message|>"]
    assert call["images"] == [image]
    assert call["add_special_tokens"] is False


def test_weights_split_across_gpus_by_default():
    assert MuseGlimmerVLM("unused", device="cpu").device_map == "auto"
    assert _REGISTRY["muse-glimmer-30b"][1] == "MuseGlimmerVLM"


def test_image_is_lanczos_resized_to_the_processor_grid():
    model = MuseGlimmerVLM("unused", device="cpu")
    model.processor = FakeProcessor()
    # Same size the processor picks for a 375x500 COCO image (checked on a GPU node).
    assert model.preprocess_image(Image.new("RGB", (375, 500))).size == (364, 476)
    already = Image.new("RGB", (364, 476))
    assert model.preprocess_image(already).size == (364, 476)
