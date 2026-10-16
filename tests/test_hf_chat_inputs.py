"""HFChatVLM prompt construction is unchanged by the ``_build_inputs`` hook.

Drives ``generate`` with a fake processor and model and checks that the chat
template gets exactly the messages and kwargs the wrapper passed before the
hook existed. No checkpoint or GPU is needed (torch is imported).

Run:  python3 -m pytest tests/test_hf_chat_inputs.py -q
"""
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.hf_chat_vlm import HFChatThinkingVLM, HFChatVLM


class FakeInputs(dict):
    def to(self, device):
        self.device = device
        return self


class FakeProcessor:
    def __init__(self):
        self.template_calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append((messages, kwargs))
        return FakeInputs(input_ids=torch.tensor([[1, 2, 3]]), pixel_values=torch.zeros(1, 3, dtype=torch.float32))

    def decode(self, tokens, skip_special_tokens=True):
        return "yes"


class FakeModel:
    device = "cpu"

    def generate(self, input_ids, **kwargs):
        return torch.cat([input_ids, torch.tensor([[7]])], dim=1)


def _run(model, image):
    model.processor, model.model = FakeProcessor(), FakeModel()
    model.is_loaded, model._raw_logged, model._thinking_end_ids = True, False, set()
    assert model.generate("Is there a dog?", image=image) == "yes"
    return model.processor.template_calls


@pytest.mark.parametrize("cls, thinking", [(HFChatVLM, False), (HFChatThinkingVLM, True)])
def test_template_call_matches_pre_hook_wrapper(cls, thinking):
    image = Image.new("RGB", (28, 28))
    (messages, kwargs), = _run(cls("unused", device="cpu"), image)
    assert messages == [{"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": "Is there a dog?"}]}]
    assert kwargs == {"add_generation_prompt": True, "tokenize": True, "return_dict": True,
                      "return_tensors": "pt", "enable_thinking": thinking}


def test_text_only_prompt_has_no_image_part():
    (messages, _), = _run(HFChatVLM("unused", device="cpu"), None)
    assert messages[0]["content"] == [{"type": "text", "text": "Is there a dog?"}]


def test_default_device_map_is_the_device():
    assert HFChatVLM("unused", device="cpu").device_map is None
