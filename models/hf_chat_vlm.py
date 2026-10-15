"""Generic wrapper for Hugging Face chat vision-language models.

Covers models whose processor ships a chat template that accepts one image
and one text turn (Gemma 4, Ministral 3, Aya Vision). Decoding is greedy by
default and only the newly generated tokens are returned.
"""

import logging
import re
import time
from pathlib import Path
from typing import List, Optional, Union

import torch
from PIL import Image

from .base import BaseModel, ModelLoadError, ModelNotLoadedError

logger = logging.getLogger(__name__)

# Tokens that close a reasoning block (Gemma 4 uses a "thought" channel). They are
# special tokens, so the answer is cut at token level before decoding.
_THINKING_END_TOKENS = ("<channel|>", "</think>")
# Reasoning blocks written as plain text.
_THINKING_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class HFChatVLM(BaseModel):
    """Chat-template vision-language model loaded with ``AutoModelForImageTextToText``.

    Args:
        model_name: HF model id or local snapshot path.
        enable_thinking: Forwarded to the chat template; off for evaluation.
        thinking_tokens: With thinking on, extra new tokens added to
            ``max_new_tokens``; thought and answer share the combined budget.
        device: Device for inference (default: CUDA when available).
        torch_dtype: Weight dtype (default: bfloat16).
    """

    def __init__(
        self,
        model_name: str,
        enable_thinking: bool = False,
        thinking_tokens: int = 1024,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        super().__init__(model_name, model_type="local", **kwargs)
        self.supports_images = True
        self.enable_thinking = enable_thinking
        self.thinking_tokens = thinking_tokens if enable_thinking else 0
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype or torch.bfloat16
        self.model = None
        self.processor = None
        self.config = {"max_new_tokens": 512, "do_sample": False, **self.config}

    def load_model(self) -> None:
        start = time.monotonic()
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor

            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name, dtype=self.torch_dtype, device_map=self.device,
            ).eval()
            self._raw_logged = False
            vocab = getattr(self.processor.tokenizer, "get_vocab", dict)()
            self._thinking_end_ids = {vocab[t] for t in _THINKING_END_TOKENS if t in vocab}
        except Exception as e:
            raise ModelLoadError(f"Failed to load {self.model_name}: {e}") from e
        self.is_loaded = True
        logger.info("Loaded %s in %.1f s", self.model_name, time.monotonic() - start)

    def unload_model(self) -> None:
        if torch.cuda.is_available() and self.model is not None:
            logger.info("Peak CUDA memory: %.2f GB", torch.cuda.max_memory_allocated() / 1e9)
        self.model = None
        self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.is_loaded = False

    def generate(
        self,
        prompt: str,
        image: Optional[Union[Image.Image, str, Path]] = None,
        **kwargs,
    ) -> str:
        if not self.is_loaded:
            raise ModelNotLoadedError("Call load_model() before generate().")
        content = [{"type": "text", "text": prompt}]
        if image is not None:
            content.insert(0, {"type": "image", "image": self.preprocess_image(image)})
        inputs = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=self.enable_thinking,
        ).to(self.model.device)
        for key, value in inputs.items():
            if torch.is_floating_point(value):
                inputs[key] = value.to(self.torch_dtype)
        gen = {**self.config, **kwargs}
        gen_kwargs = {"max_new_tokens": gen["max_new_tokens"] + self.thinking_tokens,
                      "do_sample": bool(gen.get("do_sample"))}
        if gen_kwargs["do_sample"]:
            gen_kwargs.update({k: gen[k] for k in ("temperature", "top_p", "top_k") if gen.get(k) is not None})
        with torch.inference_mode():
            output = self.model.generate(**inputs, **gen_kwargs)
        new_tokens = output[0, inputs["input_ids"].shape[1]:].tolist()
        if not self._raw_logged:
            # One raw decode per load, so a run's log shows any thinking markers.
            logger.info("Raw decode: %r", self.processor.decode(new_tokens, skip_special_tokens=False))
            self._raw_logged = True
        ends = [i for i, token in enumerate(new_tokens) if token in self._thinking_end_ids]
        if ends:
            new_tokens = new_tokens[ends[-1] + 1:]
        elif self.enable_thinking and len(new_tokens) >= gen_kwargs["max_new_tokens"]:
            logger.warning("Thinking block not closed within %d tokens", gen_kwargs["max_new_tokens"])
        text = self.processor.decode(new_tokens, skip_special_tokens=True)
        return _THINKING_RE.sub("", text).strip()

    def batch_generate(
        self,
        prompts: List[str],
        images: Optional[List[Union[Image.Image, str, Path, None]]] = None,
        **kwargs,
    ) -> List[Optional[str]]:
        images = images or [None] * len(prompts)
        return [self.generate(prompt, image=image, **kwargs) for prompt, image in zip(prompts, images)]


class HFChatThinkingVLM(HFChatVLM):
    """``HFChatVLM`` with thinking on; a separate class so cache keys never mix modes."""

    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, enable_thinking=True, **kwargs)
