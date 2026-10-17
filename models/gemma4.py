"""Gemma 4 (google/gemma-4-*-it) through the generic chat wrapper."""
from .hf_chat import HFChatVLM


class Gemma4(HFChatVLM):
    """Gemma 4; keys its cache as ``HFChatVLM``."""


class Gemma4Thinking(Gemma4):
    """Gemma 4 with thinking on; a separate class so cache keys never mix modes."""

    cache_name = "HFChatThinkingVLM"

    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, enable_thinking=True, **kwargs)
