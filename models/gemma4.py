"""Gemma 4 (google/gemma-4-*-it) through the generic chat wrapper."""
from .hf_chat import HFChatVLM


class Gemma4(HFChatVLM):
    """Gemma 4; keys its cache as ``HFChatVLM``."""
