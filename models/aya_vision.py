"""Aya Vision (CohereLabs/aya-vision-*) through the generic chat wrapper."""
from .hf_chat import HFChatVLM


class AyaVision(HFChatVLM):
    """Aya Vision; keys its cache as ``HFChatVLM``."""
