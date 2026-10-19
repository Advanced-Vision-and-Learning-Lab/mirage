"""Ministral 3 (mistralai/Ministral-3-*-Instruct-2512-BF16) through the generic chat wrapper."""
from .hf_chat import HFChatVLM


class Ministral3(HFChatVLM):
    """Ministral 3; keys its cache as ``HFChatVLM``."""
