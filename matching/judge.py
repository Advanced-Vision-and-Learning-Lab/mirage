"""LLM synonym judge — DORMANT tier (not used by the default WordNet regrade).

Kept present-but-inactive so a stronger judge can be plugged in later without a
redesign. The issue #24 Phase-0 spike found flan-t5-base insufficient; issue #27
tracks ablating larger judges. Supports both architectures that ablation needs:

    arch="seq2seq"  -> AutoModelForSeq2SeqLM  (google/flan-t5-base, -xl)
    arch="causal"   -> AutoModelForCausalLM   (meta-llama/Llama-2-7b-chat-hf)

torch/transformers are lazy-imported inside ``_load`` so importing this module
(and therefore the whole ``matching`` package) stays torch-free.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

DEFAULT_PROMPTS = [
    'Do the phrases "{a}" and "{b}" refer to the same kind of object? Answer yes or no.',
    'Is a "{a}" the same thing as a "{b}"? Answer yes or no.',
    'Can "{a}" and "{b}" describe the same object in an image? Answer yes or no.',
]
# Bump when prompts/vote logic change so PairCache invalidates stale verdicts.
PROMPT_VERSION = "v1-3prompt-majority-greedy"


class LLMJudge:
    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        arch: str = "seq2seq",
        prompts: Optional[List[str]] = None,
        device: Optional[str] = None,
        batch_size: int = 32,
        max_new_tokens: int = 3,
    ):
        if arch not in ("seq2seq", "causal"):
            raise ValueError(f"arch must be 'seq2seq' or 'causal', got {arch!r}")
        self.model_name = model_name
        self.arch = arch
        self.prompts = prompts or DEFAULT_PROMPTS
        self.device = device
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._tok = None

    def _load(self) -> None:
        import torch
        from transformers import AutoTokenizer
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        if self.arch == "seq2seq":
            from transformers import AutoModelForSeq2SeqLM
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name, torch_dtype=dtype).to(self.device).eval()
        else:
            from transformers import AutoModelForCausalLM
            if self._tok.pad_token is None:
                self._tok.pad_token = self._tok.eos_token
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=dtype).to(self.device).eval()

    @staticmethod
    def _is_yes(text: str) -> bool:
        return text.strip().lower().startswith("y")

    def judge_batch(self, pairs: List[Tuple[str, str]]) -> List[Dict]:
        """Return [{"yes": bool, "votes": int}] — majority vote over prompts, greedy."""
        import torch
        if self._model is None:
            self._load()
        flat = [p.format(a=a, b=b) for (a, b) in pairs for p in self.prompts]
        outs: List[str] = []
        with torch.no_grad():
            for i in range(0, len(flat), self.batch_size):
                batch = flat[i:i + self.batch_size]
                enc = self._tok(batch, return_tensors="pt", padding=True,
                                truncation=True).to(self.device)
                gen = self._model.generate(**enc, max_new_tokens=self.max_new_tokens,
                                           do_sample=False)
                if self.arch == "causal":
                    gen = gen[:, enc["input_ids"].shape[1]:]  # strip the prompt
                outs.extend(self._tok.batch_decode(gen, skip_special_tokens=True))
        np_ = len(self.prompts)
        results = []
        for idx in range(len(pairs)):
            ans = outs[idx * np_:(idx + 1) * np_]
            votes = sum(self._is_yes(a) for a in ans)
            results.append({"yes": votes > np_ // 2, "votes": votes})
        return results


class FlanJudge(LLMJudge):
    """Convenience alias matching the issue #24 naming (seq2seq flan-t5-*)."""

    def __init__(self, model_name: str = "google/flan-t5-base", **kwargs):
        kwargs.pop("arch", None)
        super().__init__(model_name=model_name, arch="seq2seq", **kwargs)
