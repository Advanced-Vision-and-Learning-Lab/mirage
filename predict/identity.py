"""Deterministic identity hashing for the predict cache.

Cache keys are SHA-256 hashes over a sorted-keys JSON payload of the
model identity, the input image, the prompt, and the generation config.
Bump ``CACHE_KEY_VERSION`` whenever model post-processing or key
semantics change in a way that should invalidate existing cache entries
(same policy as ``matching.judge.PROMPT_VERSION``).
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

CACHE_KEY_VERSION = "1"

SUPPORTED_GENERATION_KEYS = ("max_new_tokens", "do_sample", "temperature", "top_p", "top_k")
SAMPLING_ONLY_KEYS = ("temperature", "top_p", "top_k")

_SNAPSHOT_RE = re.compile(r"models--([^/\\]+)--([^/\\]+)[/\\]snapshots[/\\]([0-9a-f]+)")
_HF_ID_RE = re.compile(r"[\w.\-]+/[\w.\-]+")


def clean_generation_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce a generation config to the keys that affect model output.

    Args:
        config: Merged generation config (model defaults + call kwargs).

    Returns:
        Dict with supported keys only, ``None`` values dropped,
        ``do_sample`` forced to bool, and sampling-only keys dropped
        under greedy decoding.
    """
    out = {k: config[k] for k in SUPPORTED_GENERATION_KEYS if config.get(k) is not None}
    out["do_sample"] = bool(config.get("do_sample", False))
    if not out["do_sample"]:
        for k in SAMPLING_ONLY_KEYS:
            out.pop(k, None)
    return out


def parse_model_name(model_name: str) -> Tuple[str, str]:
    """Split a model name or path into a stable ``(model_id, revision)`` pair.

    Args:
        model_name: HF id (``org/name``), HF cache snapshot path, or local dir.

    Returns:
        Tuple of model id and revision (``"unpinned"`` for plain HF ids,
        a path digest for local directories).
    """
    name = str(model_name)
    m = _SNAPSHOT_RE.search(name)
    if m:
        return f"{m.group(1)}/{m.group(2)}", m.group(3)
    path = Path(name)
    # A name that exists on disk is a local checkpoint even if it looks
    # like an HF id (e.g. "checkpoints/qwen-ft").
    if not path.exists() and _HF_ID_RE.fullmatch(name):
        return name, "unpinned"
    # Distinct local dirs must not share identity even with equal basenames.
    resolved = str(path.resolve()) if path.exists() else name
    path_digest = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:12]
    return path.name, f"path-{path_digest}"


def hash_image(image: Any) -> Optional[str]:
    """Hash an image by its RGB pixel bytes.

    The digest is memoized on the image object, so repeated questions
    about one image hash it only once.

    Args:
        image: PIL image, path to an image file, or None.

    Returns:
        SHA-256 hex digest, or None when no image is given.
    """
    if image is None:
        return None
    if isinstance(image, (str, Path)):
        from PIL import Image
        with Image.open(image) as im:
            return _hash_pixels(im)
    digest = getattr(image, "_predict_pixel_hash", None)
    if digest is None:
        digest = _hash_pixels(image)
        try:
            image._predict_pixel_hash = digest
        except AttributeError:
            pass
    return digest


def _hash_pixels(image) -> str:
    # Hash what the model consumes: models convert inputs to RGB first
    # (BaseModel.preprocess_image), so hashing the raw mode would give the
    # same picture different keys depending on how it was passed in.
    if image.mode != "RGB":
        image = image.convert("RGB")
    header = f"{image.width}x{image.height}|".encode("utf-8")
    return hashlib.sha256(header + image.tobytes()).hexdigest()


def cache_key(
    identity: Dict[str, Any],
    prompt: str,
    image_hash: Optional[str],
    gen_config: Dict[str, Any],
) -> str:
    """Build the deterministic cache key for one prediction request.

    Args:
        identity: Backend identity record (provider, model_id, revision, ...).
        prompt: Raw prompt text.
        image_hash: Result of :func:`hash_image`, or None.
        gen_config: Result of :func:`clean_generation_config`.

    Returns:
        SHA-256 hex digest.
    """
    payload = {
        "v": CACHE_KEY_VERSION,
        "model": identity,
        "image": image_hash,
        "prompt": prompt,
        "gen": gen_config,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
