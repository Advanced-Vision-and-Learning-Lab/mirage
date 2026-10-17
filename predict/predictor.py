"""Predictor: the single interface drivers use to obtain model outputs.

Sits between the experiment driver and the model. On a cache hit the
model is never called; on a miss the request is forwarded to the backend,
saved, and returned. ``generate`` is an alias of ``predict``, so a
Predictor drops in wherever a ``BaseModel`` object is used today.
"""

import json
import logging
import os
from typing import Any, Optional

from predict.backends import LocalModelBackend
from predict.cache import SQLiteResponseCache
from predict.identity import CACHE_KEY_VERSION, cache_key, hash_image

logger = logging.getLogger(__name__)


class Predictor:
    """Caching front-end for a prediction backend.

    Args:
        backend: A backend (see ``predict.backends``) or any
            ``BaseModel``-shaped object, wrapped automatically.
        cache: Cache instance; None builds the default SQLite cache. Pass
            ``SQLiteResponseCache(readonly=True)`` to serve hits without
            adding entries. ``MIRAGE_CACHE_DISABLE=1`` turns caching off.

    Backend weights load lazily: ``load_model()`` records the request and
    the real load happens on the first cache miss, so fully-cached runs
    never touch the GPU.
    """

    def __init__(self, backend: Any, cache: Optional[SQLiteResponseCache] = None):
        if not hasattr(backend, "identity"):
            backend = LocalModelBackend(backend)
        self.backend = backend
        if os.environ.get("MIRAGE_CACHE_DISABLE") == "1":
            self.cache = None
        else:
            self.cache = cache or SQLiteResponseCache()
        self.stats = {"hits": 0, "misses": 0, "bypassed": 0}
        self.model_key = None  # registry key, set by predict.factory.build_predictor
        self._load_requested = False

    def predict(self, prompt: str, image: Any = None, **generation_kwargs) -> str:
        """Return a prediction, from cache when possible.

        Args:
            prompt: Text prompt.
            image: Optional PIL image or image path.
            **generation_kwargs: Generation parameters forwarded to the backend.

        Returns:
            The generated (or cached) response text.
        """
        gen = self.backend.merged_generation_config(generation_kwargs)
        if self.cache is None or gen.get("do_sample"):
            self.stats["bypassed"] += 1
            self._ensure_backend_loaded()
            return self.backend.generate(prompt, image=image, **generation_kwargs)

        identity = self.backend.identity()
        image_hash = hash_image(image)
        key = cache_key(identity, prompt, image_hash, gen)
        cached = self.cache.get(key)
        if cached is not None:
            self.stats["hits"] += 1
            return cached

        if self.cache.readonly and not (self.backend.is_loaded or self._load_requested):
            raise RuntimeError(f"cache miss in cache-only mode (model never loaded): "
                               f"{identity.get('model_id')} revision {identity.get('revision')} "
                               f"not in {self.cache.db_path}")
        self._ensure_backend_loaded()
        response = self.backend.generate(prompt, image=image, **generation_kwargs)
        self.stats["misses"] += 1
        if response is not None and not self.cache.readonly:
            gen_json = json.dumps(gen, sort_keys=True)
            self.cache.put(key, response, meta=self._meta(identity, prompt, image_hash, gen_json))
        return response

    # BaseModel-compatible name for the same operation.
    generate = predict

    def load_model(self) -> None:
        """Request backend loading; the real load happens on the first miss."""
        self._load_requested = True

    def unload_model(self) -> None:
        """Unload the backend if it was loaded and log cache statistics."""
        self._load_requested = False
        if self.backend.is_loaded:
            self.backend.unload()
        logger.info("Predictor cache stats: %s", self.stats)

    def close(self) -> None:
        """Log cache statistics and close the cache connection."""
        logger.info("Predictor cache stats: %s", self.stats)
        if self.cache is not None:
            self.cache.close()

    @property
    def model_name(self) -> str:
        return self.backend.model_name

    def _ensure_backend_loaded(self) -> None:
        if self.backend.is_loaded:
            return
        if not self._load_requested:
            raise RuntimeError("model is not loaded; call load_model() first "
                               "(cache hits are served without it)")
        self.backend.load()

    def _meta(self, identity, prompt, image_hash, gen_json):
        return {
            "key_version": CACHE_KEY_VERSION,
            "provider": identity.get("provider"),
            "model_class": identity.get("class_name"),
            "model_id": identity.get("model_id"),
            "revision": identity.get("revision"),
            "quantization": identity.get("quantization"),
            "prompt": prompt,
            "image_hash": image_hash,
            "gen_config": gen_json,
        }
