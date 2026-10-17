"""Backends a :class:`~predict.predictor.Predictor` can route requests to.

Routing is decided by which backend the Predictor is constructed with —
never by branching inside the Predictor and never inside ``BaseModel``.
A backend provides: ``generate``, ``identity()``,
``merged_generation_config(kwargs)``, ``load``/``unload``, ``is_loaded``,
and ``model_name``. A future API backend implements the same surface with
an identity like ``{"provider": "gemini", "model_id": "models/gemini-2.5-flash"}``;
``BaseModel`` keeps sole responsibility for local loading and generation.
"""

from typing import Any, Callable, Dict, List, Optional

from predict.identity import clean_generation_config, parse_model_name


class LocalModelBackend:
    """Adapts a ``BaseModel``-shaped object to the backend surface.

    Args:
        model: Any object exposing ``generate``, ``load_model``,
            ``unload_model``, ``model_name``, ``config``, and ``is_loaded``
            (duck-typed; no torch import here).
    """

    def __init__(self, model: Any):
        self.model = model
        self._identity: Optional[Dict[str, Any]] = None

    def generate(self, prompt, image=None, **kwargs):
        return self.model.generate(prompt=prompt, image=image, **kwargs)

    def identity(self) -> Dict[str, Any]:
        if self._identity is None:
            model_id, revision = parse_model_name(self.model.model_name)
            self._identity = {
                "provider": "local",
                # Pinned by the model class (BaseModel.cache_name) so the key
                # survives moves and renames; plain objects use their class name.
                "class_name": getattr(self.model, "cache_name", None) or type(self.model).__name__,
                "model_id": model_id,
                "revision": revision,
                "quantization": getattr(self.model, "quantization", None) or "none",
                # dtype changes outputs, so it is part of the key; device is
                # deliberately left out so a cache written on a GPU node can
                # be replayed on any other machine.
                "dtype": str(getattr(self.model, "torch_dtype", None)),
            }
        return self._identity

    def merged_generation_config(self, kwargs):
        return clean_generation_config({**self.model.config, **kwargs})

    def load(self):
        self.model.load_model()

    def unload(self):
        self.model.unload_model()

    @property
    def is_loaded(self):
        return self.model.is_loaded

    @property
    def model_name(self):
        return self.model.model_name


class CallableBackend:
    """Wraps a text-only batch function, e.g. a metric's evaluator LM.

    The callable owns any torch machinery; this module never imports it.

    Args:
        fn: ``fn(prompts: List[str]) -> List[str]``.
        identity: Stable identity record; must include ``model_id`` and
            should include ``provider``.
        defaults: Generation defaults merged under call kwargs for keying.
        loader: Optional callable that loads the model behind ``fn``; when
            given, the Predictor's lazy loading applies — the load runs on
            the first cache miss, not before.
        unloader: Optional callable that frees the model behind ``fn``.
    """

    def __init__(
        self,
        fn: Callable[[List[str]], List[str]],
        identity: Dict[str, Any],
        defaults: Optional[Dict[str, Any]] = None,
        loader: Optional[Callable[[], None]] = None,
        unloader: Optional[Callable[[], None]] = None,
    ):
        self._fn = fn
        self._identity = dict(identity)
        self._identity.setdefault("provider", "local-callable")
        self._defaults = dict(defaults or {})
        self._loader = loader
        self._unloader = unloader
        self._loaded = loader is None

    def generate(self, prompt, image=None, **kwargs):
        if image is not None:
            raise ValueError("CallableBackend is text-only")
        self._reject_conflicting_kwargs(kwargs)
        return self._fn([prompt])[0]

    def _reject_conflicting_kwargs(self, kwargs):
        # fn owns its generation config; kwargs it cannot honor must not be
        # silently recorded in cache keys/metadata as if they were applied.
        conflicts = {k: v for k, v in kwargs.items() if self._defaults.get(k, v) != v}
        if conflicts:
            raise ValueError(f"CallableBackend cannot honor generation kwargs {conflicts}; "
                             f"its callable is fixed to {self._defaults}")

    def identity(self) -> Dict[str, Any]:
        return dict(self._identity)

    def merged_generation_config(self, kwargs):
        return clean_generation_config({**self._defaults, **kwargs})

    def load(self):
        if self._loader is not None:
            self._loader()
        self._loaded = True

    def unload(self):
        if self._unloader is not None:
            self._unloader()
        self._loaded = self._loader is None

    @property
    def is_loaded(self):
        return self._loaded

    @property
    def model_name(self):
        return self._identity.get("model_id", "callable")
