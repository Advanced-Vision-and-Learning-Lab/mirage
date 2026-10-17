"""Muse Glimmer (meta-models/Muse-Glimmer-*) through the generic chat wrapper."""
from PIL import Image

from .hf_chat import HFChatVLM


class MuseGlimmer(HFChatVLM):
    """Muse Glimmer: reasoning skipped, weights split across all visible GPUs.

    The chat template has no switch to turn reasoning off; after
    ``<|start|>assistant`` the model chooses the reasoning channel
    (``to=self``) or the answer channel (``to=user``). The prompt is prefilled
    with the answer channel, so the response is the answer alone. The date the
    template writes into the system turn is pinned so prompts do not change
    from day to day.

    The image processor asks for LANCZOS resizing but falls back to BICUBIC
    with torchvision < 0.27. Images are therefore resized here with PIL
    LANCZOS to the size the processor would pick, which makes its own resize a
    same-size pass (pixel values identical to ``do_resize=False``).
    """

    cache_name = "MuseGlimmerVLM"
    ANSWER_PREFIX = " to=user<|message|>"
    TEMPLATE_KWARGS = {"reasoning_strength": "low", "current_date": "2026-10-15"}

    def __init__(self, model_name: str, **kwargs):
        kwargs.setdefault("device_map", "auto")
        super().__init__(model_name, **kwargs)

    def preprocess_image(self, image):
        from transformers.models.muse_glimmer.image_processing_muse_glimmer import smart_resize

        image = super().preprocess_image(image)
        processor = self.processor.image_processor
        height, width = smart_resize(image.height, image.width,
                                     processor.patch_size * processor.merge_size, processor.max_image_tokens)
        if (width, height) == image.size:
            return image
        return image.resize((width, height), Image.LANCZOS)

    def _build_inputs(self, content):
        images = [part.pop("image") for part in content if part["type"] == "image"]
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            tokenize=False,
            **self.TEMPLATE_KWARGS,
        )
        return self.processor(
            text=[text + self.ANSWER_PREFIX],
            images=images or None,
            add_special_tokens=False,
            return_tensors="pt",
        )
