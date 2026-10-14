"""DeepSeek-VL model wrapper for the Mirage evaluation interface."""

from pathlib import Path
from typing import List, Optional, Union

import torch
from PIL import Image

from .base import BaseModel, InvalidInputError, ModelLoadError


class DeepseekVLModel(BaseModel):
    """Adapter for the Transformers-native DeepSeek-VL 1.3B chat model."""

    def __init__(
        self,
        model_name: str = "deepseek-community/deepseek-vl-1.3b-chat",
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        quantization: Optional[str] = None,
        quantization_backend: str = "torchao",
        **kwargs,
    ):
        super().__init__(model_name, model_type="local", quantization=quantization, **kwargs)
        self.supports_images = True
        self.supports_text = True
        self.max_tokens = 4096
        self.model = None
        self.processor = None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype or torch.bfloat16
        self.quantization_backend = quantization_backend
        self.config = {
            "max_new_tokens": 512,
            "do_sample": False,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": None,
            **self.config,
        }

    def load_model(self) -> None:
        try:
            from transformers import AutoProcessor, DeepseekVLForConditionalGeneration

            self.processor = AutoProcessor.from_pretrained(self.model_name)
            model_kwargs = {"torch_dtype": self.torch_dtype}
            quantization_config = self.get_quantization_config(self.quantization_backend)
            if quantization_config is not None:
                model_kwargs["quantization_config"] = quantization_config

            if self.device != "cpu":
                try:
                    import accelerate  # noqa: F401

                    model_kwargs["device_map"] = self.device
                except ImportError:
                    self._logger.warning("accelerate is unavailable; using manual device placement")

            self.model = DeepseekVLForConditionalGeneration.from_pretrained(
                self.model_name, **model_kwargs
            )
            if "device_map" not in model_kwargs:
                self.model = self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
        except ImportError as error:
            raise ModelLoadError(
                "DeepSeek-VL requires a Transformers version with DeepseekVL support. "
                f"Install the repository requirements. Error: {error}"
            ) from error
        except Exception as error:
            raise ModelLoadError(f"Failed to load DeepSeek-VL: {error}") from error

    def unload_model(self) -> None:
        self.model = None
        self.processor = None
        if torch.cuda.is_available() and self.device != "cpu":
            torch.cuda.empty_cache()
        self.is_loaded = False

    def generate(
        self,
        prompt: str,
        image: Optional[Union[Image.Image, str, Path]] = None,
        **kwargs,
    ) -> str:
        is_valid, error = self.validate_input(prompt, image)
        if not is_valid:
            raise InvalidInputError(error)

        image = self.preprocess_image(image) if image is not None else None
        content = []
        if image is not None:
            content.append({"type": "image", "url": image})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(
                device=self.device,
                dtype=self.torch_dtype if value.is_floating_point() else value.dtype,
            )
            for key, value in inputs.items()
        }
        generation = {**self.config, **kwargs}
        try:
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=generation.get("max_new_tokens", 512),
                    do_sample=generation.get("do_sample", False),
                    temperature=generation.get("temperature", 0.7),
                    top_p=generation.get("top_p", 0.9),
                    top_k=generation.get("top_k"),
                    pad_token_id=(
                        self.processor.tokenizer.pad_token_id
                        or self.processor.tokenizer.eos_token_id
                    ),
                )
            input_length = inputs["input_ids"].shape[1]
            return self.processor.decode(
                output[0][input_length:], skip_special_tokens=True
            ).strip()
        except Exception as error:
            raise InvalidInputError(f"DeepSeek-VL generation failed: {error}") from error

    def batch_generate(
        self,
        prompts: List[str],
        images: Optional[List[Union[Image.Image, str, Path, None]]] = None,
        **kwargs,
    ) -> List[Optional[str]]:
        images = images or [None] * len(prompts)
        if len(prompts) != len(images):
            raise ValueError(f"got {len(prompts)} prompts but {len(images)} images")
        outputs = []
        for prompt, image in zip(prompts, images):
            try:
                outputs.append(self.generate(prompt, image=image, **kwargs))
            except Exception as error:
                self._logger.warning("DeepSeek-VL generation failed: %s", error)
                outputs.append(None)
        return outputs
