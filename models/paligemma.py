"""
Paligemma model implementation for the Mirage package.

This module provides a wrapper for Google's Paligemma vision-language model,
specifically the paligemma-3b-mix-224 variant. Paligemma is a state-of-the-art
multimodal model that can understand and reason about images and text.
"""

import torch
from typing import Any, Dict, List, Optional, Union, Literal
from pathlib import Path
from PIL import Image

from .base import BaseModel, ModelLoadError, ModelNotLoadedError, InvalidInputError


class PaligemmaModel(BaseModel):
    """
    Paligemma vision-language model implementation.

    This class provides an interface to Google's Paligemma model, specifically
    the paligemma-3b-mix-224 variant. Paligemma is designed for vision-language
    tasks and can process both images and text to generate contextual responses.

    The model supports:
    - Image understanding and description
    - Visual question answering
    - Multimodal reasoning tasks
    - Image-conditioned text generation

    Attributes:
        model: The loaded Paligemma model
        processor: The associated processor for handling inputs
        device: PyTorch device for model execution
    """

    def __init__(
        self,
        model_name: str = "google/paligemma-3b-mix-224",
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        quantization: Optional[Literal["int4", "int8", "none"]] = None,
        quantization_backend: str = "torchao",
        use_true_batching: bool = False,
        **kwargs
    ):
        """
        Initialize Paligemma model.

        Args:
            model_name: HuggingFace model identifier for Paligemma
            device: Device to load model on ('cuda', 'cpu', or None for auto)
            torch_dtype: PyTorch data type for model weights (default: bfloat16)
            quantization: Quantization method ('int4', 'int8', 'none', or None)
            quantization_backend: Backend for quantization ('torchao' or 'bitsandbytes')
            use_true_batching: Whether to use true batch processing (default: False)
            **kwargs: Additional arguments passed to base class
        """
        super().__init__(
            model_name,
            model_type="local",
            quantization=quantization,
            use_true_batching=use_true_batching,
            **kwargs
        )

        # Set model capabilities
        self.supports_images = True
        self.supports_text = True
        self.max_tokens = 2048  # Typical max for Paligemma

        # Model-specific attributes
        self.model = None
        self.processor = None
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.torch_dtype = torch_dtype or torch.bfloat16
        self.quantization_backend = quantization_backend

        # Update config with Paligemma-specific defaults
        default_config = {
            "max_new_tokens": 512,
            "do_sample": False,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": None,
        }
        default_config.update(self.config)
        self.config = default_config

        self._logger.info(f"Initialized Paligemma model: {model_name} on {self.device}")

    def load_model(self) -> None:
        """
        Load the Paligemma model and processor.

        This method loads the model from HuggingFace Hub and prepares it
        for inference. It handles device placement, dtype conversion, and
        configurable quantization.

        Raises:
            ModelLoadError: If model loading fails
        """
        try:
            from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

            self._logger.info(f"Loading Paligemma model: {self.model_name}")
            self.processor = PaliGemmaProcessor.from_pretrained(self.model_name, use_fast=True)

            # Get quantization config from base class helper
            quantization_config = self.get_quantization_config(self.quantization_backend)

            # Prepare model loading arguments
            model_kwargs = {
                "torch_dtype": self.torch_dtype,
            }

            # Add quantization config if specified
            if quantization_config is not None:
                model_kwargs["quantization_config"] = quantization_config
                self._logger.info(f"Loading with {self.quantization} quantization using {self.quantization_backend}")

            # Handle device placement more carefully
            if self.device != "cpu" and self.device is not None:
                try:
                    # Try to use accelerate for device_map
                    import accelerate
                    model_kwargs["device_map"] = self.device
                except ImportError:
                    # Explicit error if accelerate is not available
                    self._logger.warning("accelerate not available, falling back to manual device placement")
                    model_kwargs["device_map"] = None

            # Load model with specified configuration
            self.model = PaliGemmaForConditionalGeneration.from_pretrained(
                self.model_name,
                **model_kwargs
            )

            # Manual device placement if accelerate wasn't used
            if "device_map" not in model_kwargs or model_kwargs["device_map"] is None:
                if self.device is not None:
                    self.model = self.model.to(self.device)

            # Set to evaluation mode
            self.model.eval()

            self.is_loaded = True
            self._logger.info(f"Successfully loaded Paligemma model on {self.device}")

        except ImportError as e:
            raise ModelLoadError(
                f"Failed to import required libraries for Paligemma. "
                f"Please install transformers: pip install transformers. Error: {e}"
            )
        except Exception as e:
            raise ModelLoadError(f"Failed to load Paligemma model: {e}")

    def unload_model(self) -> None:
        """
        Unload the Paligemma model and free GPU memory.
        """
        if self.model is not None:
            del self.model
            self.model = None

        if self.processor is not None:
            del self.processor
            self.processor = None

        # Clear GPU cache if using CUDA
        if torch.cuda.is_available() and self.device != "cpu":
            torch.cuda.empty_cache()

        self.is_loaded = False
        self._logger.info("Unloaded Paligemma model and freed resources")

    def generate(
        self,
        prompt: str,
        image: Optional[Union[Image.Image, str, Path]] = None,
        **kwargs
    ) -> str:
        """
        Generate a response using Paligemma.

        Args:
            prompt: Text prompt for the model
            image: Optional image input (required for meaningful responses)
            **kwargs: Generation parameters (max_new_tokens, temperature, etc.)

        Returns:
            str: Generated response from Paligemma

        Raises:
            ModelNotLoadedError: If model is not loaded
            InvalidInputError: If inputs are invalid
        """
        # Validate inputs
        is_valid, error_msg = self.validate_input(prompt, image)
        if not is_valid:
            raise InvalidInputError(error_msg)

        try:
            # Prepare inputs
            inputs = self._prepare_inputs(prompt, image)

            # Merge generation config
            generation_config = self.config.copy()
            generation_config.update(kwargs)

            # Generate response using helper
            outputs = self._generate_with_config(inputs, generation_config)

            # Decode response
            input_len = inputs["input_ids"].shape[1]
            response = self.processor.decode(
                outputs[0][input_len:],
                skip_special_tokens=True
            )

            return response.strip()

        except Exception as e:
            self._logger.error(f"Generation failed: {e}")
            raise InvalidInputError(f"Failed to generate response: {e}")

    def batch_generate(
        self,
        prompts: List[str],
        images: Optional[List[Union[Image.Image, str, Path, None]]] = None,
        batch_size: Optional[int] = None,
        **kwargs
    ) -> List[Optional[str]]:
        """
        Generate responses for a batch of inputs.

        Supports both sequential processing (default) and true batch processing
        based on the `use_true_batching` setting. When true batching is enabled,
        processes multiple samples in parallel for improved efficiency.

        Args:
            prompts: List of text prompts
            images: Optional list of images (can contain None values)
            batch_size: Size of batches for true batching (only used if use_true_batching=True)
            **kwargs: Generation parameters

        Returns:
            List[Optional[str]]: One entry per prompt; None where generation failed

        Raises:
            ModelNotLoadedError: If model is not loaded
            InvalidInputError: If inputs are invalid
        """
        if not self.is_loaded:
            raise ModelNotLoadedError("Model is not loaded")

        if images is None:
            images = [None] * len(prompts)

        if len(prompts) != len(images):
            raise InvalidInputError("Number of prompts and images must match")

        # Sequential processing (original behavior)
        if not self.use_true_batching:
            responses = []
            for prompt, image in zip(prompts, images):
                try:
                    response = self.generate(prompt, image, **kwargs)
                    responses.append(response)
                except Exception as e:
                    self._logger.warning(f"Failed to generate for prompt '{prompt[:50]}...': {e}")
                    responses.append(None)
            return responses

        # True batch processing
        return self._batch_generate_parallel(prompts, images, batch_size, **kwargs)

    def _generate_with_config(
        self,
        inputs: Dict[str, torch.Tensor],
        generation_config: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Generate outputs using model.generate with the given config.

        This helper method centralizes the generation logic to avoid duplication
        between single and batch generation methods.

        Args:
            inputs: Prepared model inputs
            generation_config: Generation configuration parameters

        Returns:
            torch.Tensor: Model outputs
        """
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=generation_config.get("max_new_tokens", 256),
                do_sample=generation_config.get("do_sample", False),
                temperature=generation_config.get("temperature", 1.0),
                top_p=generation_config.get("top_p", 1.0),
                top_k=generation_config.get("top_k"),
                pad_token_id=self.processor.tokenizer.eos_token_id
            )
        return outputs

    def _batch_generate_parallel(
        self,
        prompts: List[str],
        images: Optional[List[Union[Image.Image, str, Path, None]]],
        batch_size: Optional[int] = None,
        **kwargs
    ) -> List[Optional[str]]:
        """
        Generate responses using true batch processing.

        This method processes multiple inputs in parallel batches for improved
        efficiency compared to sequential processing.

        Args:
            prompts: List of text prompts
            images: List of images (or None values)
            batch_size: Size of batches (None = all at once)
            **kwargs: Generation parameters

        Returns:
            List[Optional[str]]: One entry per prompt; None where generation failed
        """
        all_responses = []

        # Use base class helper to split into batches
        batches = self.batch_inputs(prompts, images, batch_size)

        for batch_prompts, batch_images in batches:
            try:
                # Prepare batch inputs
                batch_inputs = self._prepare_batch_inputs(batch_prompts, batch_images)

                # Merge generation config
                generation_config = self.config.copy()
                generation_config.update(kwargs)

                # Generate for batch using helper
                outputs = self._generate_with_config(batch_inputs, generation_config)

                # Decode all outputs in batch
                input_len = batch_inputs["input_ids"].shape[1]
                for output in outputs:
                    response = self.processor.decode(
                        output[input_len:],
                        skip_special_tokens=True
                    )
                    all_responses.append(response.strip())

            except Exception as e:
                self._logger.error(f"Batch generation failed: {e}")
                all_responses.extend([None] * len(batch_prompts))

        return all_responses

    def _prepare_batch_inputs(
        self,
        prompts: List[str],
        images: Optional[List[Union[Image.Image, str, Path, None]]]
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for batch processing.

        Args:
            prompts: List of text prompts
            images: List of images (or None values)

        Returns:
            Dict containing processed batch inputs for the model
        """
        # Preprocess all images and add <image> tokens to prompts
        processed_images = []
        processed_prompts = []

        for prompt, image in zip(prompts, images):
            if image is not None:
                processed_images.append(self.preprocess_image(image))
                # Add <image> token prefix if not already present
                if not prompt.startswith("<image>"):
                    processed_prompts.append(f"<image>{prompt}")
                else:
                    processed_prompts.append(prompt)
            else:
                processed_images.append(None)
                processed_prompts.append(prompt)

        # Use processor to prepare batch inputs
        inputs = self.processor(
            text=processed_prompts,
            images=processed_images if any(img is not None for img in processed_images) else None,
            return_tensors="pt",
            padding=True
        )

        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        return inputs

    def _prepare_inputs(
        self,
        prompt: str,
        image: Optional[Union[Image.Image, str, Path]]
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for the Paligemma model.

        Args:
            prompt: Text prompt
            image: Optional image input

        Returns:
            Dict containing processed inputs for the model
        """
        # Process image if provided
        if image is not None:
            image = self.preprocess_image(image)
            # Add <image> token prefix if not already present
            if not prompt.startswith("<image>"):
                prompt = f"<image>{prompt}"

        # Use processor to prepare inputs
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding=True
        )

        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        return inputs

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get Paligemma-specific model information.

        Returns:
            Dict containing model metadata and current state
        """
        info = super().get_model_info()
        info.update({
            'device': self.device,
            'torch_dtype': str(self.torch_dtype),
            'model_architecture': 'PaliGemma',
            'input_resolution': '224x224',
            'parameter_count': '3B'
        })
        return info
