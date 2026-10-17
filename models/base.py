"""
Abstract base class for all models in the Mirage package.

This module defines the interface that all models must implement to ensure
consistent behavior across different model types, whether they are API-based
or locally hosted models.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, Tuple, Literal
from pathlib import Path
from PIL import Image
import logging


class BaseModel(ABC):
    """
    Abstract base class for all models in the Mirage package.

    This class defines the essential interface that all models must implement
    to provide consistent model inference capabilities across different model
    architectures and deployment strategies (local vs API).

    The base class handles common functionality like:
    - Model metadata and configuration management
    - Input validation and preprocessing coordination
    - Output standardization
    - Error handling and logging
    - Resource management abstractions

    Attributes:
        model_name (str): Name or identifier of the model
        model_type (str): Type of model ('local' or 'api')
        config (Dict[str, Any]): Model configuration parameters
        is_loaded (bool): Whether the model is currently loaded and ready
        supports_images (bool): Whether the model can process images
        supports_text (bool): Whether the model can process text
        max_tokens (Optional[int]): Maximum tokens the model can handle
        quantization (Optional[str]): Quantization method ('int4', 'int8', 'none')
        _logger (logging.Logger): Logger instance for the model
    """

    def __init__(
        self,
        model_name: str,
        model_type: str = "local",
        config: Optional[Dict[str, Any]] = None,
        quantization: Optional[Literal["int4", "int8", "none"]] = None,
        **kwargs
    ):
        """
        Initialize the base model.

        Args:
            model_name: Name or path of the model (e.g., 'google/paligemma-3b-pt-224')
            model_type: Type of model deployment ('local' or 'api')
            config: Dictionary containing model-specific configuration
            quantization: Quantization method to use ('int4', 'int8', or 'none' for full precision)
            **kwargs: Additional keyword arguments for model initialization
        """
        self.model_name = model_name
        self.model_type = model_type.lower()
        self.config = config or {}
        self.is_loaded = False
        self.quantization = quantization

        # Model capabilities - to be set by subclasses
        self.supports_images = False
        self.supports_text = True
        self.max_tokens = None

        # Initialize logger
        self._logger = logging.getLogger(f"{self.__class__.__name__}_{model_name}")

        # Validate model type
        if self.model_type not in ['local', 'api']:
            raise ValueError(f"model_type must be 'local' or 'api', got '{model_type}'")

    @abstractmethod
    def load_model(self) -> None:
        """
        Load the model and prepare it for inference.

        This method should handle all necessary steps to prepare the model
        for inference, including:
        - Loading model weights/connecting to API
        - Initializing tokenizers/processors
        - Setting up any required configurations
        - Allocating GPU memory (for local models)

        The method should set self.is_loaded = True upon successful completion.

        Raises:
            NotImplementedError: If not implemented by subclass
            ModelLoadError: If model loading fails
        """
        pass

    @abstractmethod
    def unload_model(self) -> None:
        """
        Unload the model and free resources.

        This method should handle cleanup and resource deallocation:
        - Clearing GPU memory (for local models)
        - Closing API connections
        - Resetting internal state

        The method should set self.is_loaded = False upon completion.

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        image: Optional[Union[Image.Image, str, Path]] = None,
        **kwargs
    ) -> str:
        """
        Generate a response from the model.

        This is the main inference method that should handle both text-only
        and multimodal (text + image) inputs depending on model capabilities.

        Args:
            prompt: Text prompt/question for the model
            image: Optional image input (PIL Image, file path, or None)
            **kwargs: Additional generation parameters (temperature, max_tokens, etc.)

        Returns:
            str: Generated response from the model

        Raises:
            NotImplementedError: If not implemented by subclass
            ModelNotLoadedError: If model is not loaded
            InvalidInputError: If inputs are invalid for this model
        """
        pass

    def validate_input(
        self,
        prompt: str,
        image: Optional[Union[Image.Image, str, Path]] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate input before generation.

        This method performs common validation checks that apply to all models:
        - Prompt is not empty
        - Image is provided only if model supports images
        - Text-only input is provided if model requires it

        Args:
            prompt: Text prompt to validate
            image: Optional image input to validate

        Returns:
            Tuple[bool, Optional[str]]: (is_valid, error_message)
        """
        # Check if model is loaded
        if not self.is_loaded:
            return False, "Model is not loaded. Call load_model() first."

        # Check prompt
        if not prompt or not prompt.strip():
            return False, "Prompt cannot be empty."

        # Check image support
        if image is not None and not self.supports_images:
            return False, f"Model {self.model_name} does not support image inputs."

        # Check text support (should always be true, but included for completeness)
        if not self.supports_text:
            return False, f"Model {self.model_name} does not support text inputs."

        return True, None

    def preprocess_image(self, image: Union[Image.Image, str, Path]) -> Image.Image:
        """
        Preprocess image input into PIL Image format.

        This method handles common image preprocessing tasks:
        - Loading image from file path
        - Converting to RGB format
        - Basic validation

        Subclasses can override this for model-specific preprocessing.

        Args:
            image: Image input (PIL Image or file path)

        Returns:
            PIL.Image: Preprocessed image

        Raises:
            ValueError: If image cannot be processed
            FileNotFoundError: If image file doesn't exist
        """
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            image = Image.open(image_path)

        if not isinstance(image, Image.Image):
            raise ValueError(f"Expected PIL Image or file path, got {type(image)}")

        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        return image

    def __repr__(self) -> str:
        """String representation of the model."""
        status = "loaded" if self.is_loaded else "not loaded"
        return f"{self.__class__.__name__}(model_name='{self.model_name}', status='{status}')"

    def get_quantization_config(self, quantization_backend: str = "torchao") -> Optional[Any]:
        """
        Get the quantization configuration object based on the quantization setting.

        This method creates the appropriate quantization configuration for use with
        HuggingFace transformers models. Requires explicit specification of backend.

        Args:
            quantization_backend: Which backend to use ('torchao' or 'bitsandbytes')

        Returns:
            Quantization config object (TorchAoConfig or BitsAndBytesConfig) or None

        Raises:
            ImportError: If required quantization libraries are not installed
            ValueError: If unsupported quantization method or backend is specified
        """
        if self.quantization is None or self.quantization == "none":
            return None

        if quantization_backend not in ["torchao", "bitsandbytes"]:
            raise ValueError(
                f"Unsupported quantization backend: {quantization_backend}. "
                f"Supported backends: 'torchao', 'bitsandbytes'"
            )

        if quantization_backend == "torchao":
            if self.quantization == "int4":
                try:
                    from transformers import TorchAoConfig
                    self._logger.info("Using TorchAO int4_weight_only quantization")
                    return TorchAoConfig("int4_weight_only", group_size=128)
                except ImportError:
                    raise ImportError(
                        "TorchAO not available for int4 quantization. "
                        "Install with: pip install torchao"
                    )
            else:
                raise ValueError(
                    f"TorchAO backend only supports int4 quantization, got: {self.quantization}"
                )

        elif quantization_backend == "bitsandbytes":
            if self.quantization == "int4":
                import torch

                try:
                    from transformers import BitsAndBytesConfig
                    self._logger.info("Using BitsAndBytes 4-bit quantization")
                    return BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True
                    )
                except ImportError:
                    raise ImportError(
                        "BitsAndBytes not available for 4-bit quantization. "
                        "Install with: pip install bitsandbytes"
                    )
            elif self.quantization == "int8":
                try:
                    from transformers import BitsAndBytesConfig
                    self._logger.info("Using BitsAndBytes 8-bit quantization")
                    return BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0
                    )
                except ImportError:
                    raise ImportError(
                        "BitsAndBytes not available for 8-bit quantization. "
                        "Install with: pip install bitsandbytes"
                    )
            else:
                raise ValueError(
                    f"BitsAndBytes backend only supports int4/int8, got: {self.quantization}"
                )


class ModelError(Exception):
    """Base exception class for model-related errors."""
    pass


class ModelLoadError(ModelError):
    """Exception raised when model loading fails."""
    pass


class ModelNotLoadedError(ModelError):
    """Exception raised when trying to use an unloaded model."""
    pass


class InvalidInputError(ModelError):
    """Exception raised when model inputs are invalid."""
    pass
