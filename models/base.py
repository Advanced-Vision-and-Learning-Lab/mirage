"""
Abstract base class for all models in the Mirage package.

This module defines the interface that all models must implement to ensure
consistent behavior across different model types.
"""

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, Optional, Union
from pathlib import Path
from PIL import Image
import logging


class BaseModel(ABC):
    """
    Abstract base class for all models in the Mirage package.

    This class defines the essential interface that all models must implement
    to provide consistent model inference capabilities across different model
    architectures.

    Attributes:
        model_name (str): Name or identifier of the model
        config (Dict[str, Any]): Model configuration parameters
        is_loaded (bool): Whether the model is currently loaded and ready
        _logger (logging.Logger): Logger instance for the model
    """

    # Model name in every prediction cache key. Pinned per family so moving or
    # renaming a class never changes keys; None falls back to the class name.
    cache_name: ClassVar[Optional[str]] = None

    def __init__(
        self,
        model_name: str,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize the base model.

        Args:
            model_name: Name or path of the model (e.g., 'google/paligemma-3b-pt-224')
            config: Dictionary containing model-specific configuration
            **kwargs: Additional keyword arguments for model initialization
        """
        self.model_name = model_name
        self.config = config or {}
        self.is_loaded = False

        # Initialize logger
        self._logger = logging.getLogger(f"{self.__class__.__name__}_{model_name}")

    @abstractmethod
    def load_model(self) -> None:
        """
        Load the model and prepare it for inference.

        This method should handle all necessary steps to prepare the model
        for inference, including:
        - Loading model weights
        - Initializing tokenizers/processors
        - Setting up any required configurations
        - Allocating GPU memory

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
        - Clearing GPU memory
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
        and multimodal (text + image) inputs.

        Args:
            prompt: Text prompt/question for the model
            image: Optional image input (PIL Image, file path, or None)
            **kwargs: Additional generation parameters (temperature, max_tokens, etc.)

        Returns:
            str: Generated response from the model

        Raises:
            NotImplementedError: If not implemented by subclass
            ModelNotLoadedError: If model is not loaded
        """
        pass

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


class ModelError(Exception):
    """Base exception class for model-related errors."""
    pass


class ModelLoadError(ModelError):
    """Exception raised when model loading fails."""
    pass


class ModelNotLoadedError(ModelError):
    """Exception raised when trying to use an unloaded model."""
    pass
