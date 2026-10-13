"""
Abstract base class for all dataloaders in the Mirage package.

This module defines the interface that all dataloaders must implement
to ensure consistent behavior across different datasets.
"""

import random
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, Optional, Union, List, Tuple
from pathlib import Path

ROPE_SAMPLING_SEED = 42


def sample_for_image(items: List[Any], k: int, image_id: Any, seed: Optional[int]) -> List[Any]:
    """
    Sample k items with an RNG derived from (seed, image_id).

    The per-image derivation makes the draw reproducible across runs and models
    and independent of how many images were processed before this one.

    Args:
        items: Candidate items to sample from
        k: Number of items to draw
        image_id: Stable identifier of the image the items belong to
        seed: Base RNG seed, or None to use process-global random

    Returns:
        List of k sampled items
    """
    rng = random.Random(f"{seed}:{image_id}") if seed is not None else random
    return rng.sample(items, k)


class BaseDataLoader(ABC):
    """
    Abstract base class for all dataloaders.

    This class defines the essential interface that all dataloaders must implement
    to provide consistent data loading capabilities across different datasets.

    Attributes:
        data_path (Path): Path to the dataset
        batch_size (int): Number of samples per batch
        shuffle (bool): Whether to shuffle the data
        _loaded (bool): Internal flag to track if data has been loaded
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        batch_size: int = 1,
        shuffle: bool = False
    ):
        """
        Initialize the dataloader.

        Args:
            data_path: Path to the dataset directory or file
            batch_size: Number of samples to load per batch
            shuffle: Whether to shuffle the data during iteration
        """
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._loaded = False

    @abstractmethod
    def load_data(self) -> None:
        """
        Load the dataset from the specified path.

        This method should handle the actual loading of data from files
        and prepare it for iteration. Must be implemented by subclasses.

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.

        Returns:
            int: Total number of samples

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Get a single sample from the dataset.

        Args:
            index: Index of the sample to retrieve

        Returns:
            Dict containing the sample data

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        Iterate over the dataset.

        Yields:
            Dict containing sample data

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        pass

    def validate_data_path(self) -> bool:
        """
        Validate that the data path exists and is accessible.

        Returns:
            bool: True if path is valid, False otherwise
        """
        return self.data_path.exists()

    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the dataset.

        Returns:
            Dict containing dataset metadata
        """
        return {
            'data_path': str(self.data_path),
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'loaded': self._loaded,
            'total_samples': len(self) if self._loaded else None
        }

    @abstractmethod
    def extract_objects(self, sample_indices: Union[List[int], None] = None) -> Tuple[Dict[str, List[str]], List[str]]:
        """
        Extract objects or classes from dataset samples.

        Args:
            sample_indices: Optional list of indices to process (None for all)

        Returns:
            Tuple containing:
                - Dict mapping sample_id -> list of objects in that sample
                - List of all unique objects across all samples
        """
        raise NotImplementedError("Each dataloader must implement extract_objects method")
