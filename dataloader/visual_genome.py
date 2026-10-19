"""
Visual Genome dataset dataloader implementation.

This module provides a dataloader for the Visual Genome dataset,
supporting object annotations from the objects.json file.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union
from PIL import Image

from .base import BaseDataLoader, ROPE_SAMPLING_SEED, sample_for_image

logger = logging.getLogger(__name__)


def _object_name(obj: Dict[str, Any]) -> str:
    """First name of a VG object ('names' list or 'name' string), stripped and lowercased; '' if none."""
    names = obj.get('names', [obj.get('name', '')])
    if isinstance(names, str):
        names = [names]
    return names[0].strip().lower() if names and names[0] else ''


class VisualGenomeDataLoader(BaseDataLoader):
    """
    DataLoader for Visual Genome dataset with object annotations.

    Handles loading of Visual Genome format data from local files.

    Expected data format (objects.json):
    [
        {
            "image_id": 2,
            "objects": [
                {
                    "object_id": 1023847,
                    "x": 405, "y": 34, "w": 78, "h": 438,
                    "name": "pole",
                    "synsets": ["pole.n.01"]
                }
            ]
        }
    ]

    Example:
        >>> loader = VisualGenomeDataLoader(
        ...     data_path="datasets/visual_genome/objects.json",
        ...     images_path="datasets/visual_genome/VG_100K"
        ... )
        >>> loader.load_data(max_samples=10)
        >>> for sample in loader:
        ...     print(sample['categories'])
    """

    KEY = "vg"

    def __init__(
        self,
        data_path: Union[str, Path],
        images_path: Union[str, Path],
        image_extensions: List[str] = None
    ):
        """
        Initialize Visual Genome dataloader.

        Args:
            data_path: Path to the objects.json annotation file
            images_path: Path to the directory containing image files (VG_100K)
            image_extensions: List of valid image extensions
        """
        super().__init__(data_path)
        self.images_path = Path(images_path)
        self.image_extensions = image_extensions or ['.jpg', '.jpeg', '.png']

        # Visual Genome data structures
        self.images: List[Dict[str, Any]] = []
        self.image_id_to_objects: Dict[int, List[Dict[str, Any]]] = {}
        self.all_object_names: set = set()
        self._image_mapping: Dict[str, Path] = {}

    def _build_image_mapping(self) -> None:
        """
        Build mapping from image filenames to actual file paths.

        Scans the images directory and creates a mapping for quick lookup.
        VG images are named by ID: {image_id}.jpg (e.g., "2.jpg").
        """
        logger.info(f"Building image mapping from {self.images_path}")
        self._image_mapping = {}

        if not self.images_path.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_path}")

        # Scan for all image files in the directory
        for ext in self.image_extensions:
            for image_file in self.images_path.glob(f"*{ext}"):
                filename = image_file.name
                self._image_mapping[filename] = image_file

        logger.info(f"Found {len(self._image_mapping)} images in {self.images_path}")

    def load_data(self, max_samples: Optional[int] = None) -> None:
        """
        Load the Visual Genome dataset from objects.json file.

        Args:
            max_samples: Maximum number of samples to load (None for all)

        Raises:
            FileNotFoundError: If the objects.json file doesn't exist
            json.JSONDecodeError: If the JSON file is malformed
        """
        if not self.validate_data_path():
            raise FileNotFoundError(f"Objects file not found: {self.data_path}")

        logger.info(f"Loading Visual Genome objects from {self.data_path}")

        # Build image mapping first
        self._build_image_mapping()

        # Load Visual Genome objects.json
        with open(self.data_path, 'r', encoding='utf-8') as f:
            vg_data = json.load(f)

        # Parse objects and build image_id -> objects mapping
        self.image_id_to_objects = {}
        self.all_object_names = set()

        for img_data in vg_data:
            image_id = img_data['image_id']
            objects = img_data.get('objects', [])

            if not objects:
                continue

            # Keep only named objects
            named_objects = []
            for obj in objects:
                obj_name = _object_name(obj)
                if not obj_name:
                    continue

                named_objects.append(obj)
                self.all_object_names.add(obj_name)

            if named_objects:
                self.image_id_to_objects[image_id] = named_objects

        logger.info(f"Loaded {len(self.image_id_to_objects)} images with objects")
        logger.info(f"Found {len(self.all_object_names)} unique object names")

        # Build images list - only include images with existing files and objects
        self.images = []
        self.skipped = 0

        for image_id in self.image_id_to_objects.keys():
            # Check if image file exists (VG images are named by ID)
            filename = f"{image_id}.jpg"
            if filename not in self._image_mapping:
                self.skipped += 1
                continue

            # Get image dimensions from file
            try:
                img_path = self._image_mapping[filename]
                with Image.open(img_path) as img:
                    width, height = img.size
            except Exception as e:
                logger.warning(f"Failed to load image {filename}: {e}")
                self.skipped += 1
                continue

            # Create image metadata
            img_meta = {
                'id': image_id,
                'file_name': filename,
                'width': width,
                'height': height
            }

            self.images.append(img_meta)

            # Stop if max_samples reached
            if max_samples and len(self.images) >= max_samples:
                logger.info(f"Reached max_samples limit of {max_samples}")
                break

        self._loaded = True

        if self.skipped > 0:
            logger.warning(f"{self.skipped} images skipped (files not found)")

        logger.info(f"Loaded {len(self.images)} valid image samples")

    def __len__(self) -> int:
        """
        Return the total number of image samples.

        Returns:
            int: Number of images in the dataset
        """
        if not self._loaded:
            self.load_data()
        return len(self.images)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Get a single image sample with its object annotations.

        Args:
            index: Index of the sample to retrieve

        Returns:
            Dict containing:
                - 'image_id': Visual Genome image ID
                - 'image_path': Path to the image file
                - 'image_filename': Image filename
                - 'width': Image width
                - 'height': Image height
                - 'objects': List of object annotation dicts for this image
                - 'categories': List of unique object names present in this image
                - 'index': Sample index
                - 'raw_data': Original image metadata
        """
        if not self._loaded:
            self.load_data()

        if index >= len(self.images):
            raise IndexError(f"Index {index} out of range for dataset size {len(self.images)}")

        img = self.images[index]
        image_id = img['id']
        filename = img['file_name']
        image_path = self._image_mapping[filename]

        # Get objects for this image
        objects = self.image_id_to_objects.get(image_id, [])

        # Extract unique category names (object names)
        category_names = []
        seen = set()
        for obj in objects:
            obj_name = _object_name(obj)
            if obj_name and obj_name not in seen:
                category_names.append(obj_name)
                seen.add(obj_name)

        return {
            'image_id': image_id,
            'image_path': str(image_path),
            'image_filename': filename,
            'width': img.get('width'),
            'height': img.get('height'),
            'objects': objects,
            'categories': category_names,
            'index': index,
            'raw_data': img
        }

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        Iterate over the dataset.

        Yields:
            Dict containing image data, object annotations, and category information
        """
        if not self._loaded:
            self.load_data()

        for i in range(len(self.images)):
            yield self[i]

    def load_image(self, sample: Dict[str, Any]) -> Image.Image:
        """
        Load the image associated with a sample.

        Args:
            sample: Sample dictionary from __getitem__

        Returns:
            PIL.Image: Loaded image
        """
        image_path = sample['image_path']
        return Image.open(image_path).convert('RGB')

    def extract_bboxes_for_rope(
        self,
        num_objects: int = 5,
        sample_indices: Union[List[int], None] = None,
        seed: Optional[int] = ROPE_SAMPLING_SEED,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract bounding boxes suitable for ROPE evaluation.

        Iterates object annotations directly. Images with fewer than num_objects
        objects are skipped. Images with more are sampled to exactly
        num_objects.

        VG bbox format (x, y, w, h) is converted to {xmin, ymin, xmax, ymax, name}.

        Args:
            num_objects: Required number of bounding boxes per image.
            sample_indices: Optional list of image indices to consider.
                            If None, all loaded images are considered.
            seed: Base seed for the per-image sampling RNG, so every run and
                  every model probes the same boxes. Set to None to use
                  process-global random.

        Returns:
            Dict mapping image_id (str) to a list of exactly num_objects bbox
            dicts, each with keys: xmin, ymin, xmax, ymax, name.
        """
        if not self._loaded:
            self.load_data()

        indices = sample_indices if sample_indices is not None else range(len(self.images))
        result: Dict[str, List[Dict[str, Any]]] = {}

        for idx in indices:
            img = self.images[idx]
            image_id = img['id']
            objects = self.image_id_to_objects.get(image_id, [])

            if len(objects) < num_objects:
                continue

            selected = sample_for_image(objects, num_objects, image_id, seed)

            bboxes = []
            for obj in selected:
                x = obj.get('x', 0)
                y = obj.get('y', 0)
                w = obj.get('w', 0)
                h = obj.get('h', 0)
                bboxes.append({
                    'xmin': int(x),
                    'ymin': int(y),
                    'xmax': int(x + w),
                    'ymax': int(y + h),
                    'name': _object_name(obj),
                })

            result[str(image_id)] = bboxes

        logger.info(f"extract_bboxes_for_rope: {len(result)} images with >= {num_objects} objects")
        return result

    def get_object_names(self) -> List[str]:
        """
        Get list of all object names in the dataset.

        Returns:
            List[str]: Sorted list of object names
        """
        if not self._loaded:
            self.load_data()

        return sorted(list(self.all_object_names))
