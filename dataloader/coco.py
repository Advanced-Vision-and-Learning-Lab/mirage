"""
COCO dataset dataloader implementation.

This module provides a dataloader for the COCO (Common Objects in Context) dataset,
supporting both images and their annotations with category labels.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union
from PIL import Image

from .base import BaseDataLoader, ROPE_SAMPLING_SEED, sample_for_image

logger = logging.getLogger(__name__)


class COCODataLoader(BaseDataLoader):
    """
    DataLoader for COCO dataset with category annotations.

    Handles loading of COCO format data with images and category annotations.

    Expected annotation format (COCO JSON):
    {
        "images": [{"id": 1, "file_name": "000000001.jpg", ...}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, ...}],
        "categories": [{"id": 1, "name": "person", "supercategory": "person"}]
    }
    """

    KEY = "coco"

    def __init__(
        self,
        data_path: Union[str, Path],
        images_path: Union[str, Path],
        image_extensions: List[str] = None
    ):
        """
        Initialize COCO dataloader.

        Args:
            data_path: Path to the COCO annotation JSON file
            images_path: Path to the directory containing image files
            image_extensions: List of valid image extensions
        """
        super().__init__(data_path)
        self.images_path = Path(images_path)
        self.image_extensions = image_extensions or ['.jpg', '.jpeg', '.png']

        # COCO data structures
        self.images: List[Dict[str, Any]] = []
        self.annotations: List[Dict[str, Any]] = []
        self.categories: Dict[int, Dict[str, Any]] = {}
        self.image_id_to_annotations: Dict[int, List[Dict[str, Any]]] = {}
        self._image_mapping: Dict[str, Path] = {}

    def _build_image_mapping(self) -> None:
        """
        Build mapping from image filenames to actual file paths.

        Scans the images directory and creates a mapping for quick lookup.
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
        Load the COCO dataset from annotation JSON file.

        Args:
            max_samples: Maximum number of samples to load (None for all)

        Raises:
            FileNotFoundError: If the annotation file doesn't exist
            json.JSONDecodeError: If the JSON file is malformed
        """
        if not self.validate_data_path():
            raise FileNotFoundError(f"Annotation file not found: {self.data_path}")

        logger.info(f"Loading COCO annotations from {self.data_path}")

        # Build image mapping first
        self._build_image_mapping()

        # Load COCO annotations
        with open(self.data_path, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)

        # Parse categories
        self.categories = {cat['id']: cat for cat in coco_data.get('categories', [])}
        logger.info(f"Loaded {len(self.categories)} categories")

        # Parse annotations and build image_id -> annotations mapping
        self.annotations = coco_data.get('annotations', [])
        self.image_id_to_annotations = {}

        for anno in self.annotations:
            image_id = anno['image_id']
            if image_id not in self.image_id_to_annotations:
                self.image_id_to_annotations[image_id] = []
            self.image_id_to_annotations[image_id].append(anno)

        logger.info(f"Loaded {len(self.annotations)} annotations")

        # Parse images and filter to those with available image files and annotations
        all_images = coco_data.get('images', [])
        self.images = []
        self.skipped = 0
        no_annotations = 0

        for img in all_images:
            # Check if image file exists
            if img['file_name'] not in self._image_mapping:
                self.skipped += 1
                continue

            # Check if image has annotations
            if img['id'] not in self.image_id_to_annotations:
                no_annotations += 1
                continue

            self.images.append(img)

            # Stop if max_samples reached
            if max_samples and len(self.images) >= max_samples:
                logger.info(f"Reached max_samples limit of {max_samples}")
                break

        self._loaded = True

        if self.skipped > 0:
            logger.warning(f"{self.skipped} images skipped (files not found)")
        if no_annotations > 0:
            logger.warning(f"{no_annotations} images skipped (no matching annotations)")

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
        Get a single image sample with its annotations.

        Args:
            index: Index of the sample to retrieve

        Returns:
            Dict containing:
                - 'image_id': COCO image ID
                - 'image_path': Path to the image file
                - 'image_filename': Image filename
                - 'width': Image width
                - 'height': Image height
                - 'annotations': List of annotation dicts for this image
                - 'categories': List of category names present in this image
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

        # Get annotations for this image
        annotations = self.image_id_to_annotations.get(image_id, [])

        # Extract category names
        category_names = sorted(set(
            self.categories[anno['category_id']]['name']
            for anno in annotations
            if anno['category_id'] in self.categories
        ))

        return {
            'image_id': image_id,
            'image_path': str(image_path),
            'image_filename': filename,
            'width': img.get('width'),
            'height': img.get('height'),
            'annotations': annotations,
            'categories': category_names,
            'index': index,
            'raw_data': img
        }

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        Iterate over the dataset.

        Yields:
            Dict containing image data, annotations, and category information
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

        Iterates annotations directly. Images with fewer than num_objects
        distinct annotations are skipped. Images with more than num_objects
        annotations are sampled down to exactly num_objects.

        COCO bbox format [x, y, width, height] is converted to
        {xmin, ymin, xmax, ymax, name}.

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
            annotations = self.image_id_to_annotations.get(image_id, [])

            if len(annotations) < num_objects:
                continue

            selected = sample_for_image(annotations, num_objects, image_id, seed)

            bboxes = []
            for anno in selected:
                x, y, w, h = anno['bbox']
                cat_name = self.categories.get(anno['category_id'], {}).get('name', '')
                bboxes.append({
                    'xmin': int(x),
                    'ymin': int(y),
                    'xmax': int(x + w),
                    'ymax': int(y + h),
                    'name': cat_name,
                })

            result[str(image_id)] = bboxes

        logger.info(f"extract_bboxes_for_rope: {len(result)} images with >= {num_objects} annotations")
        return result

    def get_category_names(self) -> List[str]:
        """
        Get list of all category names in the dataset.

        Returns:
            List[str]: Sorted list of category names
        """
        if not self._loaded:
            self.load_data()

        return sorted([cat['name'] for cat in self.categories.values()])
