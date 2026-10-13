"""
COCO dataset dataloader implementation.

This module provides a dataloader for the COCO (Common Objects in Context) dataset,
supporting both images and their annotations with category labels. Includes functionality
to download subsets of the dataset for efficient evaluation.
"""

import json
import logging
import random
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union, Tuple
from PIL import Image
from tqdm import tqdm

from .base import BaseDataLoader, ROPE_SAMPLING_SEED, sample_for_image

logger = logging.getLogger(__name__)


class COCODataLoader(BaseDataLoader):
    """
    DataLoader for COCO dataset with support for annotations and subset downloading.

    Handles loading of COCO format data with images and category annotations.
    Supports downloading specific subsets and filtering by categories.

    Expected annotation format (COCO JSON):
    {
        "images": [{"id": 1, "file_name": "000000001.jpg", ...}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, ...}],
        "categories": [{"id": 1, "name": "person", "supercategory": "person"}]
    }
    """

    # COCO dataset URLs
    COCO_URLS = {
        'train2017_images': 'http://images.cocodataset.org/zips/train2017.zip',
        'val2017_images': 'http://images.cocodataset.org/zips/val2017.zip',
        'test2017_images': 'http://images.cocodataset.org/zips/test2017.zip',
        'train2017_annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
        'val2017_annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
    }

    def __init__(
        self,
        data_path: Union[str, Path],
        images_path: Union[str, Path],
        batch_size: int = 1,
        shuffle: bool = False,
        image_extensions: List[str] = None,
        category_filter: Optional[List[str]] = None
    ):
        """
        Initialize COCO dataloader.

        Args:
            data_path: Path to the COCO annotation JSON file
            images_path: Path to the directory containing image files
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle the data
            image_extensions: List of valid image extensions
            category_filter: Optional list of category names to filter by
        """
        super().__init__(data_path, batch_size, shuffle)
        self.images_path = Path(images_path)
        self.image_extensions = image_extensions or ['.jpg', '.jpeg', '.png']
        self.category_filter = category_filter

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

    def download_individual_images(
        self,
        split: str = 'val2017',
        num_images: int = 10,
        output_dir: Optional[Union[str, Path]] = None,
        download_annotations: bool = True
    ) -> None:
        """
        Download individual COCO images without downloading the full zip.
        Perfect for local development with small subsets.

        Args:
            split: Dataset split ('train2017', 'val2017', 'test2017')
            num_images: Number of images to download
            output_dir: Directory to save files (uses current data_path parent if None)
            download_annotations: Whether to download annotation file

        Raises:
            ValueError: If invalid split is specified
        """
        if split not in ['train2017', 'val2017', 'test2017']:
            raise ValueError(f"Invalid split: {split}. Must be one of: train2017, val2017, test2017")

        output_dir = Path(output_dir) if output_dir else self.data_path.parent
        images_dir = output_dir / split
        images_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading {num_images} individual images from COCO {split}")

        # Download annotations first to know which images to get
        if download_annotations:
            anno_url = self.COCO_URLS[f'{split}_annotations']
            anno_zip = output_dir / f'annotations_{split}.zip'
            anno_json = output_dir / 'annotations' / f'instances_{split}.json'

            if not anno_json.exists():
                logger.info(f"Downloading annotations from {anno_url}")
                self._download_file(anno_url, anno_zip)

                logger.info("Extracting annotations...")
                with zipfile.ZipFile(anno_zip, 'r') as zip_ref:
                    zip_ref.extractall(output_dir)
                logger.info(f"Annotations extracted to {output_dir / 'annotations'}")
            else:
                logger.info(f"Annotations already exist at {anno_json}")

            # Load annotations to get image list
            with open(anno_json, 'r') as f:
                coco_data = json.load(f)

            # Filter to images that actually have annotations
            images_with_annotations = set(anno['image_id'] for anno in coco_data['annotations'])
            annotated_images = [img for img in coco_data['images'] if img['id'] in images_with_annotations]
            image_list = annotated_images[:num_images]
        else:
            # Generate sequential filenames if no annotations
            logger.warning("Downloading without annotations - using sequential filenames")
            image_list = [{'file_name': f'{i:012d}.jpg'} for i in range(1, num_images + 1)]

        # Download individual images
        base_url = f'http://images.cocodataset.org/{split}/'
        downloaded = 0

        logger.info(f"Downloading {len(image_list)} images...")
        for img_info in tqdm(image_list, desc="Downloading images"):
            filename = img_info['file_name']
            img_path = images_dir / filename

            if img_path.exists():
                logger.debug(f"Skipping {filename} (already exists)")
                downloaded += 1
                continue

            img_url = base_url + filename

            try:
                urllib.request.urlretrieve(img_url, img_path)
                downloaded += 1
            except Exception as e:
                logger.warning(f"Failed to download {filename}: {e}")
                continue

        logger.info(f"Successfully downloaded {downloaded}/{len(image_list)} images to {images_dir}")

    def download_subset(
        self,
        split: str = 'val2017',
        download_images: bool = True,
        download_annotations: bool = True,
        max_images: Optional[int] = None,
        output_dir: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Download a subset of the COCO dataset.

        Note: For small subsets (< 100 images), consider using download_individual_images()
        instead, as this method downloads the full zip file (~20GB) and then extracts a subset.

        Args:
            split: Dataset split to download ('train2017', 'val2017', 'test2017')
            download_images: Whether to download images
            download_annotations: Whether to download annotations
            max_images: Maximum number of images to extract (None for all)
            output_dir: Directory to save downloaded files (uses current data_path parent if None)

        Raises:
            ValueError: If invalid split is specified
        """
        if split not in ['train2017', 'val2017', 'test2017']:
            raise ValueError(f"Invalid split: {split}. Must be one of: train2017, val2017, test2017")

        # Recommend individual download for small subsets
        if max_images is not None and max_images < 100:
            logger.info(f"For small subsets ({max_images} images), consider using download_individual_images()")
            logger.info("to avoid downloading the full ~20GB zip file.")

        output_dir = Path(output_dir) if output_dir else self.data_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading COCO {split} to {output_dir}")

        # Download annotations
        if download_annotations:
            anno_url = self.COCO_URLS[f'{split}_annotations']
            anno_zip = output_dir / f'annotations_{split}.zip'

            if not anno_zip.exists():
                logger.info(f"Downloading annotations from {anno_url}")
                self._download_file(anno_url, anno_zip)

                # Extract annotations
                logger.info("Extracting annotations...")
                with zipfile.ZipFile(anno_zip, 'r') as zip_ref:
                    zip_ref.extractall(output_dir)
                logger.info(f"Annotations extracted to {output_dir}")
            else:
                logger.info(f"Annotations already exist at {anno_zip}")

        # Download images
        if download_images:
            img_url = self.COCO_URLS[f'{split}_images']
            img_zip = output_dir / f'{split}.zip'

            if not img_zip.exists():
                logger.info(f"Downloading images from {img_url}")
                logger.warning("This will download the full dataset (~20GB).")

                self._download_file(img_url, img_zip)

                # Extract images
                logger.info("Extracting images...")
                with zipfile.ZipFile(img_zip, 'r') as zip_ref:
                    members = zip_ref.namelist()

                    # If max_images specified, only extract that many
                    if max_images is not None:
                        members = members[:max_images + 1]  # +1 for potential directory entry

                    for member in tqdm(members, desc="Extracting"):
                        zip_ref.extract(member, output_dir)

                logger.info(f"Images extracted to {output_dir / split}")
            else:
                logger.info(f"Images already exist at {img_zip}")

    def _download_file(self, url: str, destination: Path) -> None:
        """
        Download a file with progress bar.

        Args:
            url: URL to download from
            destination: Path to save the file
        """
        logger.info(f"Downloading from {url} to {destination}")
        urllib.request.urlretrieve(url, destination)
        logger.info("Download complete")

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

        # Apply category filter if specified
        if self.category_filter:
            filtered_cat_ids = {
                cat_id for cat_id, cat in self.categories.items()
                if cat['name'] in self.category_filter
            }
            logger.info(f"Filtering to {len(filtered_cat_ids)} categories: {self.category_filter}")
        else:
            filtered_cat_ids = None

        # Parse annotations and build image_id -> annotations mapping
        self.annotations = coco_data.get('annotations', [])
        self.image_id_to_annotations = {}

        for anno in self.annotations:
            image_id = anno['image_id']

            # Skip if category filter is active and this annotation doesn't match
            if filtered_cat_ids and anno['category_id'] not in filtered_cat_ids:
                continue

            if image_id not in self.image_id_to_annotations:
                self.image_id_to_annotations[image_id] = []
            self.image_id_to_annotations[image_id].append(anno)

        logger.info(f"Loaded {len(self.annotations)} annotations")

        # Parse images and filter to those with available image files and annotations
        all_images = coco_data.get('images', [])
        self.images = []
        missing_images = 0
        no_annotations = 0

        for img in all_images:
            # Check if image file exists
            if img['file_name'] not in self._image_mapping:
                missing_images += 1
                continue

            # Check if image has annotations (after filtering)
            if img['id'] not in self.image_id_to_annotations:
                no_annotations += 1
                continue

            self.images.append(img)

            # Stop if max_samples reached
            if max_samples and len(self.images) >= max_samples:
                logger.info(f"Reached max_samples limit of {max_samples}")
                break

        self._loaded = True

        if missing_images > 0:
            logger.warning(f"{missing_images} images skipped (files not found)")
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

        indices = list(range(len(self.images)))
        if self.shuffle:
            random.shuffle(indices)

        # Handle batching
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]

            if self.batch_size == 1:
                # Return single item for batch_size=1
                yield self[batch_indices[0]]
            else:
                # Return batch of items
                batch = [self[idx] for idx in batch_indices]
                yield batch

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

    def get_stats(self) -> Dict[str, Any]:
        """
        Get dataset statistics.

        Returns:
            Dict containing dataset statistics including category distribution
        """
        if not self._loaded:
            self.load_data()

        # Count category occurrences
        category_counts = {}
        total_annotations = 0

        for img_id, annos in self.image_id_to_annotations.items():
            for anno in annos:
                cat_id = anno['category_id']
                if cat_id in self.categories:
                    cat_name = self.categories[cat_id]['name']
                    category_counts[cat_name] = category_counts.get(cat_name, 0) + 1
                    total_annotations += 1

        # Calculate images per category
        images_per_category = {}
        for img in self.images:
            img_id = img['id']
            annos = self.image_id_to_annotations.get(img_id, [])
            categories_in_image = set(anno['category_id'] for anno in annos)

            for cat_id in categories_in_image:
                if cat_id in self.categories:
                    cat_name = self.categories[cat_id]['name']
                    images_per_category[cat_name] = images_per_category.get(cat_name, 0) + 1

        return {
            **self.get_info(),
            'total_images': len(self.images),
            'total_categories': len(self.categories),
            'total_annotations': total_annotations,
            'avg_annotations_per_image': total_annotations / len(self.images) if self.images else 0,
            'category_counts': category_counts,
            'images_per_category': images_per_category,
            'available_image_files': len(self._image_mapping)
        }

    def extract_objects(self, sample_indices: Union[List[int], None] = None) -> Tuple[Dict[str, List[str]], List[str]]:
        """
        Extract object categories from the loaded dataset.

        This method extracts ground-truth object categories from COCO annotations,
        suitable for POPE (Polling-based Object Probing Evaluation) or similar metrics.

        Args:
            sample_indices: Optional list of indices to extract objects from.
                          If None, extracts from all samples.

        Returns:
            Tuple containing:
                - Dict mapping image_id -> list of category names in that image
                - List of all unique category names across all samples
        """
        if not self._loaded:
            self.load_data()

        sample_objects = {}
        all_objects = set()

        indices = sample_indices if sample_indices is not None else range(len(self.images))

        for idx in indices:
            sample = self[idx]
            image_id = str(sample['image_id'])
            categories = sample['categories']

            if categories:
                sample_objects[image_id] = categories
                all_objects.update(categories)

        logger.info(f"Extracted objects from {len(sample_objects)} samples")
        logger.info(f"Found {len(all_objects)} unique object categories")

        # Show most common categories
        if sample_objects:
            category_frequency = {}
            for objects in sample_objects.values():
                for obj in objects:
                    category_frequency[obj] = category_frequency.get(obj, 0) + 1

            most_common = sorted(category_frequency.items(), key=lambda x: x[1], reverse=True)[:10]
            logger.info(f"Most frequent categories: {most_common[:5]}")

        return sample_objects, sorted(all_objects)

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

    def filter_by_categories(self, category_names: List[str]) -> 'COCODataLoader':
        """
        Create a new dataloader filtered to specific categories.

        Args:
            category_names: List of category names to filter by

        Returns:
            COCODataLoader: New dataloader instance with category filter applied
        """
        return COCODataLoader(
            data_path=self.data_path,
            images_path=self.images_path,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            image_extensions=self.image_extensions,
            category_filter=category_names
        )
