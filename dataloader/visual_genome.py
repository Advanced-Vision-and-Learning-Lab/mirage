"""
Visual Genome dataset dataloader implementation.

This module provides a dataloader for the Visual Genome dataset,
supporting object annotations from the objects.json file. Includes functionality
to download subsets from Hugging Face for efficient evaluation.
"""

import json
import logging
import random
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union, Tuple
from PIL import Image
from tqdm import tqdm

from .base import BaseDataLoader, ROPE_SAMPLING_SEED, sample_for_image

logger = logging.getLogger(__name__)


class VisualGenomeDataLoader(BaseDataLoader):
    """
    DataLoader for Visual Genome dataset with object annotations.

    Handles loading of Visual Genome format data from Hugging Face or local files.
    Supports downloading subsets and filtering by object names.

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
        >>> loader.download_individual_images(num_images=10)
        >>> loader.load_data(max_samples=10)
        >>> for sample in loader:
        ...     print(sample['categories'])
    """

    # Visual Genome base URL for image downloads
    VG_IMAGE_BASE_URL = 'https://cs.stanford.edu/people/rak248/VG_100K_2/'

    def __init__(
        self,
        data_path: Union[str, Path],
        images_path: Union[str, Path],
        batch_size: int = 1,
        shuffle: bool = False,
        image_extensions: List[str] = None,
        object_filter: Optional[List[str]] = None
    ):
        """
        Initialize Visual Genome dataloader.

        Args:
            data_path: Path to the objects.json annotation file
            images_path: Path to the directory containing image files (VG_100K)
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle the data
            image_extensions: List of valid image extensions
            object_filter: Optional list of object names to filter by
        """
        super().__init__(data_path, batch_size, shuffle)
        self.images_path = Path(images_path)
        self.image_extensions = image_extensions or ['.jpg', '.jpeg', '.png']
        self.object_filter = object_filter

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

    def download_dataset(self, output_dir: Optional[Union[str, Path]] = None, save_objects_json: bool = True) -> None:
        """
        Download Visual Genome objects annotations from the Visual Genome API.

        Args:
            output_dir: Directory to save downloaded files (uses current data_path parent if None)
            save_objects_json: Whether to save objects.json file from the dataset
        """
        output_dir = Path(output_dir) if output_dir else self.data_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        if save_objects_json:
            objects_json_path = output_dir / "objects.json"

            logger.info("Downloading Visual Genome objects annotations from visualgenome.org...")
            logger.info("This will download ~550MB of object annotations.")

            # Download objects.json from Visual Genome
            objects_url = 'https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/objects.json.zip'
            zip_path = output_dir / 'objects.json.zip'

            try:
                # Download the zip file
                logger.info(f"Downloading from {objects_url}...")
                urllib.request.urlretrieve(objects_url, zip_path)
                logger.info("Download complete")

                # Extract the JSON file
                logger.info("Extracting objects.json...")
                import zipfile
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(output_dir)

                logger.info(f"Extracted objects.json to {objects_json_path}")

                # Clean up zip file
                zip_path.unlink()
                logger.info("Cleaned up temporary files")

            except Exception as e:
                logger.error(f"Failed to download Visual Genome objects.json: {e}")
                logger.info("You can manually download from: https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/objects.json.zip")
                raise

        logger.info(f"Visual Genome annotations downloaded to {output_dir}")

    def download_individual_images(
        self,
        num_images: int = 10,
        output_dir: Optional[Union[str, Path]] = None,
        start_id: int = 1,
        download_annotations: bool = True
    ) -> None:
        """
        Download individual Visual Genome images without downloading the full dataset.
        Perfect for local development with small subsets.

        Args:
            num_images: Number of images to download
            output_dir: Directory to save files (uses current images_path if None)
            start_id: Starting image ID for sequential download
            download_annotations: Whether to download objects.json file if it doesn't exist

        Note:
            VG images are named by ID (e.g., 2.jpg, 3.jpg). This method attempts
            to download images sequentially starting from start_id.
        """
        output_dir = Path(output_dir) if output_dir else self.images_path
        output_dir.mkdir(parents=True, exist_ok=True)

        # Download annotations if requested and they don't exist
        if download_annotations:
            objects_json_path = self.data_path
            if not objects_json_path.exists():
                logger.info(f"objects.json not found at {objects_json_path}")
                logger.info("Downloading Visual Genome annotations from Hugging Face...")
                self.download_dataset(output_dir=self.data_path.parent, save_objects_json=True)

        logger.info(f"Downloading {num_images} individual images from Visual Genome")

        # Download individual images by ID
        downloaded = 0
        attempted = 0
        current_id = start_id

        # Try to download from both VG_100K and VG_100K_2
        base_urls = [
            'https://cs.stanford.edu/people/rak248/VG_100K/',
            'https://cs.stanford.edu/people/rak248/VG_100K_2/'
        ]

        logger.info(f"Downloading {num_images} images...")
        pbar = tqdm(total=num_images, desc="Downloading images")

        while downloaded < num_images and attempted < num_images * 3:
            filename = f"{current_id}.jpg"
            img_path = output_dir / filename

            if img_path.exists():
                logger.debug(f"Skipping {filename} (already exists)")
                downloaded += 1
                pbar.update(1)
                current_id += 1
                continue

            # Try both base URLs
            success = False
            for base_url in base_urls:
                img_url = base_url + filename

                try:
                    urllib.request.urlretrieve(img_url, img_path)
                    downloaded += 1
                    pbar.update(1)
                    success = True
                    break
                except Exception as e:
                    if img_path.exists():
                        img_path.unlink()
                    continue

            if not success:
                logger.debug(f"Failed to download image ID {current_id}")

            current_id += 1
            attempted += 1

        pbar.close()
        logger.info(f"Successfully downloaded {downloaded}/{num_images} images to {output_dir}")

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

            # Extract object names and apply filter
            filtered_objects = []
            for obj in objects:
                # Handle both 'names' (array) and 'name' (string) formats
                names = obj.get('names', [obj.get('name', '')])
                if isinstance(names, str):
                    names = [names]

                # Take first name if multiple names exist
                obj_name = names[0].strip().lower() if names and names[0] else ''
                if not obj_name:
                    continue

                # Apply object filter if specified
                if self.object_filter and obj_name not in self.object_filter:
                    continue

                filtered_objects.append(obj)
                self.all_object_names.add(obj_name)

            if filtered_objects:
                self.image_id_to_objects[image_id] = filtered_objects

        logger.info(f"Loaded {len(self.image_id_to_objects)} images with objects")
        logger.info(f"Found {len(self.all_object_names)} unique object names")

        # Build images list - only include images with existing files and objects
        self.images = []
        missing_images = 0
        no_objects = 0

        for image_id in self.image_id_to_objects.keys():
            # Check if image file exists (VG images are named by ID)
            filename = f"{image_id}.jpg"
            if filename not in self._image_mapping:
                missing_images += 1
                continue

            # Get image dimensions from file
            try:
                img_path = self._image_mapping[filename]
                with Image.open(img_path) as img:
                    width, height = img.size
            except Exception as e:
                logger.warning(f"Failed to load image {filename}: {e}")
                missing_images += 1
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

        if missing_images > 0:
            logger.warning(f"{missing_images} images skipped (files not found)")

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
            # Handle both 'names' (array) and 'name' (string) formats
            names = obj.get('names', [obj.get('name', '')])
            if isinstance(names, str):
                names = [names]

            # Take first name if multiple names exist
            obj_name = names[0].strip().lower() if names and names[0] else ''
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
            Dict containing dataset statistics including object distribution
        """
        if not self._loaded:
            self.load_data()

        # Count object occurrences
        object_counts = {}
        total_objects = 0

        for img_id, objects in self.image_id_to_objects.items():
            for obj in objects:
                # Handle both 'names' (array) and 'name' (string) formats
                names = obj.get('names', [obj.get('name', '')])
                if isinstance(names, str):
                    names = [names]
                obj_name = names[0].strip().lower() if names and names[0] else ''
                if obj_name:
                    object_counts[obj_name] = object_counts.get(obj_name, 0) + 1
                    total_objects += 1

        # Calculate images per object
        images_per_object = {}
        for img in self.images:
            img_id = img['id']
            objects = self.image_id_to_objects.get(img_id, [])
            objects_in_image = set()
            for obj in objects:
                # Handle both 'names' (array) and 'name' (string) formats
                names = obj.get('names', [obj.get('name', '')])
                if isinstance(names, str):
                    names = [names]
                obj_name = names[0].strip().lower() if names and names[0] else ''
                if obj_name:
                    objects_in_image.add(obj_name)

            for obj_name in objects_in_image:
                images_per_object[obj_name] = images_per_object.get(obj_name, 0) + 1

        return {
            **self.get_info(),
            'total_images': len(self.images),
            'total_unique_objects': len(self.all_object_names),
            'total_object_annotations': total_objects,
            'avg_objects_per_image': total_objects / len(self.images) if self.images else 0,
            'object_counts': object_counts,
            'images_per_object': images_per_object,
            'available_image_files': len(self._image_mapping)
        }

    def extract_objects(self, sample_indices: Union[List[int], None] = None) -> Tuple[Dict[str, List[str]], List[str]]:
        """
        Extract object names from the loaded dataset.

        This method extracts ground-truth object names from Visual Genome annotations,
        suitable for POPE (Polling-based Object Probing Evaluation) or similar metrics.

        Args:
            sample_indices: Optional list of indices to extract objects from.
                          If None, extracts from all samples.

        Returns:
            Tuple containing:
                - Dict mapping image_id -> list of object names in that image
                - List of all unique object names across all samples
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
        logger.info(f"Found {len(all_objects)} unique object names")

        # Show most common objects
        if sample_objects:
            object_frequency = {}
            for objects in sample_objects.values():
                for obj in objects:
                    object_frequency[obj] = object_frequency.get(obj, 0) + 1

            most_common = sorted(object_frequency.items(), key=lambda x: x[1], reverse=True)[:10]
            logger.info(f"Most frequent objects: {most_common[:5]}")

        return sample_objects, sorted(all_objects)

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

                # Resolve object name (VG uses 'names' list or 'name' string)
                names = obj.get('names', [obj.get('name', '')])
                if isinstance(names, str):
                    names = [names]
                obj_name = names[0].strip().lower() if names and names[0] else ''

                bboxes.append({
                    'xmin': int(x),
                    'ymin': int(y),
                    'xmax': int(x + w),
                    'ymax': int(y + h),
                    'name': obj_name,
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

    def filter_by_objects(self, object_names: List[str]) -> 'VisualGenomeDataLoader':
        """
        Create a new dataloader filtered to specific object names.

        Args:
            object_names: List of object names to filter by

        Returns:
            VisualGenomeDataLoader: New dataloader instance with object filter applied
        """
        return VisualGenomeDataLoader(
            data_path=self.data_path,
            images_path=self.images_path,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            image_extensions=self.image_extensions,
            object_filter=object_names
        )
