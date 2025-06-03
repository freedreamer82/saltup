"""
COCO Dataset Utilities
=====================

This module provides utilities for handling COCO format annotations.

COCO Format Overview:
--------------------
Single JSON file per dataset with structure:
{
    "images": [{"id": int, "file_name": str, "width": int, "height": int}],
    "annotations": [{"image_id": int, "bbox": [x,y,w,h], "category_id": int}],
    "categories": [{"id": int, "name": str}]
}

Key functions:
- Reading/writing COCO annotations
- Dataset organization and validation
- Statistics and analysis
- Dataset splitting
"""

import os
import json
import shutil
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Union
import random

from saltup.utils.data.image.image_utils import Image
from saltup.ai.object_detection.utils.bbox import BBoxClassId, BBoxFormat
from saltup.ai.base_dataformat.base_dataset import BaseDataloader, ColorMode
from saltup.utils.configure_logging import logging


class COCOLoader(BaseDataloader):
    def __init__(
        self,
        images_dir: str,
        annotations_file: str,
        color_mode: ColorMode = ColorMode.RGB
    ):
        """
        Initialize COCO dataset loader.

        Args:
            image_dir: Directory containing images
            annotations_file: Path to COCO annotations JSON file
            color_mode: Color mode for loading images
        
        Raises:
            ValueError: If paths are invalid
            FileNotFoundError: If directories or files don't exist
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing COCO dataset loader")
        
        # Validate input paths
        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        if not os.path.exists(annotations_file):
            raise FileNotFoundError(f"Annotations file not found: {annotations_file}")
            
        self.image_dir = Path(images_dir)
        self.annotations_file = Path(annotations_file)
        self.color_mode = color_mode
        self._current_index = 0
        
        # Load annotations and create pairs
        self.annotations = self._load_annotations()
        self.image_annotation_pairs = self._create_image_annotation_pairs()
        
        self.logger.info(f"Found {len(self.image_annotation_pairs)} image-annotation pairs")

    def __iter__(self):
        """Return iterator object (self in this case)."""
        self._current_index = 0  # Reset position when creating new iterator
        return self

    def __next__(self) -> Tuple[Union[np.ndarray, Image], List[BBoxClassId]]:
        """Get next item from dataset."""
        if self._current_index >= len(self.image_annotation_pairs):
            self._current_index = 0  # Reset for next iteration
            raise StopIteration
        
        image, annotations = self._load_item(self._current_index)
        self._current_index += 1        
        return image, annotations

    def __len__(self):
        """Return total number of samples in dataset."""
        return len(self.image_annotation_pairs)
    
    def __getitem__(self, idx: Union[int, slice]) -> Union[
        Tuple[Union[np.ndarray, Image], List[BBoxClassId]],
        List[Tuple[Union[np.ndarray, Image], List[BBoxClassId]]]
    ]:
        """Get item(s) by index.
        
        Args:
            idx: Integer index or slice object
            
        Returns:
            Single (image, annotations) tuple or list of tuples if slice
            
        Raises:
            IndexError: If index out of range
        """
        if isinstance(idx, slice):
            # Handle slice
            indices = range(*idx.indices(len(self)))
            return [self._load_item(i) for i in indices]
        else:
            # Handle single index
            return self._load_item(idx)
    
    def _load_item(self, idx: int) -> Tuple[Union[np.ndarray, Image], List[BBoxClassId]]:
        """Load single item by index.
        
        A differenza di YOLO e Pascal VOC, non necessitiamo di caricare e parsare
        file di annotazione poiché le annotazioni sono già state processate in
        _create_image_annotation_pairs().
        
        Args:
            idx: Index of the item to load
            
        Returns:
            Tuple of (image, annotations)
            
        Raises:
            IndexError: If index out of range
        """
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError("Index out of range")
            
        image_path, annotations = self.image_annotation_pairs[idx]
        image = self.load_image(image_path, self.color_mode)
        
        return image, annotations

    def _load_annotations(self) -> Dict:
        """Load COCO annotations from JSON file."""
        try:
            with open(self.annotations_file, 'r') as f:
                annotations = json.load(f)
                
            # Validate COCO format
            required_keys = ["images", "annotations", "categories"]
            if not all(key in annotations for key in required_keys):
                raise ValueError(f"Invalid COCO format. Missing one or more required keys: {required_keys}")
                
            return annotations
        except json.JSONDecodeError:
            self.logger.error(f"Invalid JSON format in annotation file: {self.annotations_file}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading annotations from {self.annotations_file}: {str(e)}")
            raise
    def split(self, ratio):
        raise NotImplementedError("Splitting COCO datasets is not implemented yet")
    
    @staticmethod
    def merge(coco_dl1, coco_dl2) -> 'COCOLoader': 
        """Merge multiple COCO datasets into one."""
        raise NotImplementedError("Merging COCO datasets is not implemented yet")
    def _create_image_annotation_pairs(self) -> List[Tuple[str, List[Dict]]]:
        """
        Create pairs of image paths and their corresponding annotations.
        
        Returns:
            List of tuples containing (image_path, annotations_list) pairs
        """
        image_annotation_pairs = []
        
        # Create a mapping from image_id to annotations
        image_to_annotations = defaultdict(list)
        for ann in self.annotations['annotations']:
            image_to_annotations[ann['image_id']].append(ann)

        # Create pairs and verify image existence
        skipped_images = 0
        for img in self.annotations['images']:
            image_path = os.path.join(self.image_dir, img['file_name'])
            if os.path.exists(image_path):
                annotations = [BBoxClassId(
                    coordinates=annotation_raw['bbox'],
                    class_id=annotation_raw['category_id'],
                    img_height=img['height'],
                    img_width=img['width'],
                    fmt=BBoxFormat.TOPLEFT_ABSOLUTE
                ) for annotation_raw in image_to_annotations[img['id']]]
                image_annotation_pairs.append(
                    (image_path, annotations)
                )
            else:
                skipped_images += 1
                self.logger.warning(f"Image not found: {img['file_name']}")
                
        if skipped_images > 0:
            self.logger.warning(f"Skipped {skipped_images} images due to missing files")
            
        return image_annotation_pairs


def create_dataset_structure(root_dir: str) -> Dict:
    """Creates COCO dataset directory structure.

    Args:
        root_dir: Root directory for dataset

    Returns:
        Dict containing paths to created directories
    """
    directories = {
        'images': {
            'train': os.path.join(root_dir, 'images', 'train'),
            'val': os.path.join(root_dir, 'images', 'val'),
            'test': os.path.join(root_dir, 'images', 'test')
        },
        'annotations': os.path.join(root_dir, 'annotations')
    }

    for category in directories['images'].values():
        os.makedirs(category, exist_ok=True)
    os.makedirs(directories['annotations'], exist_ok=True)

    return directories


def validate_dataset_structure(root_dir: str) -> Dict:
    """Verifies COCO dataset structure and validates image-annotation pairs.

    Args:
        root_dir: Dataset root directory

    Returns:
        Dict containing dataset statistics
    """
    stats = {
        'train': {'images': 0, 'annotations': 0},
        'val': {'images': 0, 'annotations': 0},
        'test': {'images': 0, 'annotations': 0}
    }

    ann_dir = os.path.join(root_dir, 'annotations')

    for split in ['train', 'val', 'test']:
        img_dir = os.path.join(root_dir, 'images', split)
        ann_file = os.path.join(ann_dir, f'instances_{split}.json')

        if os.path.exists(img_dir):
            stats[split]['images'] = len([f for f in os.listdir(img_dir)
                                          if f.endswith(('.jpg', '.jpeg', '.png'))])

        if os.path.exists(ann_file):
            annotations = read_annotations(ann_file)
            stats[split]['annotations'] = len(
                annotations.get('annotations', []))

    return stats


def get_dataset_paths(root_dir: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Get directory paths for dataset in COCO format and verify correlation between
    images and their annotations.

    Args:
        root_dir: Dataset root directory

    Returns:
        Tuple of (train_images_dir, train_annotations_file, val_images_dir, val_annotations_file)
        Returns None for paths that don't exist or don't have correlation
    """
    logger = logging.getLogger(__name__)
    
    # Verify root directory exists
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"Root directory {root_dir} does not exist")

    # Build COCO paths
    train_images_dir = os.path.join(root_dir, 'images', 'train')
    train_annotations_file = os.path.join(root_dir, 'annotations', 'instances_train.json')
    val_images_dir = os.path.join(root_dir, 'images', 'val')
    val_annotations_file = os.path.join(root_dir, 'annotations', 'instances_val.json')

    def verify_split_correlation(images_dir: str, annotation_file: str, split: str) -> Tuple[Optional[str], Optional[str]]:
        """Helper function to verify correlation between images and annotations for a split"""
        has_images = os.path.exists(images_dir) and len(os.listdir(images_dir)) > 0
        has_annotations = os.path.exists(annotation_file)
        
        if has_images and not has_annotations:
            logger.warning(f"{split} split: Found images in {images_dir} but missing annotations file {annotation_file}")
            return None, None
        
        if has_annotations and not has_images:
            try:
                with open(annotation_file, 'r') as f:
                    annotations = json.load(f)
                if annotations.get('images'):
                    logger.warning(f"{split} split: Found annotations in {annotation_file} but missing images in {images_dir}")
                    return None, None
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON format in annotation file: {annotation_file}")
                return None, None
            except Exception as e:
                logger.warning(f"Error reading annotation file {annotation_file}: {str(e)}")
                return None, None
        
        if not has_images and not has_annotations:
            logger.warning(f"{split} split: Neither images nor annotations found")
            return None, None
            
        return images_dir if has_images else None, annotation_file if has_annotations else None

    # Verify correlation for both splits
    train_imgs, train_anns = verify_split_correlation(
        train_images_dir, train_annotations_file, "Training"
    )
    val_imgs, val_anns = verify_split_correlation(
        val_images_dir, val_annotations_file, "Validation"
    )

    return train_imgs, train_anns, val_imgs, val_anns


def analyze_dataset(root_dir: str, class_names: List[str] = None):
    """Analyze COCO dataset structure and annotations.
    
    Args:
        root_dir: Dataset root directory
        class_names: Optional list of class names
    """
    structure = validate_dataset_structure(root_dir)
    ann_dir = os.path.join(root_dir, 'annotations')
    
    print("\n=== COCO Dataset Analysis ===")
    
    # Analyze each split
    for split in ['train', 'val', 'test']:
        ann_file = os.path.join(ann_dir, f'instances_{split}.json')
        if not os.path.exists(ann_file):
            continue
            
        # Structure analysis
        print(f"\n{split.upper()} Set Structure:")
        print(f"- Images directory: {structure[split]['images']} images")
        print(f"- Annotations file: {structure[split]['annotations']} annotations")
        
        # Detailed annotation analysis
        counts, total_images = count_annotations(ann_file, class_names)
        
        print(f"\n{split.upper()} Set Content:")
        print(f"- Total annotated images: {total_images}")
        print(f"- Total annotations: {sum(counts.values())}")
        print("- Annotations per category:")
        
        for cat_id, count in counts.items():
            cat_name = cat_id if isinstance(cat_id, str) else f"Category {cat_id}"
            print(f"  * {cat_name}: {count}")
            
        # Calculate average annotations per image
        avg_ann = sum(counts.values()) / total_images if total_images > 0 else 0
        print(f"- Average annotations per image: {avg_ann:.2f}")


def read_annotations(json_path: str) -> Dict:
    """Read COCO format annotations from JSON file.

    Args:
        json_path: Path to COCO annotation JSON file

    Returns:
        Dict containing parsed COCO annotations
    """
    with open(json_path, 'r') as f:
        return json.load(f)


def write_annotations(annotations: Dict, output_path: str) -> None:
    """Write annotations in COCO format to JSON file.

    Args:
        annotations: Dict containing COCO format annotations
        output_path: Output JSON file path
    """
    with open(output_path, 'w') as f:
        json.dump(annotations, f, indent=4)


def replace_annotations_class(
    old_class_id: int,
    new_class_id: int,
    coco_json: str,  
    output_json: str = None,
    verbose: bool = False
) -> tuple[int, Dict]:
    """Replace specified category ID in COCO annotations with a new ID.

    Args:
        old_class_id: Original category ID to replace
        new_class_id: New category ID to assign
        coco_json: Path to COCO annotation file
        output_json: Output annotation file path. If None, returns modified dict
        verbose: Enable progress output

    Returns:
        Tuple of (number of modified annotations, modified annotations dict)
    """
    with open(coco_json, 'r') as f:
        annotations = json.load(f)

    modified_count = 0
    modified_annotations = []

    # Update category IDs
    categories = []
    for category in annotations['categories']:
        if category['id'] == old_class_id:
            category = category.copy()
            category['id'] = new_class_id
            modified_count += 1
        categories.append(category)

    # Update annotation category IDs
    for ann in annotations['annotations']:
        if ann['category_id'] == old_class_id:
            ann = ann.copy()
            ann['category_id'] = new_class_id
            modified_count += 1
        modified_annotations.append(ann)

    modified_data = {
        'images': annotations['images'],
        'annotations': modified_annotations,
        'categories': categories
    }

    if output_json:
        with open(output_json, 'w') as f:
            json.dump(modified_data, f, indent=4)

        if verbose:
            print(f"Modified {modified_count} annotations/categories")
            print(f"Saved to {output_json}")

    return modified_count, modified_data


def shift_class_ids(
    coco_json: str, 
    shift_value: int, 
    output_json: str = None
) -> Dict:
    """Shift all category IDs in COCO annotations by a constant value.

    Args:
        coco_json: Input COCO annotation file
        shift_value: Integer value to add to all category IDs
        output_json: Output annotation file. If None, returns modified dict

    Returns:
        Dict with shifted COCO annotations
    """
    with open(coco_json, 'r') as f:
        annotations = json.load(f)

    # Shift category IDs
    shifted_categories = []
    for category in annotations['categories']:
        category = category.copy()
        category['id'] += shift_value
        shifted_categories.append(category)

    # Update annotations with new category IDs
    shifted_annotations = []
    for ann in annotations['annotations']:
        ann = ann.copy()
        ann['category_id'] += shift_value
        shifted_annotations.append(ann)

    shifted_data = {
        'images': annotations['images'],
        'annotations': shifted_annotations,
        'categories': shifted_categories
    }

    if output_json:
        with open(output_json, 'w') as f:
            json.dump(shifted_data, f, indent=4)

    return shifted_data


def convert_coco_to_yolo_labels(
    coco_json: str,
    output_dir: str = None
) -> Dict[str, List]:
    """Convert COCO annotations to YOLO format.
    
    Args:
        coco_json: Path to COCO annotation file
        output_dir: Directory to save YOLO labels (optional)
        
    Returns:
        Dict mapping image filenames to YOLO annotations
    """
    from saltup.ai.object_detection.utils.bbox import BBox, BBoxFormat
    
    with open(coco_json, 'r') as f:
        coco_data = json.load(f)
        
    # Create lookup maps
    images = {img['id']: img for img in coco_data['images']}
    categories = {cat['id']: idx for idx, cat in enumerate(coco_data['categories'])}
    
    # Convert annotations
    yolo_annotations = defaultdict(list)
    for ann in coco_data['annotations']:
        img = images[ann['image_id']]
        
        # Convert bbox coordinates
        x, y, w, h = ann['bbox']
        yolo_bbox = BBox.converter(
            coordinates=[x, y, w, h],
            from_fmt=BBoxFormat.TOPLEFT_ABSOLUTE,
            to_fmt=BBoxFormat.YOLO,
            img_shape= (img['height'], img['width'])
        )
        
        # Create YOLO annotation: class_id, x, y, w, h
        class_id = categories[ann['category_id']]
        yolo_ann = [class_id] + list(yolo_bbox)
        
        # Store by image filename
        filename = os.path.splitext(img['file_name'])[0]
        yolo_annotations[filename].append(yolo_ann)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Write YOLO label files
        for filename, annotations in yolo_annotations.items():
            label_path = os.path.join(output_dir, f"{filename}.txt")
            with open(label_path, 'w') as f:
                for ann in annotations:
                    f.write(f"{' '.join(map(str, ann))}\n")
                    
        # Write category names
        with open(os.path.join(output_dir, 'classes.names'), 'w') as f:
            sorted_categories = sorted(
                coco_data['categories'], 
                key=lambda x: categories[x['id']]
            )
            f.write('\n'.join(cat['name'] for cat in sorted_categories))
    
    return dict(yolo_annotations)


def split_dataset(
    annotations: Dict,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    max_images_per_class: Optional[int] = None
) -> Tuple[Dict, Dict, Dict]:
    """Split COCO dataset into train/val/test sets with optional size limits.

    Args:
        annotations: COCO format annotations
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        max_images_per_class: Maximum number of images per class in each split.
                            Limits are applied separately to train and val sets.
                            None means no limit.

    Returns:
        Tuple of train, validation, and test annotation dicts

    Raises:
        ValueError: If ratios don't sum to 1.0 or max_images_per_class is negative
    """
    train_ratio, val_ratio, test_ratio = map(abs, [train_ratio, val_ratio, test_ratio])
    if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10:
        raise ValueError("Split ratios must sum to 1.0")
    
    if max_images_per_class is not None and max_images_per_class < 0:
        raise ValueError("max_images_per_class must be positive or None")

    image_ids = list(set(img['id'] for img in annotations['images']))
    random.shuffle(image_ids)

    n_train = int(len(image_ids) * train_ratio)
    n_val = int(len(image_ids) * val_ratio)
    
    if max_images_per_class:
        n_train = min(max_images_per_class, n_train)
        n_val = min(max_images_per_class, n_val)

    train_ids = set(image_ids[:n_train])
    val_ids = set(image_ids[n_train:n_train + n_val])
    test_ids = set(image_ids[n_train + n_val:])

    def filter_by_image_ids(ids: set) -> Dict:
        return {
            'images': [img for img in annotations['images']
                      if img['id'] in ids],
            'annotations': [ann for ann in annotations['annotations']
                          if ann['image_id'] in ids],
            'categories': annotations['categories']
        }

    return (filter_by_image_ids(train_ids),
            filter_by_image_ids(val_ids),
            filter_by_image_ids(test_ids))


def split_and_organize_dataset(
    root_dir: str,
    annotations_file: str,
    max_images_per_class: Optional[int] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1
) -> None:
    """Split and organize COCO dataset into train/val/test directories.

    Args:
        root_dir: Dataset root directory 
        annotations_file: Path to COCO annotations JSON
        max_images_per_class: Maximum images per class in each split (None = no limit)
        train_ratio: Training set proportion
        val_ratio: Validation set proportion
        test_ratio: Test set proportion

    Raises:
        FileExistsError: If destination image paths already exist
    """
    # Read annotations
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)

    # Split dataset
    train_anns, val_anns, test_anns = split_dataset(
        annotations,
        train_ratio,
        val_ratio,
        test_ratio,
        max_images_per_class
    )

    # Create directory structure
    annotations_dir = os.path.join(root_dir, 'annotations')
    os.makedirs(annotations_dir, exist_ok=True)

    splits = {
        'train': train_anns,
        'val': val_anns,
        'test': test_anns
    }

    # Track moved files to prevent duplicates
    moved_files = set()

    for split_name, split_anns in splits.items():
        # Create image directory
        split_img_dir = os.path.join(root_dir, 'images', split_name)
        os.makedirs(split_img_dir, exist_ok=True)

        # Save split annotations
        ann_file = os.path.join(
            annotations_dir, f'instances_{split_name}.json')
        with open(ann_file, 'w') as f:
            json.dump(split_anns, f)

        # Move images
        for img in split_anns['images']:
            src = os.path.join(root_dir, 'images', img['file_name'])
            dst = os.path.join(split_img_dir, img['file_name'])
            
            if img['file_name'] not in moved_files:
                if os.path.exists(src):
                    shutil.move(src, dst)
                    moved_files.add(img['file_name'])

    print(f"Training set: {len(train_anns['images'])} images")
    print(f"Validation set: {len(val_anns['images'])} images")
    print(f"Test set: {len(test_anns['images'])} images")


def count_annotations(
    coco_json: str,
    class_names: Optional[List[str]] = None,
    verbose: bool = False
) -> Tuple[Dict[Union[int, str], int], int]:
    """Count occurrences of each category and total annotated images in COCO dataset.

    Args:
        coco_json: Path to COCO annotation JSON
        class_names: Optional list of class names to map to category IDs.
                    If provided, must have same length as number of categories.
                    Categories not in class_names will be skipped.
        verbose: Enable progress output

    Returns:
        Tuple of:
            Dict mapping category ID/name to count of annotations
            Number of unique annotated images
    """
    with open(coco_json, 'r') as f:
        data = json.load(f)

    # Count annotations per category
    category_counts = defaultdict(int)
    annotated_images = set()

    for ann in tqdm(data['annotations'], disable=not verbose):
        category_counts[ann['category_id']] += 1
        annotated_images.add(ann['image_id'])

    # Replace IDs with names if provided
    if class_names:
        if len(class_names) != len(data['categories']):
            raise ValueError("class_names must have same length as number of categories")
            
        categories = {cat['id']: class_names[i] 
                     for i, cat in enumerate(data['categories'])
                     if i < len(class_names)}
        category_counts = {
            categories[cat_id]: count 
            for cat_id, count in category_counts.items()
            if cat_id in categories
        }

    return dict(category_counts), len(annotated_images)

