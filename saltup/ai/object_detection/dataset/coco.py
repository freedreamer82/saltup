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
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Union
import random


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
            annotations = read_coco_annotations(ann_file)
            stats[split]['annotations'] = len(
                annotations.get('annotations', []))

    return stats


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


def read_coco_annotations(json_path: str) -> Dict:
    """Read COCO format annotations from JSON file.

    Args:
        json_path: Path to COCO annotation JSON file

    Returns:
        Dict containing parsed COCO annotations
    """
    with open(json_path, 'r') as f:
        return json.load(f)


def write_coco_annotations(annotations: Dict, output_path: str) -> None:
    """Write annotations in COCO format to JSON file.

    Args:
        annotations: Dict containing COCO format annotations
        output_path: Output JSON file path
    """
    with open(output_path, 'w') as f:
        json.dump(annotations, f, indent=4)


def replace_label_class(
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


def convert_coco_to_yolo_annotations(
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
    from saltup.ai.object_detection.dataset.bbox_utils import coco_to_yolo_bbox
    
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
        yolo_bbox = coco_to_yolo_bbox(
            [x, y, w, h],
            img['width'],
            img['height']
        )
        
        # Create YOLO annotation: class_id, x, y, w, h
        class_id = categories[ann['category_id']]
        yolo_ann = [class_id] + yolo_bbox
        
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

