"""
Pascal VOC Dataset Utilities
===========================

This module provides utilities for handling Pascal VOC format annotations.

Pascal VOC Format Overview:
-------------------------
XML files per image with structure:
<annotation>
    <folder>VOC2007</folder>
    <filename>image.jpg</filename>
    <size>
        <width>500</width>
        <height>375</height>
        <depth>3</depth>
    </size>
    <object>
        <name>car</name>
        <bndbox>
            <xmin>156</xmin>
            <ymin>97</ymin>
            <xmax>351</xmax>
            <ymax>270</ymax>
        </bndbox>
    </object>
</annotation>

Key functions:
- Reading/writing VOC annotations
- Dataset organization and validation
- Statistics and analysis
- Dataset splitting
"""

import os
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

from saltup.utils.data.image.image_utils import Image
from saltup.ai.object_detection.utils.bbox import BBoxClassId, BBoxFormat
from saltup.ai.base_dataformat.base_dataset import BaseDataloader, ColorMode, StorageFormat
from saltup.utils import configure_logging


class PascalVOCLoader(BaseDataloader):
    def __init__(
        self,
        images_dir: str,
        annotations_dir: str,
        color_mode: ColorMode = ColorMode.RGB
    ):
        """
        Initialize Pascal VOC dataset loader.
        
        Args:
            image_dir: Directory containing images
            annotations_dir: Directory containing XML annotations
            color_mode: Color mode for loading images
            
        Raises:
            FileNotFoundError: If directories don't exist
        """
        self.logger = configure_logging.get_logger(__name__)
        self.logger.info("Initializing Pascal VOC dataset loader")
        
        # Validate directories existence
        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        if not os.path.exists(annotations_dir):
            raise FileNotFoundError(f"Annotations directory not found: {annotations_dir}")
            
        self.image_dir = Path(images_dir)
        self.annotations_dir = Path(annotations_dir)
        self.color_mode = color_mode
        self._current_index = 0
        
        # Load image-annotation pairs
        self.image_annotation_pairs = self._load_image_annotation_pairs()
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
            
        image_path, annotation_path = self.image_annotation_pairs[idx]
        image = self.load_image(image_path, self.color_mode)
        annotations = read_annotation(annotation_path)
        
        return image, annotations
    
    def split(self, ratio):
        """Split dataset into subsets based on given ratio."""
        raise NotImplementedError("Split method not implemented for Pascal VOC format")
    
    @staticmethod
    def merge(pascalVOC_ld1, pascalVOC_ld2) -> 'PascalVOCLoader':
        """Merge two Pascal VOC loaders into one.
        
        Args:
            pascalVOC_ld1: First Pascal VOC loader
            pascalVOC_ld2: Second Pascal VOC loader
            """
        raise NotImplementedError("Merge method not implemented for Pascal VOC format")
        
    def _load_image_annotation_pairs(self) -> List[Tuple[str, str]]:
        """
        Load pairs from images and annotations directories.
        
        Returns:
            List of tuples containing (image_path, annotation_path) pairs
        """
        image_annotation_pairs = []
        skipped_images = 0
        
        for image_file in os.listdir(self.image_dir):
            if image_file.endswith(('.jpg', '.jpeg', '.png')):
                base_name = os.path.splitext(image_file)[0]
                image_path = str(self.image_dir / image_file)
                annotation_path = str(self.annotations_dir / f"{base_name}.xml")
                
                if os.path.exists(annotation_path):
                    image_annotation_pairs.append((image_path, annotation_path))
                else:
                    skipped_images += 1
                    self.logger.warning(f"Annotation not found for {image_file}")
        
        if skipped_images > 0:
            self.logger.warning(f"Skipped {skipped_images} images due to missing annotations")
            
        return image_annotation_pairs

def create_dataset_structure(root_dir: str):
    """Creates Pascal VOC directory structure if it doesn't exist.
    
    Args:
        root_dir (str): Root directory for the dataset
        
    Returns:
        dict: Dictionary containing paths to created directories
    """
    # Create main directories
    directories = {
        'images': {
            'train': os.path.join(root_dir, 'images', 'train'),
            'val': os.path.join(root_dir, 'images', 'val'),
            'test': os.path.join(root_dir, 'images', 'test')
        },
        'annotations': {
            'train': os.path.join(root_dir, 'annotations', 'train'),
            'val': os.path.join(root_dir, 'annotations', 'val'),
            'test': os.path.join(root_dir, 'annotations', 'test')
        }
    }
    
    # Create directories if they don't exist
    for category in directories.values():
        for dir_path in category.values():
            os.makedirs(dir_path, exist_ok=True)
            
    return directories


def get_dataset_paths(root_dir: str) -> Tuple[str, str, str, str]:
    """Get directory paths for dataset in Pascal VOC format.
    
    Args:
        root_dir: Dataset root directory
        
    Returns:
        Tuple of (train_images_dir, train_annotations_dir, val_images_dir, val_annotations_dir)
        
    Raises:
        FileNotFoundError: If required directories don't exist
    """
    # Verify root directory exists
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"Root directory {root_dir} does not exist")
    
    # Build Pascal VOC paths
    train_images_dir = os.path.join(root_dir, 'images', 'train')
    train_annotations_dir = os.path.join(root_dir, 'annotations', 'train')
    val_images_dir = os.path.join(root_dir, 'images', 'val')
    val_annotations_dir = os.path.join(root_dir, 'annotations', 'val')
    
    # Verify required Pascal VOC directories exist
    required_dirs = [
        (train_images_dir, "Train Images"),
        (train_annotations_dir, "Train Annotations"), 
        (val_images_dir, "Validation Images"),
        (val_annotations_dir, "Validation Annotations")
    ]
    
    for dir_path, dir_name in required_dirs:
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"{dir_name} directory not found at {dir_path}")
    
    return train_images_dir, train_annotations_dir, val_images_dir, val_annotations_dir


def validate_dataset_structure(root_dir: str) -> Dict[str, Dict[str, Union[int, List[str]]]]:
    """Verify directory structure and validate image-annotation pairs.
    
    Args:
        root_dir: Dataset root directory
        
    Returns:
        Dict containing per-split statistics:
            images: Number of images
            annotations: Number of annotations
            matched: Number of matched pairs
            unmatched_images: List of images without annotations
            unmatched_annotations: List of annotations without images
    """
    # Ottieni i percorsi per le directory train e val
    train_images_dir, train_annotations_dir, val_images_dir, val_annotations_dir = get_dataset_paths(root_dir)
    
    # Aggiungi i percorsi per la directory test
    test_images_dir = os.path.join(root_dir, 'images', 'test')
    test_annotations_dir = os.path.join(root_dir, 'annotations', 'test')
    
    # Inizializza le statistiche per train, val e test
    stats = {
        'train': {'images': 0, 'annotations': 0, 'matched': 0, 'unmatched_images': [], 'unmatched_annotations': []},
        'val': {'images': 0, 'annotations': 0, 'matched': 0, 'unmatched_images': [], 'unmatched_annotations': []},
        'test': {'images': 0, 'annotations': 0, 'matched': 0, 'unmatched_images': [], 'unmatched_annotations': []}
    }
    
    # Helper function to check image-annotation correspondences
    def check_matches(images_dir, annotations_dir, split):
        """Verifica le corrispondenze tra immagini e annotazioni per uno split specifico."""
        if os.path.exists(images_dir):
            image_files = {os.path.splitext(f)[0] for f in os.listdir(images_dir) 
                          if f.endswith(('.jpg', '.jpeg', '.png'))}
        else:
            image_files = set()
        
        if os.path.exists(annotations_dir):
            annotation_files = {os.path.splitext(f)[0] for f in os.listdir(annotations_dir) 
                          if f.endswith('.xml')}
        else:
            annotation_files = set()
        
        stats[split]['images'] = len(image_files)
        stats[split]['annotations'] = len(annotation_files)
        stats[split]['matched'] = len(image_files & annotation_files)
        stats[split]['unmatched_images'] = list(image_files - annotation_files)
        stats[split]['unmatched_annotations'] = list(annotation_files - image_files)
    
    # Verifica le corrispondenze per train, val e test
    check_matches(train_images_dir, train_annotations_dir, 'train')
    check_matches(val_images_dir, val_annotations_dir, 'val')
    check_matches(test_images_dir, test_annotations_dir, 'test')
    
    return stats


def read_annotation(annotation_file: str) -> List[BBoxClassId]:
    """Parse Pascal VOC format annotations from an XML file.

    Args:
        annotation_file (str): Path to the Pascal VOC annotation file

    Returns:
        List of dictionaries containing object annotations:
            - class_name: Name of the object class
            - bbox: Tuple of (xmin, ymin, xmax, ymax)
    """
    tree = ET.parse(annotation_file)
    root = tree.getroot()
    
    width = int(root.find('size/width').text)
    height = int(root.find('size/height').text)

    annotations = []
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        annotations.append(BBoxClassId(
            coordinates=(xmin, ymin, xmax, ymax),
            class_name=class_name,
            # Pascal VOC don't have class IDs
            class_id=None,
            img_height=height,
            img_width=width,
            fmt=BBoxFormat.CORNERS_ABSOLUTE
        ))

    return annotations


def write_annotation(annotation_file: str, annotations: List[Dict], image_info: Dict) -> None:
    """Write annotations in Pascal VOC format to an XML file.

    Args:
        annotation_file: Path to output XML file
        annotations: List of dictionaries containing object annotations:
            - class_name: Name of the object class
            - bbox: Tuple of (xmin, ymin, xmax, ymax)
        image_info: Dictionary containing image information:
            - filename: Name of the image file
            - width: Width of the image
            - height: Height of the image
    """
    root = ET.Element('annotation')

    # Add image information
    filename = ET.SubElement(root, 'filename')
    filename.text = image_info['filename']

    size = ET.SubElement(root, 'size')
    width = ET.SubElement(size, 'width')
    width.text = str(image_info['width'])
    height = ET.SubElement(size, 'height')
    height.text = str(image_info['height'])
    depth = ET.SubElement(size, 'depth')
    depth.text = '3'

    # Add object annotations
    for ann in annotations:
        obj = ET.SubElement(root, 'object')
        name = ET.SubElement(obj, 'name')
        name.text = ann['class_name']
        bndbox = ET.SubElement(obj, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        xmin.text = str(ann['bbox'][0])
        ymin = ET.SubElement(bndbox, 'ymin')
        ymin.text = str(ann['bbox'][1])
        xmax = ET.SubElement(bndbox, 'xmax')
        xmax.text = str(ann['bbox'][2])
        ymax = ET.SubElement(bndbox, 'ymax')
        ymax.text = str(ann['bbox'][3])

    # Convert the ElementTree to a string with pretty formatting
    xml_str = ET.tostring(root, encoding='utf-8')
    xml_pretty = minidom.parseString(xml_str).toprettyxml(indent="    ")

    # Write the pretty XML to the file
    with open(annotation_file, 'w', encoding='utf-8') as f:
        f.write(xml_pretty)


def _find_matching_annotation(base_name: str, annotations_dir: str) -> Optional[str]:
    """Find matching annotation file for a given base name.
    
    Args:
        base_name: Base name without extension
        annotations_dir: Directory containing Pascal VOC annotations
        
    Returns:
        Full path to matching annotation if found, None otherwise
    """
    annotation_path = os.path.join(annotations_dir, f"{base_name}.xml")
    return annotation_path if os.path.exists(annotation_path) else None
