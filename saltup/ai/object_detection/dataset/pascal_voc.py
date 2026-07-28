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
import tempfile
from typing import Dict, List, Tuple, Optional, Union

from saltup.utils.data.image.image_utils import Image
from saltup.utils.data.s3.s3_utils import S3
from botocore.exceptions import ClientError
from saltup.ai.object_detection.utils.bbox import BBoxClassId, BBoxFormat
from saltup.ai.base_dataformat.base_dataloader import BaseDataloader, ColorMode
from saltup.ai.base_dataformat.label_map import LabelMap
from saltup.utils import configure_logging


class PascalVOCLoader(BaseDataloader):
    def __init__(
        self,
        images_dir: Union[str, Path],
        annotations_dir: Union[str, Path],
        color_mode: ColorMode = ColorMode.RGB,
        label_map: Optional[LabelMap] = None
    ):
        """
        Initialize Pascal VOC dataset loader.

        Args:
            image_dir: Directory containing images
            annotations_dir: Directory containing XML annotations
            color_mode: Color mode for loading images
            label_map: Optional LabelMap collapsing or renaming labels as they are loaded

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
        self._label_map = label_map
        self._current_index = 0

        # Load image-annotation pairs
        self.image_annotation_pairs = self._load_image_annotation_pairs()
        self.logger.info(f"Found {len(self.image_annotation_pairs)} image-annotation pairs")

    def __iter__(self):
        """Return iterator object (self in this case)."""
        self._current_index = 0  # Reset position when creating new iterator
        return self

    def __next__(self) -> Tuple[Path, Optional[Image], List[BBoxClassId]]:
        """Get next item from dataset."""
        if self._current_index >= len(self.image_annotation_pairs):
            self._current_index = 0  # Reset for next iteration
            raise StopIteration

        image_path, image, annotations = self._load_item(self._current_index)
        self._current_index += 1
        return image_path, image, annotations

    def __len__(self):
        """Return total number of samples in dataset."""
        return len(self.image_annotation_pairs)
    
    def __getitem__(self, idx: Union[int, slice]) -> Union[
        Tuple[Path, Optional[Image], List[BBoxClassId]],
        List[Tuple[Path, Optional[Image], List[BBoxClassId]]]
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

    def _load_item(self, idx: int) -> Tuple[Path, Optional[Image], List[BBoxClassId]]:
        """Load single item by index.
        
        Args:
            idx: Index of the item to load
            
        Returns:
            Tuple of (image_path, image, annotations)
            
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

        return Path(image_path), image, self._apply_label_map(annotations)

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
    
class PascalVOCS3Loader(BaseDataloader):
    def __init__(
        self,
        images_dir: str,
        annotations_dir: str,
        s3_client: S3,
        download_file: bool = False,
        max_files: int = -1,
        color_mode: ColorMode = ColorMode.RGB,
        label_map: Optional[LabelMap] = None
    ):
        """
        Initialize Pascal VOC S3 dataset loader.

        Args:
            images_dir: Directory containing images (local or S3 path)
            annotations_file: Path to annotation file or directory (local or S3 path)
            s3_client: S3 client for downloading files
            max_files: Maximum number of files to download from S3 (-1 for all)
            download_file: Whether to download files from S3
            color_mode: Color mode for loading images
            label_map: Optional LabelMap collapsing or renaming labels as they are loaded

        Raises:
            FileNotFoundError: If directories don't exist
        """
        self.download_file = download_file
        self.max_files = max_files
        self.downloaded_files = 0
        self.s3_client = s3_client

        if self.download_file:
            if self.max_files <= 0:
                raise ValueError("max_files must be > 0 when download_file is True")
        self.logger = configure_logging.get_logger(__name__)
        self.logger.info("Initializing Pascal VOC dataset loader")
    
        self.images_dir = Path(images_dir)
        self.annotations_dir = Path(annotations_dir)
        self.color_mode = color_mode
        self._label_map = label_map
        self._current_index = 0

        # Load image-annotation pairs
        self.image_annotation_pairs = self._load_image_annotation_pairs()
        self.logger.info(f"Found {len(self.image_annotation_pairs)} image-annotation pairs")

    def __iter__(self):
        """Return iterator object (self in this case)."""
        self._current_index = 0  # Reset position when creating new iterator
        return self

    def __next__(self) -> Tuple[Union[Path, str], Optional[Image], List[BBoxClassId]]:
        """Get next item from dataset."""
        if self._current_index >= len(self.image_annotation_pairs):
            self._current_index = 0  # Reset for next iteration
            raise StopIteration

        image_path, image, annotations = self._load_item(self._current_index)
        self._current_index += 1
        return image_path, image, annotations

    def __len__(self):
        """Return total number of samples in dataset."""
        return len(self.image_annotation_pairs)
    
    def __getitem__(self, idx: Union[int, slice]) -> Union[
        Tuple[Union[Path, str], Optional[Image], List[BBoxClassId]],
        List[Tuple[Union[Path, str], Optional[Image], List[BBoxClassId]]]
    ]:
        """Get item(s) by index.
        
        Args:
            idx: Integer index or slice object
            
        Returns:
            Single (image_path, image, annotations) tuple or list of tuples if slice
            
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

    def _load_item(self, idx: int) -> Tuple[Union[Path, str], Optional[Image], List[BBoxClassId]]:
        """Load single item by index.
        
        Args:
            idx: Index of the item to load
            
        Returns:
            Tuple of (image_path, image, annotations)
            
        Raises:
            IndexError: If index out of range
        """
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError("Index out of range")
            
        image_path, annotation_path = self.image_annotation_pairs[idx]
        
        if self.download_file and self.downloaded_files < self.max_files:
            with tempfile.TemporaryDirectory() as tmpdirname:
                print('created temporary directory', tmpdirname)
                
                try:
                    self.s3_client.download_file(
                        image_path,
                        tmpdirname
                    )
                    self.logger.info(f"Downloaded {image_path} to {tmpdirname}")
                    self.downloaded_files += 1
                except ClientError as e:
                    self.logger.error(f"Failed to download {image_path} from S3: {str(e)}")
                    image = None
                    raise
                image = self.load_image(os.path.join(tmpdirname, os.path.basename(image_path)), self.color_mode)     
        else:
            image = None
        with tempfile.TemporaryDirectory() as tmpdirname:
            print('created temporary directory', tmpdirname)
            try:
                self.s3_client.download_file(
                    annotation_path,
                    tmpdirname
                )
                self.logger.info(f"Downloaded {annotation_path} to {tmpdirname}")
                annotation_path = os.path.join(tmpdirname, os.path.basename(annotation_path))
            except ClientError as e:
                self.logger.error(f"Failed to download {annotation_path} from S3: {str(e)}")
                raise
            temp_annotation_path = os.path.join(tmpdirname, os.path.basename(annotation_path))
            annotations = read_annotation(temp_annotation_path)
        image_path = os.path.join("s3://", self.s3_client._bucket_name, image_path)
        return image_path, image, self._apply_label_map(annotations)

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
        
        # List files by date if start_date and end_date are provided
        image_list = self.s3_client.ls(str(self.images_dir), ['*.jpg', '*.jpeg', '*.png'], only_basename=True)
        annotation_list = self.s3_client.ls(str(self.annotations_dir), ['*.xml'], only_basename=True)
        
        for image_file in image_list:
            common_name = image_file.split(".")[0]
            annotation_name = f"{common_name}.xml"
            if annotation_name in annotation_list:
                self.logger.info(f"Found annotation for {image_file}: {annotation_name}")
                image_path = str(self.images_dir / image_file)
                annotation_path = str(self.annotations_dir / annotation_name)
                image_annotation_pairs.append((image_path, annotation_path))
            else:
                self.logger.warning(f"Annotation not found for {image_file}")
                skipped_images += 1
                continue
        
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


def is_pascal_voc_dataset(root_dir: Union[str, Path]) -> bool:
    """
    Checks whether the given directory contains a dataset in Pascal VOC format.

    This function locates the dataset's image and annotation directories via
    `get_dataset_paths` -- which accepts both the split layout
    ('images'/'annotations' with train/val/test subdirectories) and a flat,
    unsplit one ('JPEGImages'/'Annotations', or plain 'images'/'annotations') --
    and then requires at least one XML annotation file to be present.

    The XML requirement is what keeps this from matching a COCO dataset, whose
    annotations directory holds JSON.

    Args:
        root_dir: The root directory to check for Pascal VOC dataset structure.

    Returns:
        bool: True if the directory appears to be a Pascal VOC dataset, False otherwise.
    """
    if not isinstance(root_dir, Path):
        root_dir = Path(root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"Root directory {root_dir} does not exist")

    train_images_dir, train_annotations_dir, val_images_dir, val_annotations_dir, test_images_dir, test_annotations_dir = get_dataset_paths(str(root_dir))

    if not any([train_images_dir, train_annotations_dir, val_images_dir, val_annotations_dir, test_images_dir, test_annotations_dir]):
        # If all directories are None, it's not a valid Pascal VOC dataset
        return False
    
    # Check if at least one of the required directories exists and contains at least one annotation file
    for d in [train_annotations_dir, val_annotations_dir, test_annotations_dir]:
        if d and os.path.exists(d):
            if any(f.endswith('.xml') for f in os.listdir(d)):
                return True
    return False


# Image/annotation directory pairs recognized for a flat, unsplit dataset. The
# canonical Pascal VOC layout (as shipped in VOCdevkit) comes first.
_FLAT_LAYOUTS = (
    ('JPEGImages', 'Annotations'),
    ('images', 'annotations'),
)


def get_dataset_paths(root_dir: Union[str, Path]) -> Tuple[
    Optional[Union[str, Path]], Optional[Union[str, Path]],
    Optional[Union[str, Path]], Optional[Union[str, Path]],
    Optional[Union[str, Path]], Optional[Union[str, Path]]
]:
    """Get directory paths for dataset in Pascal VOC format.

    Two layouts are recognized:

    1. Split directories, `images/<split>` and `annotations/<split>` for
       train/val/test, as produced by `create_dataset_structure`.
    2. A flat, unsplit dataset: the canonical `JPEGImages/` + `Annotations/`
       pair used by VOCdevkit, or a plain `images/` + `annotations/` pair.
       These are reported as the train split, with val and test set to None.

    The split layout takes precedence: the flat fallback is only considered when
    no split directory is found.

    Note that the official train/val/test membership of a canonical VOC dataset
    lives in `ImageSets/Main/*.txt`, which is not interpreted here -- the whole
    directory is returned as a single split. A warning is logged when those files
    are present so the ignored split is not silent.

    Args:
        root_dir: Dataset root directory

    Returns:
        Tuple of (train_images_dir, train_annotations_dir, val_images_dir, val_annotations_dir, test_images_dir, test_annotations_dir)
        If a directory does not exist or is empty, its value will be None.
    """
    def check_dir(path):
        if os.path.exists(path) and any(Path(path).iterdir()):
            return path
        return None

    train_images_dir = check_dir(os.path.join(root_dir, 'images', 'train'))
    train_annotations_dir = check_dir(os.path.join(root_dir, 'annotations', 'train'))
    val_images_dir = check_dir(os.path.join(root_dir, 'images', 'val'))
    val_annotations_dir = check_dir(os.path.join(root_dir, 'annotations', 'val'))
    test_images_dir = check_dir(os.path.join(root_dir, 'images', 'test'))
    test_annotations_dir = check_dir(os.path.join(root_dir, 'annotations', 'test'))

    split_paths = (
        train_images_dir,
        train_annotations_dir,
        val_images_dir,
        val_annotations_dir,
        test_images_dir,
        test_annotations_dir
    )
    if any(split_paths):
        return split_paths

    # No split directories: fall back to a flat, unsplit dataset.
    for images_name, annotations_name in _FLAT_LAYOUTS:
        images_dir = check_dir(os.path.join(root_dir, images_name))
        annotations_dir = check_dir(os.path.join(root_dir, annotations_name))
        if images_dir and annotations_dir:
            if os.path.isdir(os.path.join(root_dir, 'ImageSets', 'Main')):
                configure_logging.get_logger(__name__).warning(
                    f"{root_dir} has ImageSets/Main, but its train/val/test membership "
                    f"is not applied: the whole dataset is returned as a single split."
                )
            return (images_dir, annotations_dir, None, None, None, None)

    return split_paths


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
    # Get paths for train, val, and test directories
    train_images_dir, train_annotations_dir, val_images_dir, val_annotations_dir, test_images_dir, test_annotations_dir = get_dataset_paths(root_dir)

    # Initialize statistics for train, val, and test
    stats = {
        'train': {'images': 0, 'annotations': 0, 'matched': 0, 'unmatched_images': [], 'unmatched_annotations': []},
        'val': {'images': 0, 'annotations': 0, 'matched': 0, 'unmatched_images': [], 'unmatched_annotations': []},
        'test': {'images': 0, 'annotations': 0, 'matched': 0, 'unmatched_images': [], 'unmatched_annotations': []}
    }

    # Helper function to check image-annotation correspondences
    def check_matches(images_dir, annotations_dir, split):
        if images_dir and os.path.exists(images_dir):
            image_files = {os.path.splitext(f)[0] for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))}
        else:
            image_files = set()

        if annotations_dir and os.path.exists(annotations_dir):
            annotation_files = {os.path.splitext(f)[0] for f in os.listdir(annotations_dir) if f.endswith('.xml')}
        else:
            annotation_files = set()

        stats[split]['images'] = len(image_files)
        stats[split]['annotations'] = len(annotation_files)
        stats[split]['matched'] = len(image_files & annotation_files)
        stats[split]['unmatched_images'] = list(image_files - annotation_files)
        stats[split]['unmatched_annotations'] = list(annotation_files - image_files)

    # Check matches for train, val, and test
    check_matches(train_images_dir, train_annotations_dir, 'train')
    check_matches(val_images_dir, val_annotations_dir, 'val')
    check_matches(test_images_dir, test_annotations_dir, 'test')

    return stats


def read_annotation(annotation_file: str) -> List[BBoxClassId]:
    """Parse Pascal VOC format annotations from an XML file."""
    tree = ET.parse(annotation_file)
    root = tree.getroot()

    # Helper function to safely get text content
    def get_text(element, path: str) -> Optional[str]:
        elem = element.find(path)
        return elem.text if elem is not None else None

    # Helper function to safely get int value
    def get_int(element, path: str) -> Optional[int]:
        text = get_text(element, path)
        return int(text) if text is not None else None

    width = get_int(root, 'size/width')
    height = get_int(root, 'size/height')

    annotations = []
    for obj in root.findall('object'):
        class_name = get_text(obj, 'name')
        bbox = obj.find('bndbox')
        
        if bbox is not None:
            xmin = get_int(bbox, 'xmin')
            ymin = get_int(bbox, 'ymin')
            xmax = get_int(bbox, 'xmax')
            ymax = get_int(bbox, 'ymax')
            
            # Skip objects with incomplete bbox data
            if all(coord is not None for coord in [xmin, ymin, xmax, ymax]):
                annotations.append(BBoxClassId(
                    coordinates=(xmin, ymin, xmax, ymax),
                    class_name=class_name or "unknown",
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
