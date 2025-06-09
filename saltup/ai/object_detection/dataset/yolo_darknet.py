from PIL import Image
import os
import json
import random
import shutil
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict, Counter
from typing import Iterable, Union, List, Dict, Optional, Tuple, Set

from saltup.utils.data.image.image_utils import Image
from saltup.ai.object_detection.utils.bbox import BBoxClassId, BBoxFormat
from saltup.ai.base_dataformat.base_dataloader import BaseDataloader, ColorMode
from saltup.ai.base_dataformat.base_dataset import Dataset
from saltup.utils import configure_logging


class YoloDarknetLoader(BaseDataloader):
    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        color_mode: ColorMode = ColorMode.RGB
    ):
        """
        Initialize YoloDarknet dataset loader.
        
        Args:
            image_dir: Directory containing images
            labels_dir: Directory containing labels
            color_mode: Color mode for loading images
            
        Raises:
            FileNotFoundError: If directories don't exist
        """
        self.__logger = configure_logging.get_logger(__name__)
        self.__logger.info("Initializing YOLO Darknet dataset loader")
        
        # Validate directories existence
        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        if not os.path.exists(labels_dir):
            raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
            
        self._images_dir = Path(images_dir)
        self._labels_dir = Path(labels_dir)
        self.color_mode = color_mode
        self._current_index = 0
        
        # Load image-label pairs
        self.image_label_pairs = self._load_image_label_pairs()
        self.__logger.info(f"Found {len(self.image_label_pairs)} image-label pairs")

    def __iter__(self):
        """Return iterator object (self in this case)."""
        self._current_index = 0  # Reset position when creating new iterator
        return self

    def __next__(self) -> Tuple[Image, List[BBoxClassId]]:
        """Get next item from dataset."""
        if self._current_index >= len(self.image_label_pairs):
            self._current_index = 0  # Reset for next iteration
            raise StopIteration
        
        image, annotations = self._load_item(self._current_index) 
        self._current_index += 1
        
        return image, annotations
    
    def __getitem__(self, idx: Union[int, slice])-> Union[
        Tuple[Image, List[BBoxClassId]],
        List[Tuple[Image, List[BBoxClassId]]]
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
        
    def _load_item(self, idx: int)-> Tuple[Image, List[BBoxClassId]]:
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
        
        image_path, label_path = self.image_label_pairs[idx]
        image = self.load_image(image_path, self.color_mode)
        image_width, image_height = image.get_width(), image.get_height()
        
        annotations = [BBoxClassId(
            # (class_id, xc, yc, w, h)
            coordinates=lbl[1:],
            class_id=lbl[0],
            class_name=None,
            fmt=BBoxFormat.YOLO,
            img_width=image_width,
            img_height=image_height
        ) for lbl in read_label(label_path)]
        
        return image, annotations        

    def __len__(self):
        """Return total number of samples in dataset."""
        return len(self.image_label_pairs)
    
    def split(self, ratio):
        raise NotImplementedError("Not implemented yet")
    
    @staticmethod
    def merge(YoloDarknetLoader1, YoloDarknetLoader2) -> 'YoloDarknetLoader':
        """Merge two YoloDarknetLoader instances into one.
        
        Args:
            YoloDarknetLoader1: First loader instance
            YoloDarknetLoader2: Second loader instance
        """
        raise NotImplementedError("Merging not implemented yet")
    
    def _load_image_label_pairs(self) -> List[Tuple[str, str]]:
        """
        Load pairs from images and labels directories.
        
        Returns:
            List of tuples containing (image_path, label_path) pairs
        """
        image_label_pairs = []
        skipped_images = 0
        
        for image_file in os.listdir(self._images_dir):
            if image_file.endswith(('.jpg', '.jpeg', '.png')):
                base_name = os.path.splitext(image_file)[0]
                image_path = str(self._images_dir / image_file)
                label_path = str(self._labels_dir / f"{base_name}.txt")
                
                if os.path.exists(label_path):
                    image_label_pairs.append((image_path, label_path))
                else:
                    skipped_images += 1
                    self.__logger.warning(f"Label not found for {image_file}")
        
        if skipped_images > 0:
            self.__logger.warning(f"Skipped {skipped_images} images due to missing labels")
            
        return image_label_pairs


class YoloDataset(Dataset):
    def __init__(self, images_dir: str, labels_dir: str, refresh_each: int = -1):
        """
        Initialize the YoloDataset for medium to large datasets.

        Args:
            images_dir (str): Directory containing image files.
            labels_dir (str): Directory containing label files.
            refresh_each (int): Number of operations before refreshing image IDs cache.
                If -1, refreshes only on explicit calls. Defaults to -1.
                
        Raises:
            FileNotFoundError: If images_dir or labels_dir don't exist.
            RuntimeError: If dataset integrity check fails.
        """
        super().__init__(images_dir)
        self._images_dir = Path(images_dir)
        self._labels_dir = Path(labels_dir)
        self.__logger = configure_logging.get_logger(__name__)
        
        # Validate directories
        if not self._images_dir.exists():
            raise FileNotFoundError(
                f"Images directory not found: {self._images_dir}")
        if not self._labels_dir.exists():
            raise FileNotFoundError(
                f"Labels directory not found: {self._labels_dir}")
        
        self._image_ids = self._load_image_ids()        
        self._operations_counter = 0
        self._operations_to_refresh = refresh_each
        
        # Check dataset integrity
        if not self.check_integrity():
            raise RuntimeError("Dataset integrity check failed")

    def _refresh_image_ids(self) -> None:
        """
        Refresh the internal set of image IDs from disk based on operations counter.
        
        This method checks the operations counter against the refresh threshold.
        If the threshold is reached, it reloads all image IDs from disk.
        Otherwise, it just increments the counter.
        """
        if self._operations_counter < self._operations_to_refresh:
            self._operations_counter += 1
        else:
            self._operations_counter = 0
            self._image_ids = self._load_image_ids()
        
    def _load_image_ids(self) -> Set[str]:
        """
        Load image IDs by scanning the images directory.

        Returns:
            Set[str]: Set of image IDs (filenames without extensions).
            
        Note:
            Only considers files with .jpg, .jpeg, or .png extensions.
        """
        return {
            os.path.splitext(f)[0]
            for f in os.listdir(self._images_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        }

    def get_image(self, image_id: str) -> Image:
        """
        Load and return the image for the given ID.

        Args:
            image_id (str): The ID of the image to retrieve.

        Returns:
            Image: The image object.

        Raises:
            FileNotFoundError: If the image file does not exist.
        """
        image_path = self._images_dir / f"{image_id}.jpg"
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        return Image(image_path)

    def save_image(
        self, 
        image: Union[np.ndarray, Image, Path], 
        image_id: str, 
        overwrite: bool = False
    ):
        """
        Save an image to the dataset.

        Args:
            image (Union[np.ndarray, Image, Path]): The image to save.
            image_id (str): The ID of the image.
            overwrite (bool): Whether to overwrite an existing image.

        Raises:
            TypeError: If the image type is invalid.
            FileExistsError: If the image already exists and overwrite is False.
        """
        image_dest_filepath = self._images_dir / f"{image_id}.jpg"

        if image_dest_filepath.exists() and not overwrite:
            raise FileExistsError(f"Image {image_id} already exists and overwrite is False.")

        if isinstance(image, Image):
            img_to_save = image
        elif isinstance(image, (np.ndarray, Path)):
            img_to_save = Image(image)
        else:
            raise TypeError(f"Invalid image type: {type(image)}")

        img_to_save.save_jpg(str(image_dest_filepath))
        self._refresh_image_ids()

    def remove_image(self, image_id: str) -> bool:
        """Remove an image and its annotations from the dataset."""
        image_path = self._images_dir / f"{image_id}.jpg"
        label_path = self._labels_dir / f"{image_id}.txt"
        
        removed = False
        try:
            if image_path.exists():
                image_path.unlink()
                removed = True
            if label_path.exists():
                label_path.unlink()
                removed = True
            
            if removed:
                self._refresh_image_ids()
                
            return removed
            
        except Exception as e:
            self.__logger.error(f"Failed to remove {image_id}: {e}")
            return False
        
    def get_annotations(self, image_id: str) -> List[BBoxClassId]:
        """
        Load and return the annotations for the given image ID.

        Args:
            image_id (str): The ID of the image to retrieve annotations for.

        Returns:
            List[BBoxClassId]: List of bounding box annotations.
            Empty list if no annotations exist.
            
        Raises:
            FileNotFoundError: If the image corresponding to image_id doesn't exist.
            ValueError: If the annotations file exists but contains invalid data.
            
        Note:
            Results are cached to improve performance on repeated access.
        """
        label_path = self._labels_dir / f"{image_id}.txt"
        if not label_path.exists():
            return []

        # Load image to get dimensions
        image = self.get_image(image_id)
        image_width, image_height = image.get_width(), image.get_height()

        # Parse annotations
        annotations = []
        for annotation in read_label(label_path):
            class_id, x_center, y_center, width, height = annotation
            annotations.append(BBoxClassId(
                coordinates=[x_center, y_center, width, height],
                class_id=int(class_id),
                class_name=None,
                fmt=BBoxFormat.YOLO,
                img_width=image_width,
                img_height=image_height
            ))

        return annotations

    def save_annotations(
        self, 
        annotations: Union[List, Tuple, BBoxClassId], 
        image_id: str, 
        overwrite: bool = False
    ):
        """
        Save annotations for an image in YOLO format.

        Args:
            annotations: The annotations to save. Can be:
                - BBoxClassId: Single bounding box with class information
                - List/Tuple: Raw YOLO format annotations as (class_id, x, y, w, h)
                where all coordinates are normalized [0-1]
            image_id (str): The ID of the image.
            overwrite (bool): Whether to overwrite existing annotations.

        Raises:
            TypeError: If the annotations type is invalid.
            FileExistsError: If the annotations already exist and overwrite is False.
            ValueError: If coordinates are not normalized or in wrong format.
        """
        label_dest_filepath = self._labels_dir / f"{image_id}.txt"
        if isinstance(annotations, BBoxClassId):
            annotations = annotations.get_coordinates(BBoxFormat.YOLO)
        elif isinstance(annotations, List) or isinstance(annotations, Tuple):
            pass
        else:
            raise TypeError(
                f"Type '{type(annotations)}' not correct for annotations.")

        if label_dest_filepath.exists() and not overwrite:
            raise FileExistsError(
                f"Annotations for {image_id} already exist and overwrite is False.")

        write_label(label_dest_filepath, annotations)

    def save_image_annotations(
        self, 
        image: Union[np.ndarray, Image, Path], 
        image_id: str, 
        annotations: Union[List, Tuple, BBoxClassId],
        overwrite: bool = False
    ) -> None:
        """
        Save both an image and its annotations to the dataset.

        This method saves the provided image to the images directory and its associated 
        annotations to the labels directory. The image will be saved in JPG format and 
        the annotations in YOLO format (normalized coordinates).

        Args:
            image: The image to save. Can be:
                - numpy.ndarray: Raw image data
                - Image: Instance of the Image class
                - Path: Path to an existing image file
            image_id: Unique identifier for the image. Will be used as the filename 
                (without extension) for both the image and its annotations.
            annotations: The annotations to save. Can be:
                - List/Tuple: Raw YOLO format annotations as (class_id, x, y, w, h)
                - BBoxClassId: Single bounding box with class information
                - List[BBoxClassId]: Multiple bounding boxes

        Raises:
            TypeError: If image or annotations are not of supported types
            FileExistsError: If image or annotations already exist and overwrite=False
            ValueError: If the annotations format is invalid
            OSError: If there are issues writing the files to disk
        """
        self.save_image(image, image_id, overwrite=overwrite)
        self.save_annotations(annotations, image_id, overwrite=overwrite)

    def list_images_ids(self, max_entries: Optional[int] = None) -> List[str]:
        """
        List all image IDs in the dataset.

        Args:
            max_entries (Optional[int]): Maximum number of IDs to return.

        Returns:
            List[str]: List of image IDs.
        """
        self._image_ids = self._load_image_ids()
        ids = sorted(self._image_ids)  # Convert to sorted list
        return ids[:max_entries] if max_entries else ids

    def check_integrity(self, stats: dict = None) -> bool:
        """
        Check the integrity of the dataset.
        
        Performs the following checks:
        - All image IDs have corresponding image files
        - All image IDs have corresponding label files
        - All label files contain valid annotations
        
        Args:
            stats (dict, optional): Dictionary to store detailed integrity statistics.
                If provided, will be populated with lists of:
                - missing_images: Images referenced but not found
                - missing_labels: Labels referenced but not found 
                - invalid_annotations: Images with malformed annotations
                
        Returns:
            bool: True if all checks pass, False otherwise.
            
        Note:
            Issues are logged using the class logger.
        """
        if stats is None:
            stats = {
                'missing_images': [], 'missing_labels': [], 'invalid_annotations': []
            }
        
        intact = True
        for image_id in self._image_ids:
            # Check for missing labels
            label_path = _find_matching_label(image_id, self._labels_dir)
            if label_path is None:
                self.__logger.warning(f"Missing label file for image {label_path}")
                stats['missing_labels'].append(label_path)
                intact = False
                
            # Check for missing images
            image_path = _find_matching_image(image_id, self._images_dir)
            if image_id is None:
                self.__logger.warning(f"Missing image file for label {image_path}")
                stats['missing_images'].append(image_path)
                intact = False

            # Check for invalid annotations
            try:
                self.get_annotations(image_id)
            except Exception as e:
                self.__logger.warning(f"Invalid annotations for image {image_id}: {e}")
                stats['invalid_annotations'].append(image_path)
                intact = False

        return intact

    def is_annotation_valid(self, annotation):
        # Not used in this case
        raise NotImplementedError

def create_dataset_structure(root_dir: str):
    """Creates YOLO Darknet directory structure if it doesn't exist.
    
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
        'labels': {
            'train': os.path.join(root_dir, 'labels', 'train'),
            'val': os.path.join(root_dir, 'labels', 'val'),
            'test': os.path.join(root_dir, 'labels', 'test')
        }
    }
    
    # Create directories if they don't exist
    for category in directories.values():
        for dir_path in category.values():
            os.makedirs(dir_path, exist_ok=True)
            
    return directories


def get_dataset_paths(root_dir: str) -> Tuple[str, str, str, str, Optional[str], Optional[str]]:
    """Get directory paths for dataset in YOLO format, including test set if present.
    
    Args:
        root_dir: Dataset root directory
        
    Returns:
        Tuple of (train_images_dir, train_labels_dir, val_images_dir, val_labels_dir, test_images_dir, test_labels_dir)
        Test directories are None if not present.
        
    Raises:
        FileNotFoundError: If required train/val directories don't exist
    """
    # Verify root directory exists
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"Root directory {root_dir} does not exist")
    
    # Build YOLO Darknet paths
    train_images_dir = os.path.join(root_dir, 'images', 'train')
    train_labels_dir = os.path.join(root_dir, 'labels', 'train')
    val_images_dir = os.path.join(root_dir, 'images', 'val')
    val_labels_dir = os.path.join(root_dir, 'labels', 'val')
    test_images_dir = os.path.join(root_dir, 'images', 'test')
    test_labels_dir = os.path.join(root_dir, 'labels', 'test')
    
    # Verify required YOLO Darknet directories exist
    required_dirs = [
        (train_images_dir, "Train Images"),
        (train_labels_dir, "Train Labels"), 
        (val_images_dir, "Validation Images"),
        (val_labels_dir, "Validation Labels")
    ]
    
    for dir_path, dir_name in required_dirs:
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"{dir_name} directory not found at {dir_path}")
    
    # Test directories are optional
    if os.path.exists(test_images_dir) and os.path.exists(test_labels_dir):
        return train_images_dir, train_labels_dir, val_images_dir, val_labels_dir, test_images_dir, test_labels_dir
    else:
        return train_images_dir, train_labels_dir, val_images_dir, val_labels_dir, None, None


def validate_dataset_structure(root_dir: str) -> Dict[str, Dict[str, Union[int, List[str]]]]:
    """Verify directory structure and validate image-label pairs.

    Args:
        root_dir: Dataset root directory

    Returns:
        Dict containing per-split statistics:
            images: Number of images
            labels: Number of labels
            matched: Number of matched pairs
            unmatched_images: List of images without labels
            unmatched_labels: List of labels without images
    """
    train_images_dir, train_labels_dir, val_images_dir, val_labels_dir, test_images_dir, test_labels_dir = get_dataset_paths(root_dir)

    stats = {
        'train': {'images': 0, 'labels': 0, 'matched': 0, 'unmatched_images': [], 'unmatched_labels': []},
        'val': {'images': 0, 'labels': 0, 'matched': 0, 'unmatched_images': [], 'unmatched_labels': []},
    }

    def check_matches(images_dir, labels_dir, split):
        image_files = {os.path.splitext(f)[0] for f in os.listdir(images_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))}
        label_files = {os.path.splitext(f)[0] for f in os.listdir(labels_dir)
                       if f.lower().endswith('.txt')}

        stats[split]['images'] = len(image_files)
        stats[split]['labels'] = len(label_files)
        stats[split]['matched'] = len(image_files & label_files)
        stats[split]['unmatched_images'] = sorted(list(image_files - label_files))
        stats[split]['unmatched_labels'] = sorted(list(label_files - image_files))

    check_matches(train_images_dir, train_labels_dir, 'train')
    check_matches(val_images_dir, val_labels_dir, 'val')

    # Check test split if present
    if test_images_dir and test_labels_dir:
        stats['test'] = {'images': 0, 'labels': 0, 'matched': 0, 'unmatched_images': [], 'unmatched_labels': []}
        check_matches(test_images_dir, test_labels_dir, 'test')

    return stats


def analyze_dataset(root_dir: str, class_names: Optional[List[str]] = None) -> None:
    """Analyze dataset structure and content.
    
    Args:
        root_dir: Dataset root directory
        class_names: Optional list of class names to map class IDs
    """
    structure = validate_dataset_structure(root_dir)
    
    print("\n=== YOLO Dataset Analysis ===")
    
    # Dataset structure analysis
    for split in structure.keys():
        print(f"\n{split.upper()} Set Structure:")
        print(f"- Total images: {structure[split]['images']}")
        print(f"- Total labels: {structure[split]['labels']}")
        print(f"- Matched pairs: {structure[split]['matched']}")
        
        if structure[split]['unmatched_images']:
            print(f"- Images without labels ({len(structure[split]['unmatched_images'])}):")
            for img in structure[split]['unmatched_images'][:5]:
                print(f"  * {img}")
            if len(structure[split]['unmatched_images']) > 5:
                print("    ...")
                
        if structure[split]['unmatched_labels']:
            print(f"- Labels without images ({len(structure[split]['unmatched_labels'])}):")
            for lbl in structure[split]['unmatched_labels'][:5]:
                print(f"  * {lbl}")
            if len(structure[split]['unmatched_labels']) > 5:
                print("    ...")
        
        # Object count analysis
        labels_dir = os.path.join(root_dir, 'labels', split)
        if os.path.exists(labels_dir):
            counts, total_images = count_objects(labels_dir, class_names)
            
            print(f"\n{split.upper()} Set Objects:")
            print(f"- Total annotated images: {total_images}")
            print("- Objects per class:")
            for class_id, count in counts.items():
                class_name = class_id if isinstance(class_id, str) else f"Class {class_id}"
                print(f"  * {class_name}: {count}")


def read_label(label_file: Union[str, Path]) -> list:
    """Parse YOLO Darknet format labels from a text file.

    Args:
        label_file (str, Path): Path to the YOLO Darknet label file

    Returns:
        list: List of tuples containing (class_id, x, y, width, height) for each bounding box
    """
    labels = []
    with open(label_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            box_info = line.strip().split()
            if len(box_info) != 5:
                raise ValueError("Invalid label format: must have 5 values")
            
            class_id = int(box_info[0])
            xc = float(box_info[1])
            yc = float(box_info[2])
            w = float(box_info[3])
            h = float(box_info[4])
            
            if not all(0 <= coord <= 1 for coord in (xc, yc)):
                raise ValueError("Coordinates must be normalized [0-1]")
            if not all(0 < dim <= 1 for dim in (w, h)):
                raise ValueError("Width and height must be normalized (0-1]")
            
            labels.append((class_id, xc, yc, w, h))
    return labels


def write_label(label_file: str, annotations: Iterable[Union[list, tuple]], file_mode: str = 'w') -> None:
    """Write object detection labels in YOLO format.

    Args:
        label_file: Path to output label file
        labels: Sequence of (class_id, x_center, y_center, width, height) tuples 
               where coordinates are normalized to [0-1]
        file_mode: 'w' to overwrite file, 'a' to append

    Raises:
        ValueError: If coordinates are not normalized [0-1] or invalid format
    """
    with open(label_file, file_mode) as file:
        for annotation in annotations:
            if len(annotation) != 5:
                raise ValueError("Each label must have 5 values: class_id, x, y, w, h")
                
            class_id, xc, yc, w, h = annotation
            
            # Validate coordinates
            if not all(0 <= coord <= 1 for coord in (xc, yc, w, h)):
                raise ValueError("Coordinates must be normalized [0-1]")
            
            file.write(f'{" ".join(map(str, annotation))}\n')


def list_all_labels(label_dir: str) -> List[str]:
    """
    List all labels from text files in a directory.
    
    Args:
        label_dir: Directory path containing label files
        
    Returns:
        List of labels read from all .txt files in the directory
        
    Raises:
        FileNotFoundError: If the directory doesn't exist
        PermissionError: If there's no read access to the directory
    """
    # Get all .txt files from the directory using list comprehension
    # Creates a list of full paths to each .txt file
    labels_files = [
        os.path.join(label_dir, file) 
        for file in os.listdir(label_dir) 
        if file.endswith('.txt')
    ]
        
    # Read all labels using list comprehension
    # Calls read_label() for each file path and returns the results as a list
    return [
        read_label(label_file) for label_file in labels_files
    ]


def replace_label_class(
   old_class_id: int,
   new_class_id: int, 
   label_dir: str = None, 
   filepath_list: list = None,
   verbose: bool = False 
) -> tuple[int, list]:
   """Replace specified class ID in YOLO Darknet label files with a new ID.

   Args:
       old_class_id: Original class ID to replace
       new_class_id: New class ID to assign
       label_dir: Directory containing YOLO Darknet labels (used if filepath_list not provided)
       filepath_list: List of specific label files to process (overrides label_dir)
       verbose: Enable detailed progress output with tqdm

   Returns:
       tuple: (Number of modified files, List of modified file paths)

   Raises:
       ValueError: If neither label_dir nor filepath_list is provided
       FileNotFoundError: If label_dir does not exist
   """
   if not label_dir and not filepath_list:
       raise ValueError("Must provide either label_dir or filepath_list")
       
   if label_dir and not os.path.exists(label_dir):
       raise FileNotFoundError(f"Directory not found: {label_dir}")

   logger = configure_logging.get_logger(__name__)

   # Create file list if not provided
   if label_dir and not filepath_list:
       filepath_list = [
           os.path.join(root, file) 
           for root, _, files in os.walk(label_dir)
           for file in files if file.endswith('.txt')
       ]

   modified_count = 0
   modified_files = []

   if verbose:
       configure_logging.enable_tqdm()

   for filepath in tqdm(filepath_list, disable=not verbose):
       if not os.path.exists(filepath):
           logger.warning(f'File not found: {filepath}')
           continue
       
       # Read and modify labels
       modified = False
       with open(filepath, 'r') as f:
           lines = f.readlines()
           
       modified_lines = []
       for line in lines:
           components = line.strip().split()
           if len(components) >= 5 and int(components[0]) == old_class_id:
               components[0] = str(new_class_id)
               modified = True
           modified_lines.append(" ".join(components) + "\n")
       
       # Write modified content only if changes made
       if modified:
           with open(filepath, 'w') as f:
               f.writelines(modified_lines)
           modified_files.append(filepath)
           modified_count += 1
           
           if verbose:
               logger.info(f"Modified {os.path.basename(filepath)}")
   
   if verbose:
       configure_logging.disable_tqdm()
   
   return modified_count, modified_files


def shift_class_ids(label_folder: str, shift_value: int, output_folder: str = None) -> None:
    """Shift all class IDs in YOLO Darknet label files by a constant value.

    Args:
        label_folder: Input directory containing labels
        shift_value: Integer value to add to all class IDs
        output_folder: Output directory for modified labels. If None, modifies in-place
    """
    logger = configure_logging.get_logger(__name__)
    
    if output_folder is None:
        output_folder = label_folder

    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(label_folder):
        if filename.endswith(".txt"):
            input_file = os.path.join(label_folder, filename)
            output_file = os.path.join(output_folder, filename)

            with open(input_file, "r") as infile:
                lines = infile.readlines()

            updated_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    logger.warning(f"Skipping invalid line in {filename}: {line}")
                    continue

                # Adjust the class ID
                class_id = int(parts[0]) + shift_value
                updated_line = f"{class_id} " + " ".join(parts[1:])
                updated_lines.append(updated_line)

            # Save the updated labels
            with open(output_file, "w") as outfile:
                outfile.write("\n".join(updated_lines))

            logger.info(f"Processed {filename}: saved to {output_file}")


def convert_to_coco_annotations(image_dir: str, label_dir: str, classes: list[str], output_json: str):
    """Convert object detection annotations from YOLO Darknet to COCO JSON format.
    
    YOLO Darknet format: class_id x_center y_center width height (normalized coordinates)
    COCO format: {
        "images": [{id, file_name, width, height}],
        "annotations": [{id, image_id, category_id, bbox, area, iscrowd}],
        "categories": [{id, name}]
    }

    Args:
        image_dir (str): Directory containing source images
        label_dir (str): Directory containing YOLO Darknet .txt annotation files 
        classes (list[str]): List of class names matching YOLO Darknet indices
        output_json (str): Path for output COCO JSON file
    """
    from saltup.ai.object_detection.utils.bbox import yolo_to_coco_bbox
    
    images = []
    annotations = []
    categories = []

    # Create categories based on the number of classes
    for idx, class_name in enumerate(classes):
        categories.append({
            "id": idx + 1,
            "name": class_name,
            "supercategory": "none"
        })

    ann_id = 0
    img_id = 0

    for img_filename in os.listdir(image_dir):
        if img_filename.endswith((".jpg", ".png")):
            img_id += 1
            img_path = os.path.join(image_dir, img_filename)
            img = Image.open(img_path)
            img_width, img_height = img.size

            # Add image to the COCO images array
            images.append({
                "id": img_id,
                "file_name": img_filename,
                "width": img_width,
                "height": img_height
            })

            # Load the corresponding YOLO Darknet label file
            label_filename = os.path.splitext(img_filename)[0] + ".txt"
            label_path = os.path.join(label_dir, label_filename)

            if os.path.exists(label_path):
                # Use read_label to get the boxes
                yolo_boxes = read_label(label_path)
                
                for yolo_box in yolo_boxes:
                    x, y, w, h, class_id = yolo_box
                    # Convert YOLO Darknet bbox to COCO bbox
                    coco_bbox = yolo_to_coco_bbox([x, y, w, h], img_width, img_height)
                    area = coco_bbox[2] * coco_bbox[3]  # width * height

                    annotations.append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": class_id + 1,  # COCO category id starts from 1
                        "bbox": coco_bbox,
                        "area": area,
                        "iscrowd": 0,
                        "segmentation": []  # Empty list since YOLO Darknet does not use segmentation
                    })
                    ann_id += 1

    coco_data = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    try:
        # Write the final output to a json file
        with open(output_json, "w") as outfile:
            json.dump(coco_data, outfile, indent=4)
        print("Conversion succeeded")
    except:
        print("Conversion failed")


def split_dataset(
    class_to_images: Dict[int, List], 
    split_ratio: float = 0.8, 
    split_val_ratio: float = 0.5,
    max_images_per_class: Optional[int] = None
) -> Tuple[List, List, List]:
    """Split dataset into train, validation and test sets.

    First splits data into training and remaining sets using split_ratio.
    Then divides remaining data into validation/test using split_val_ratio.
    Images are randomly selected up to optional max_images_per_class limit.

    Args:
        class_to_images: Dict mapping class IDs to lists of (image_path, label_path) tuples
        split_ratio: Proportion for training set (0.0-1.0) 
        split_val_ratio: Proportion for validation vs test split (0.0-1.0)
        max_images_per_class: Max images per class (None = no limit)

    Returns:
        Tuple of (train, validation, test) lists containing (image_path, label_path) pairs

    Raises:
        ValueError: If ratios not in [0,1] or max_images_per_class negative
   
    Examples:
        >>> class_data = {0: ['img1.jpg', 'img2.jpg'], 1: ['img3.jpg', 'img4.jpg']}
        >>> train, val, test = split_dataset(class_data, min_images_per_class=2)
    """
    # Validate input ratios
    if not (0.0 <= split_ratio <= 1.0 and 0.0 <= split_val_ratio <= 1.0):
        raise ValueError("Split ratios must be in range [0, 1]")
    if max_images_per_class is not None and max_images_per_class < 0:
        raise ValueError("max_images_per_class must be positive or None")

    # Track which split each image belongs to
    image_to_split = {}

    for class_id, images in class_to_images.items():
        random.shuffle(images)
        limit = min(max_images_per_class, len(images)) if max_images_per_class else len(images)

        # Assign images to train set
        train_count = int(limit * split_ratio)
        for image in images[:train_count]:
            if image not in image_to_split:
                image_to_split[image] = "train"

        # Assign remaining images to val/test set
        val_test_images = images[train_count:limit]
        random.shuffle(val_test_images)
        mid_point = int(len(val_test_images) * split_val_ratio)
        for image in val_test_images[:mid_point]:
            if image not in image_to_split:
                image_to_split[image] = "val"
        for image in val_test_images[mid_point:]:
            if image not in image_to_split:
                image_to_split[image] = "test"

    # Create final splits
    train_set = [image for image, split in image_to_split.items() if split == "train"]
    val_set = [image for image, split in image_to_split.items() if split == "val"]
    test_set = [image for image, split in image_to_split.items() if split == "test"]

    return train_set, val_set, test_set

def split_and_organize_dataset(
    labels_dir: str,
    images_dir: str, 
    max_images_per_class: int = 0,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1
) -> None:
    """Split YOLO dataset into train/val/test directories.

    Randomly splits dataset according to provided ratios and organizes files
    into standard YOLO directory structure.

    Args:
        labels_dir: YOLO label files directory (.txt)
        images_dir: Image files directory (.jpg/.jpeg)
        max_images_per_class: Max images per class (0 = no limit)
        train_ratio: Training set proportion (e.g. 0.7)
        val_ratio: Validation set proportion (e.g. 0.2) 
        test_ratio: Test set proportion (e.g. 0.1)

    Note:
        - Moves files to train/val/test subdirs (not copied)
        - train_ratio + val_ratio + test_ratio should equal 1.0
    """
    class_to_images = _image_per_class_id(labels_dir, images_dir)
    
    # Split dataset
    train_set, val_set, test_set = split_dataset(
        class_to_images,
        max_images_per_class,
        train_ratio,
        val_ratio,
        test_ratio
    )

    print(f"Training set size: {len(train_set)} images")
    print(f"Validation set size: {len(val_set)} images")
    print(f"Test set size: {len(test_set)} images")

    # Create subfolders
    subdirs = {
        "train": {"images": os.path.join(images_dir, 'train'), 
                 "labels": os.path.join(labels_dir, 'train')},
        "val": {"images": os.path.join(images_dir, 'val'),
                "labels": os.path.join(labels_dir, 'val')},
        "test": {"images": os.path.join(images_dir, 'test'),
                 "labels": os.path.join(labels_dir, 'test')}
    }

    for paths in subdirs.values():
        os.makedirs(paths["images"], exist_ok=True)
        os.makedirs(paths["labels"], exist_ok=True)

    # Move files
    moved_files = set()
    for dataset, split_name in [(train_set, "train"), 
                              (val_set, "val"), 
                              (test_set, "test")]:
        for image_path, label_path in dataset:
            if image_path not in moved_files and label_path not in moved_files:
                image_dest = os.path.join(subdirs[split_name]["images"], 
                                        os.path.basename(image_path))
                label_dest = os.path.join(subdirs[split_name]["labels"], 
                                        os.path.basename(label_path))

                shutil.move(image_path, image_dest)
                shutil.move(label_path, label_dest)
                moved_files.update([image_path, label_path])


def count_objects(
    labels_dir: str,
    class_names: list = None,
    verbose: bool = False
) -> tuple[Dict[Union[int, str], int], int]:
    """Count labels instances and annotated images in YOLO dataset.

    Args:
        labels_dir: Path to directory containing YOLO label files
        class_names: Optional list of class names to map IDs to names
        verbose: Enable progress output

    Returns:
        Tuple of:
            Dict mapping class ID/name to labels count
            Number of annotated images
    """
    label_counts = Counter()
    
    txt_label_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(labels_dir)
        for file in files if file.endswith('.txt')
    ]
    
    txt_label_files = tqdm(txt_label_files, desc='Processing YOLO labels', disable=not verbose)
    for label_file in txt_label_files:
        lbls = read_label(label_file)
        for lbl in lbls:
            class_id = lbl[0]
            label_counts[class_id] += 1

    if class_names:
        label_counts = {
            class_names[class_id]: count 
            for class_id, count in label_counts.items()
        }

    return dict(label_counts), len(txt_label_files)


def create_symlinks_by_class(
    imgs_dirpath: str, 
    lbls_dirpath: str, 
    dest_dir: str, 
    class_names: Optional[List[str]] = None
) -> None:
    """Create symbolic links for images and labels organized by class.
    
    Only creates class directories if there are files to link for that class.

    Args:
        imgs_dirpath (str): Path to directory containing images
        lbls_dirpath (str): Path to directory containing label files
        dest_dir (str): Destination directory for symlinks
        class_names (Optional[List[str]]): List of class names, defaults to None
    """
    logger = configure_logging.get_logger(__name__)
    
    # Validate input directories
    imgs_path = Path(imgs_dirpath)
    lbls_path = Path(lbls_dirpath)
    dest_path = Path(dest_dir)

    if not imgs_path.is_dir():
        logger.error(f"Images directory not found: {imgs_path}")
        raise ValueError("Input images path must be an existing directory")

    if not lbls_path.is_dir():
        logger.error(f"Labels directory not found: {lbls_path}")
        raise ValueError("Input labels path must be an existing directory")

    # Get label files
    lbl_files = list(lbls_path.glob('*.txt'))
    
    if not lbl_files:
        logger.warning(f"No label files found in {lbls_path}")
        return

    # If no class names provided, use unique classes from label files
    if class_names is None:
        class_names = _extract_unique_classes(lbl_files)
        logger.info(f"Extracted classes: {class_names}")

    # Create destination directory
    dest_path.mkdir(parents=True, exist_ok=True)

    # Create classes.names file
    class_names_file = dest_path / 'classes.names'
    with class_names_file.open('w') as f:
        f.write('\n'.join(map(str, class_names)))
    logger.info(f"Created classes file: {class_names_file}")

    # Create labels map
    labels_map = _create_labels_map(lbl_files, class_names)
    
    # Find which classes have files
    classes_with_files = set()
    for lbl_file in labels_map:
        # Get the actual image file path
        img_file = Path(_find_matching_image(lbl_file.stem, str(imgs_path)))
        
        # Only count classes with both image and label files
        if img_file is not None and img_file.exists():
            classes_with_files.update(labels_map[lbl_file].keys())
    
    logger.info(f"Found {len(classes_with_files)} classes with files to link")
    
    # Create only necessary class directories
    for class_name in classes_with_files:
        class_dir = dest_path / str(class_name)
        class_dir.mkdir(exist_ok=True)
        logger.debug(f"Created class directory: {class_dir}")

    # Create symbolic links
    symlink_count = 0
    for lbl_file in labels_map:
        # Get actual paths for image and label files
        img_file = Path(_find_matching_image(lbl_file.stem, str(imgs_path)))
        lbl_file = lbls_path / lbl_file.name

        if img_file is None or not img_file.exists():
            logger.warning(f"No matching image found for {lbl_file.name}")
            continue

        for class_name in labels_map[lbl_file]:
            try:
                # Create symlinks
                lbl_symlink = dest_path / str(class_name) / lbl_file.name
                img_symlink = dest_path / str(class_name) / img_file.name

                os.symlink(str(lbl_file.absolute()), str(lbl_symlink))
                os.symlink(str(img_file.absolute()), str(img_symlink))

                symlink_count += 2
                logger.debug(f"Created symlinks for {lbl_file.name} and {img_file.name}")
            except FileExistsError:
                logger.warning(f"Symlink already exists for {lbl_file.name}")
            except PermissionError:
                logger.error(f"Permission denied creating symlink for {lbl_file.name}")

    logger.info(f"Created {symlink_count} symlinks across {len(classes_with_files)} classes")


def find_image_label_pairs(labels_dir: str, images_dir: str) -> Tuple[List[str], List[List[str]], List[str]]:
    """Find matching image-label pairs and identify unmatched labels.
    
    Args:
        labels_dir: Directory containing label files (.txt)
        images_dir: Directory containing image files (.jpg/.jpeg)
        
    Returns:
        Tuple containing:
        - List of matched image paths
        - List of parsed labels for matched images  
        - List of label paths with no matching image
        
    Raises:
        FileNotFoundError: If either directory doesn't exist
    """
    # Validate directories
    if not os.path.exists(labels_dir):
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    matched_images = []
    matched_labels = []
    unmatched_labels = []

    # Find matches for each label file
    for label_file in os.listdir(labels_dir):
        if not label_file.endswith('.txt'):
            continue
            
        base_name = os.path.splitext(label_file)[0]
        label_path = os.path.join(labels_dir, label_file)
        image_path = _find_matching_image(base_name, images_dir)
        
        if image_path:
            matched_images.append(image_path)
            matched_labels.append(read_label(label_path))
        else:
            unmatched_labels.append(label_path)

    return matched_images, matched_labels, unmatched_labels


def _extract_unique_classes(label_files: List[Path]) -> List[str]:
    """Extract unique classes from label files.

    Args:
        label_files: List of label file paths

    Returns:
        List of unique class names
    """
    return list({
        int(label[0])
        for file in label_files
        for label in read_label(str(file))
    })


def _create_labels_map(
   lbl_files: List[Path], 
   class_names: Optional[List[str]] = None
) -> Dict[Path, Dict[Union[str, int], int]]:
   """Create mapping between label files and their class counts.

   Maps each label file to a dictionary tracking how many times each class appears.
   Class identifiers can be either numeric IDs or names if class_names is provided.

   Args:
       lbl_files: List of YOLO format label file paths
       class_names: Optional list to map class IDs to names. None means use numeric IDs

   Returns:
       Dictionary mapping each label file to its class distribution count.
       Inner dictionary maps either class names (str) or IDs (int) to counts

   Example returned structure:
       {
           Path('img1.txt'): {'person': 2, 'car': 1},  # with class_names
           Path('img2.txt'): {0: 2, 1: 1}              # without class_names
       }
   """
   labels_map = {}
   for lbl_file in lbl_files:
       labels_map[lbl_file] = defaultdict(int)
       labels = read_label(str(lbl_file))
       for label in labels:
           class_id = int(label[0])
           if class_names:
               labels_map[lbl_file][class_names[class_id]] += 1
           else:
               labels_map[lbl_file][class_id] += 1
   
   return labels_map


def _image_per_class_id(labels_dir: str, images_dir: str) -> dict:
    """Group images by their class IDs from YOLO labels.

    Args:
        labels_dir (str): Directory containing YOLO label files
        images_dir (str): Directory containing image files

    Returns:
        dict: Dictionary mapping class IDs to lists of (image_path, label_path) tuples
    """
    class_to_images = defaultdict(list)

    for label_file in os.listdir(labels_dir):
        if label_file.endswith('.txt'):
            base_name = os.path.splitext(label_file)[0]
            label_path = os.path.join(labels_dir, label_file)
            image_path = _find_matching_image(base_name, images_dir)
            
            # Process only if we found a matching image
            if image_path:
                labels = read_label(label_path)
                classes_in_image = set(int(label[0]) for label in labels)
                for class_id in classes_in_image:
                    class_to_images[class_id].append((image_path, label_path))

    return class_to_images


def _find_matching_image(base_name: str, images_dir: str) -> Optional[Path]:
    """Find matching image file for a given base name.
    
    Args:
        base_name: Base name without extension
        images_dir: Directory containing images
        
    Returns:
        Full path to matching image if found, None otherwise
    """
    for ext in ('.jpg', '.jpeg'):
        image_path = os.path.join(images_dir, f"{base_name}{ext}")
        if os.path.exists(image_path):
            return Path(image_path)
    return None


def _find_matching_label(base_name: str, labels_dir: str) -> Optional[Path]:
    """Find matching label file for a given base name.
    
    Args:
        base_name: Base name without extension
        labels_dir: Directory containing YOLO labels
        
    Returns:
        Full path to matching label if found, None otherwise
    """
    label_path = Path(labels_dir) / f"{base_name}.txt"
    return label_path if label_path.exists() else None
