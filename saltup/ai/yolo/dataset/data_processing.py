from PIL import Image
import os
import json
import random
import shutil
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict, Counter
from typing import Iterable, Union, List, Dict, Optional

from saltup.utils import configure_logging


def read_yolo_label(label_file: str) -> list:
    """Parse YOLO format labels from a text file.

    Args:
        label_file (str): Path to the YOLO label file

    Returns:
        list: List of tuples containing (class_id, x, y, width, height) for each bounding box
    """
    labels = []
    with open(label_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            box_info = line.strip().split()
            class_id = int(box_info[0])
            x = float(box_info[1])
            y = float(box_info[2])
            w = float(box_info[3])
            h = float(box_info[4])
            labels.append((class_id, x, y, w, h))
    return labels


def write_yolo_label(label_file: str, labels: Iterable[Union[list, tuple]], file_mode: str = 'w') -> None:
   """Write object detection labels in YOLO format.

   Args:
       label_file: Path to output label file
       labels: Sequence of (class_id, x_center, y_center, width, height) tuples 
              where coordinates are normalized to [0-1]
       file_mode: 'w' to overwrite file, 'a' to append
   """
   # TODO: Validate label format and coordinate ranges (?)
   with open(label_file, file_mode) as file:
       for label in labels:
           file.write(f'{" ".join(map(str, label))}\n')


def replace_yolo_label_class(
   old_class_id: int,
   new_class_id: int, 
   label_dir: str = None, 
   filepath_list: list = None,
   verbose: bool = False 
) -> tuple[int, list]:
   """Replace specified class ID in YOLO label files with a new ID.

   Args:
       old_class_id: Original class ID to replace
       new_class_id: New class ID to assign
       label_dir: Directory containing YOLO labels (used if filepath_list not provided)
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


def shift_yolo_class_ids(label_folder: str, shift_value: int, output_folder: str = None) -> None:
    """Shift all class IDs in YOLO label files by a constant value.

    Args:
        label_folder: Input directory containing YOLO labels
        shift_value: Integer value to add to all class IDs
        output_folder: Output directory for modified labels. If None, modifies in-place
    """
    if output_folder is None:
        output_folder = label_folder

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

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
                    print(f"Skipping invalid line in {filename}: {line}")
                    continue

                # Adjust the class ID
                class_id = int(parts[0]) + shift_value
                updated_line = f"{class_id} " + " ".join(parts[1:])
                updated_lines.append(updated_line)

            # Save the updated labels
            with open(output_file, "w") as outfile:
                outfile.write("\n".join(updated_lines))

            print(f"Processed {filename}: saved to {output_file}")


def convert_yolo_to_coco_annotations_file(image_dir: str, label_dir: str, classes: list[str], output_json: str):
    """
    Convert object detection annotations from YOLO to COCO JSON format.
    
    YOLO format: class_id x_center y_center width height (normalized coordinates)
    COCO format: {
        "images": [{id, file_name, width, height}],
        "annotations": [{id, image_id, category_id, bbox, area, iscrowd}],
        "categories": [{id, name}]
    }

    Args:
        image_dir (str): Directory containing source images
        label_dir (str): Directory containing YOLO .txt annotation files 
        classes (list[str]): List of class names matching YOLO indices
        output_json (str): Path for output COCO JSON file
    """
    from saltup.ai.yolo.bbox_utils import yolo_to_coco_bbox
    
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

            # Load the corresponding YOLO label file
            label_filename = os.path.splitext(img_filename)[0] + ".txt"
            label_path = os.path.join(label_dir, label_filename)

            if os.path.exists(label_path):
                # Use read_yolo_label to get the boxes
                yolo_boxes = read_yolo_label(label_path)
                
                for yolo_box in yolo_boxes:
                    x, y, w, h, class_id = yolo_box
                    # Convert YOLO bbox to COCO bbox
                    coco_bbox = yolo_to_coco_bbox([x, y, w, h], img_width, img_height)
                    area = coco_bbox[2] * coco_bbox[3]  # width * height

                    annotations.append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": class_id + 1,  # COCO category id starts from 1
                        "bbox": coco_bbox,
                        "area": area,
                        "iscrowd": 0,
                        "segmentation": []  # Empty list since YOLO does not use segmentation
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


def split_dataset(class_to_images, min_images_per_class, split_ratio=0.8) -> list:
    """Split dataset into training, validation, and test sets while maintaining class distribution.

    Args:
        class_to_images (dict): Mapping of class IDs to image paths
        min_images_per_class (int): Maximum number of images to use per class (0 for no limit)
        split_ratio (float): Ratio for training set size (remaining split equally between val/test)

    Returns:
        tuple: Lists of (image_path, label_path) tuples for train, validation, and test sets
    """
    train_set, val_test_set = set(), set()

    for class_id, images in class_to_images.items():
        random.shuffle(images)
        if min_images_per_class != 0:
            limit = min(min_images_per_class, len(images))
        else:
            limit = len(images)

        # Determine training and val/test counts
        train_count = int(limit * split_ratio)
        train_set.update(images[:train_count])
        val_test_set.update(images[train_count:limit])

    # Split val_test_set into validation and test sets (50% each)
    val_test_list = list(val_test_set)
    random.shuffle(val_test_list)
    mid_point = len(val_test_list) // 2
    val_set = val_test_list[:mid_point]
    test_set = val_test_list[mid_point:]

    return list(train_set), list(val_set), list(test_set)


def create_subfolders_and_move_files(train_set: list, val_set: list, test_set: list, images_dir: str, labels_dir: str):
    """Create train/val/test subdirectories and move files while preventing duplicates.

    Args:
        train_set (list): List of (image_path, label_path) for training set
        val_set (list): List of (image_path, label_path) for validation set
        test_set (list): List of (image_path, label_path) for test set
        images_dir (str): Root directory for images
        labels_dir (str): Root directory for labels
    """
    subdirs = {
        "train": {"images": os.path.join(images_dir, 'train'), "labels": os.path.join(labels_dir, 'train')},
        "val": {"images": os.path.join(images_dir, 'val'), "labels": os.path.join(labels_dir, 'val')},
        "test": {"images": os.path.join(images_dir, 'test'), "labels": os.path.join(labels_dir, 'test')}
    }

    # Create subdirectories if they don't already exist
    for key in subdirs:
        os.makedirs(subdirs[key]["images"], exist_ok=True)
        os.makedirs(subdirs[key]["labels"], exist_ok=True)

    # Set to track files that have already been moved
    moved_files = set()

    # Helper function to move an image and label to the respective folders if not already moved
    def move_file(image_path, label_path, subdir):
        # Check if the file has already been moved
        if image_path not in moved_files and label_path not in moved_files:
            image_dest = os.path.join(subdirs[subdir]["images"], os.path.basename(image_path))
            label_dest = os.path.join(subdirs[subdir]["labels"], os.path.basename(label_path))

            # Move image and label files to the destination
            shutil.move(image_path, image_dest)
            shutil.move(label_path, label_dest)

            # Mark as moved
            moved_files.add(image_path)
            moved_files.add(label_path)

    # Move files into respective directories without duplicating
    for image_path, label_path in train_set:
        move_file(image_path, label_path, "train")
    for image_path, label_path in val_set:
        move_file(image_path, label_path, "val")
    for image_path, label_path in test_set:
        move_file(image_path, label_path, "test")


def split_yolo_dataset(labels_dir, images_dir, min_images_per_class=0, train_ratio=0.8):
    """Split YOLO dataset into train/val/test sets and organize into subdirectories.

    Args:
        labels_dir (str): Directory containing YOLO label files
        images_dir (str): Directory containing image files
        min_images_per_class (int): Maximum number of images to use per class (0 for no limit)
        train_ratio (float): Ratio for training set size (remaining split equally between val/test)
    """
    class_to_images = _image_per_class_id(labels_dir, images_dir)
    
    # Determine the minimum image count across all classes if not provided
    #if min_images_per_class is None:
        #min_images_per_class = min(len(images) for images in class_to_images.values())

    # Split the dataset into training, validation, and test sets
    train_set, val_set, test_set = split_dataset(class_to_images, min_images_per_class, train_ratio)

    print(f"Training set size: {len(train_set)} images")
    print(f"Validation set size: {len(val_set)} images")
    print(f"Test set size: {len(test_set)} images")

    # Create subfolders and move files without duplicates
    create_subfolders_and_move_files(train_set, val_set, test_set, images_dir, labels_dir)


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
            image_file = label_file.replace('.txt', '.jpg')  # Assuming images are .jpg files
            image_path = os.path.join(images_dir, image_file)
            label_path = os.path.join(labels_dir, label_file)

            # Check if the corresponding image exists in the images directory
            if os.path.exists(image_path):
                with open(label_path, 'r') as f:
                    classes_in_image = set(int(line.split()[0]) for line in f)  # Extract class IDs
                    for class_id in classes_in_image:
                        class_to_images[class_id].append((image_path, label_path))

    return class_to_images


def counts_labels(labels_dir: str, format_type: str = 'yolo', class_names: list = None, verbose: bool = False) -> tuple:
    """
    Counts the occurrences of each label in the specified dataset annotation files and the number of annotated images.

    Parameters:
        labels_dir (str): Path to the directory containing annotation label files.
        format_type (str): The format of the annotation files. Supported values are:
                           - 'yolo' for YOLO format (default).
                           - 'coco' for COCO JSON format.
        class_names (list, optional): A list of class names where the index corresponds to the class ID.
                                      If provided, the output dictionary will use class names instead of numeric IDs.
        verbose (bool): If True, displays progress information during processing (default: False).

    Returns:
        tuple:
            dict: A dictionary where keys are class IDs (or class names if `class_names` is provided) 
                  and values are the counts of occurrences.
            int: The total number of annotated images.

    Raises:
        ValueError: If an unsupported format type is provided.
    """

    # Initialize a counter for label occurrences
    label_counts = Counter()
    num_annotated_images = 0
    format_type = format_type.lower()  # Ensure the format type is case-insensitive

    # Process labels based on the specified annotation format
    if format_type == 'yolo':
        # YOLO format: Each file corresponds to one image
        txt_label_files = [
            os.path.join(root, file)
            for root, _, files in os.walk(labels_dir)
            for file in files if file.endswith('.txt')
        ]
        num_annotated_images = len(txt_label_files)  # Each label file corresponds to one annotated image

        if verbose:
            txt_label_files = tqdm(txt_label_files, desc='Processing YOLO label files')

        for label_file in txt_label_files:
            with open(label_file, 'r') as file:
                for line in file:
                    # Parse the class ID (first element on each line)
                    class_id = int(line.split()[0])
                    label_counts[class_id] += 1

    elif format_type == 'coco':
        # COCO format: A single JSON file contains multiple annotations
        json_label_files = [
            os.path.join(root, file)
            for root, _, files in os.walk(labels_dir)
            for file in files if file.endswith('.json')
        ]

        for count, label_file in enumerate(json_label_files, start=1):
            with open(label_file, 'r') as file:
                data = json.load(file)

                # Count unique image IDs from the "annotations" key
                image_ids = set(annotation['image_id'] for annotation in data.get("annotations", []))
                num_annotated_images += len(image_ids)  # Update the total count of annotated images

                annotations = data.get("annotations", [])
                if verbose:
                    if len(json_label_files) > 1:
                        desc = f"Processing {count}/{len(json_label_files)} COCO label files"
                    else:
                        desc = "Processing COCO labels"
                    annotations = tqdm(annotations, desc=desc)

                for annotation in annotations:
                    class_id = annotation['category_id']
                    label_counts[class_id] += 1
    else:
        raise ValueError("Unsupported format type. Use 'yolo' or 'coco'.")

    # Replace class IDs with class names if provided
    if class_names:
        label_counts = {class_names[class_id]: count for class_id, count in label_counts.items()}

    return dict(label_counts), num_annotated_images

def create_symlinks_by_class(
    imgs_dirpath: str, 
    lbls_dirpath: str, 
    dest_dir: str, 
    class_names: Optional[List[str]] = None
) -> None:
    """
    Create symbolic links for images and labels organized by class.

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
        f.write('\n'.join(class_names))
    logger.info(f"Created classes file: {class_names_file}")

    # Create class-specific directories
    for class_name in class_names:
        class_dir = dest_path / class_name
        class_dir.mkdir(exist_ok=True)
        logger.debug(f"Created class directory: {class_dir}")

    # Create labels map and symbolic links
    labels_map = _create_labels_map(lbl_files, class_names)

    symlink_count = 0
    for lbl_file in labels_map:
        # Get actual paths for image and label files
        img_file = imgs_path / lbl_file.with_suffix(".jpg").name
        lbl_file = lbls_path / lbl_file.name

        for class_name in labels_map[lbl_file]:
            if img_file.exists():
                try:
                    # Create symlinks
                    lbl_symlink = dest_path / class_name / lbl_file.name
                    img_symlink = dest_path / class_name / img_file.name

                    os.symlink(str(lbl_file.absolute()), str(lbl_symlink))
                    os.symlink(str(img_file.absolute()), str(img_symlink))

                    symlink_count += 2
                    logger.debug(f"Created symlinks for {lbl_file.name} and {img_file.name}")
                except FileExistsError:
                    logger.warning(f"Symlink already exists for {lbl_file.name}")
                except PermissionError:
                    logger.error(f"Permission denied creating symlink for {lbl_file.name}")

    logger.info(f"Created {symlink_count} symlinks across {len(class_names)} classes")


def _extract_unique_classes(lbl_files: List[Path]) -> List[str]:
    """
    Extract unique classes from YOLO label files.

    Args:
        lbl_files (List[Path]): List of label file paths

    Returns:
        List[str]: Unique class names
    """
    unique_classes = set()
    for lbl_file in lbl_files:
        with lbl_file.open('r') as f:
            unique_classes.update(
                line.split()[0] for line in f
            )
    return list(unique_classes)

def _create_labels_map(
    lbl_files: List[Path], 
    class_names: List[str]
) -> Dict[Path, Dict[str, int]]:
    """
    Create a map of label files to their class distributions.

    Args:
        lbl_files (List[Path]): List of label file paths
        class_names (List[str]): List of class names

    Returns:
        Dict[Path, Dict[str, int]]: Map of label files to class counts
    """
    labels_map = {}
    for lbl_file in lbl_files:
        labels_map[lbl_file] = defaultdict(int)
        with lbl_file.open('r') as f:
            for line in f:
                class_id = line.split()[0]
                if class_id in class_names:
                    labels_map[lbl_file][class_id] += 1
    
    return labels_map

