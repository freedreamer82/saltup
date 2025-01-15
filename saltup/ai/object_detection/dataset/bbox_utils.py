"""
Bounding Box Utility Functions
==============================

This module provides utilities for handling different bounding box formats and their conversions.

Supported annotations
---------------------

YOLO:
- One .txt file per image (same filename)
- Format: <class_id> <x_center> <y_center> <width> <height>
- Normalized coordinates (0-1)
- Example: "0 0.5 0.5 0.2 0.3"

COCO:
- Single JSON file for dataset
- Absolute pixel coordinates: [x_min, y_min, width, height]
- Structure:
  {
    "images": [{"id": int, "file_name": str, "width": int, "height": int}],
    "annotations": [{"image_id": int, "bbox": [x,y,w,h], "category_id": int}],
    "categories": [{"id": int, "name": str}]
  }

Pascal VOC:
- XML file per image
- Absolute pixels: [xmin, ymin, xmax, ymax]
- Example:
  <annotation>
      <object>
          <name>class_name</name>
          <bndbox>
              <xmin>100</xmin>
              <ymin>200</ymin>
              <xmax>150</xmax>
              <ymax>260</ymax>
          </bndbox>
      </object>
  </annotation>

Annotation BBox Formats Overview
--------------------------------

Format   Coordinates      Notation         Values      Example
-------- --------------- ---------------- ----------- ----------------------
YOLO     Normalized      (xc, yc, w, h)   0 to 1      0 0.5 0.5 0.2 0.3
COCO     Absolute        (x1, y1, w, h)   pixels      100 200 50 60
Pascal   Absolute        (x1, y1, x2, y2) pixels      100 200 150 260

Common Conversion Formulas
--------------------------

YOLO to COCO:
    x_min = (x_center - width/2) * img_width
    y_min = (y_center - height/2) * img_height
    w = width * img_width
    h = height * img_height

COCO to YOLO:
    x_center = (x_min + width/2) / img_width
    y_center = (y_min + height/2) / img_height
    width = width / img_width
    height = height / img_height

Pascal VOC to YOLO:
    x_center = (xmin + xmax)/(2.0 * img_width)
    y_center = (ymin + ymax)/(2.0 * img_height)
    width = (xmax - xmin)/img_width
    height = (ymax - ymin)/img_height
"""

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Union
from enum import auto, IntEnum


class BBoxFormat(IntEnum):
    CORNERS = auto()
    CENTER = auto()
    TOPLEFT = auto()


class NotationFormat(IntEnum):
    YOLO = auto()
    COCO = auto()
    PASCALVOC = auto()


def yolo_to_coco_bbox(yolo_bbox: Union[List, Tuple], img_width: int, img_height: int) -> list[float]:
    """Convert YOLO bbox format to COCO format.

    YOLO: (x_center, y_center, width, height) normalized [0-1]
    COCO: (x_min, y_min, width, height) in pixels
    """
    x_center, y_center, width, height = yolo_bbox

    x_min = int((x_center - width/2) * img_width)
    y_min = int((y_center - height/2) * img_height)
    w = width * img_width
    h = height * img_height

    return [x_min, y_min, w, h]


def coco_to_yolo_bbox(coco_bbox: Union[List, Tuple], img_width: int, img_height: int) -> list[float]:
    """Convert COCO bbox format to YOLO format.

    COCO: (x_min, y_min, width, height) in pixels
    YOLO: (x_center, y_center, width, height) normalized [0-1]
    """
    x_min, y_min, w, h = coco_bbox

    x_center = (x_min + w/2) / img_width
    y_center = (y_min + h/2) / img_height
    width = w / img_width
    height = h / img_height

    return [x_center, y_center, width, height]


def pascalvoc_to_yolo_bbox(voc_bbox: Union[List, Tuple], img_width: int, img_height: int) -> list[float]:
    """Convert Pascal VOC bbox format to YOLO format.

    Pascal VOC: (xmin, ymin, xmax, ymax) in pixels
    YOLO: (x_center, y_center, width, height) normalized [0-1]
    """
    xmin, ymin, xmax, ymax = voc_bbox

    x_center = (xmin + xmax)/(2.0 * img_width)
    y_center = (ymin + ymax)/(2.0 * img_height)
    width = (xmax - xmin)/img_width
    height = (ymax - ymin)/img_height

    return [x_center, y_center, width, height]


def corners_to_center_format(box: Union[List, Tuple]) -> Tuple[float, float, float, float]:
    """
    Convert box from (x1, y1, x2, y2) format to (xc, yc, w, h) format.

    Args:
        box: List or tuple containing [x1, y1, x2, y2] coordinates

    Returns:
        Tuple containing (xc, yc, w, h) coordinates

    Raises:
        ValueError: If input box doesn't contain exactly 4 values
    """
    if len(box) != 4:
        raise ValueError("Box must contain exactly 4 values: [x1, y1, x2, y2]")

    x1, y1, x2, y2 = box
    xc = (x1 + x2) / 2
    yc = (y1 + y2) / 2
    w = abs(x2 - x1)  # Using abs to handle reversed coordinates
    h = abs(y2 - y1)
    return xc, yc, w, h


def corners_to_topleft_format(box: Union[List, Tuple]) -> Tuple[float, float, float, float]:
    """
    Convert box from (x1, y1, x2, y2) format to top-left format (x1, y1, w, h).

    Args:
        box: List or tuple containing [x1, y1, x2, y2] coordinates

    Returns:
        Tuple containing (x1, y1, w, h) coordinates in top-left format

    Raises:
        ValueError: If input box doesn't contain exactly 4 values
    """
    if len(box) != 4:
        raise ValueError("Box must contain exactly 4 values: [x1, y1, x2, y2]")

    x1, y1, x2, y2 = box
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    return x1, y1, w, h


def center_to_corners_format(box: Union[List, Tuple]) -> Tuple[float, float, float, float]:
    """
    Convert box from (xc, yc, w, h) format to (x1, y1, x2, y2) format.

    Args:
        box: List or tuple containing [xc, yc, w, h] coordinates

    Returns:
        Tuple containing (x1, y1, x2, y2) coordinates

    Raises:
        ValueError: If input box doesn't contain exactly 4 values or if w/h are negative
    """
    if len(box) != 4:
        raise ValueError("Box must contain exactly 4 values: [xc, yc, w, h]")

    xc, yc, w, h = box
    if w < 0 or h < 0:
        raise ValueError("Width and height must be non-negative")

    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return x1, y1, x2, y2


def center_to_topleft_format(box: Union[List, Tuple]) -> Tuple[float, float, float, float]:
    """
    Convert box from (xc, yc, w, h) format to top-left format (x1, y1, w, h).

    Args:
        box: List or tuple containing [xc, yc, w, h] coordinates

    Returns:
        Tuple containing (x1, y1, w, h) coordinates in top-left format
        
    Raises:
        ValueError: If input box doesn't contain exactly 4 values or if w/h are negative
    """
    if len(box) != 4:
        raise ValueError("Box must contain exactly 4 values: [xc, yc, w, h]")
    
    xc, yc, w, h = box
    if w < 0 or h < 0:
        raise ValueError("Width and height must be non-negative")
    
    x1 = xc - w / 2
    y1 = yc - h / 2
    return x1, y1, w, h


def topleft_to_center_format(box: Union[List, Tuple]) -> Tuple[float, float, float, float]:
    """
    Convert box from (x1, y1, w, h) format to center format (xc, yc, w, h).

    Args:
        box: List or tuple containing [x1, y1, w, h] coordinates

    Returns:
        Tuple containing (xc, yc, w, h) coordinates in center format

    Raises:
        ValueError: If input box doesn't contain exactly 4 values or if w/h are negative
    """
    if len(box) != 4:
        raise ValueError("Box must contain exactly 4 values: [x1, y1 w, h]")
    
    x1, y1, w, h = box
    if w < 0 or h < 0:
        raise ValueError("Width and height must be non-negative")
    
    xc = x1 + w / 2
    yc = y1 + h / 2
    return xc, yc, w, h


def topleft_to_corners_format(box: Union[List, Tuple]) -> Tuple[float, float, float, float]:
    """
    Convert box from (x1, y1, w, h) format to corners (x1, y1, x2, y2) format.

    Args:
        box: List or tuple containing [x1, y1, w, h] coordinates

    Returns:
        Tuple containing (x1, y1, x2, y2) coordinates.

    Raises:
        ValueError: If input box doesn't contain exactly 4 values or if w/h are negative
    """
    if len(box) != 4:
        raise ValueError("Box must contain exactly 4 values: [x1, y1, w, h]")
    
    x1, y1, w, h = box
    if w < 0 or h < 0:
        raise ValueError("Width and height must be non-negative")
    
    x2 = x1 + w
    y2 = y1 + h
    return x1, y1, x2, y2


def is_normalized(box: Union[List, Tuple]) -> bool:
    """
    Check if a bounding box is normalized (all values between 0 and 1).

    Args:
        box: Box coordinates to check

    Returns:
        bool: True if all coordinates are between 0 and 1, False otherwise
    """
    return all(0 <= x <= 1 for x in box)


def normalize_bbox(bbox: Union[List, Tuple], img_width: int, img_height: int, format: BBoxFormat = BBoxFormat.CORNERS) -> Tuple[float, float, float, float]:
    """
    Normalize bounding box coordinates relative to image dimensions.

    Args:
        bbox: Bounding box coordinates in one of three formats:
            - corners: (x1, y1, x2, y2) where (x1,y1) is top-left and (x2,y2) is bottom-right
            - topleft: (x1, y1, width, height) where (x1,y1) is top-left corner
            - center: (xc, yc, width, height) where (xc,yc) is center point
        img_width: Width of the image in pixels
        img_height: Height of the image in pixels
        format: Format of input bbox ('corners', 'topleft', or 'center')

    Returns:
        Tuple of normalized coordinates in same format as input
        All values are in range [0.0, 1.0]

    Raises:
        ValueError: If coordinates are invalid or exceed image dimensions
        TypeError: If input types are incorrect
    """
    # Input validation
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise TypeError("bbox must be a list or tuple of 4 elements")
    if not isinstance(img_width, (int, np.integer)) or not isinstance(img_height, (int, np.integer)):
        raise TypeError("Image dimensions must be integers")
    if img_width <= 0 or img_height <= 0:
        raise ValueError("Image dimensions must be positive")
    print(type(format))
    if not isinstance(format, BBoxFormat):
        raise TypeError("Format must be a  BBoxFormat")

    # Convert to float for calculations
    bbox = [float(x) for x in bbox]

    if format == BBoxFormat.CORNERS:
        x1, y1, x2, y2 = bbox

        # Validate coordinates
        if x1 > x2 or y1 > y2:
            raise ValueError(
                "Invalid box coordinates: x1/y1 must be less than x2/y2")
        if any(c < 0 for c in [x1, y1, x2, y2]):
            raise ValueError("Coordinates must be non-negative")
        if x2 > img_width or y2 > img_height:
            raise ValueError("Coordinates exceed image dimensions")

        # Normalize
        return (
            x1 / img_width,
            y1 / img_height,
            x2 / img_width,
            y2 / img_height
        )

    elif format == BBoxFormat.TOPLEFT:
        x1, y1, w, h = bbox

        # Validate coordinates and dimensions
        if w <= 0 or h <= 0:
            raise ValueError("Width and height must be positive")
        if x1 < 0 or y1 < 0:
            raise ValueError("Coordinates must be non-negative")
        if x1 + w > img_width or y1 + h > img_height:
            raise ValueError("Box exceeds image dimensions")

        # Normalize
        return (
            x1 / img_width,
            y1 / img_height,
            w / img_width,
            h / img_height
        )

    elif format == BBoxFormat.CENTER:
        xc, yc, w, h = bbox

        # Validate coordinates and dimensions
        if w <= 0 or h <= 0:
            raise ValueError("Width and height must be positive")

        # Calculate corners from center
        x1 = xc - w/2
        y1 = yc - h/2
        x2 = xc + w/2
        y2 = yc + h/2

        # Validate box is within image
        if x1 < 0 or y1 < 0:
            raise ValueError(
                "Box extends outside image (negative coordinates)")
        if x2 > img_width or y2 > img_height:
            raise ValueError("Box extends outside image dimensions")

        # Normalize
        return (
            xc / img_width,
            yc / img_height,
            w / img_width,
            h / img_height
        )


def absolute_bbox(bbox: Union[List, Tuple], img_width: int, img_height: int, format: BBoxFormat = BBoxFormat.CORNERS) -> Tuple[float, float, float, float]:
    """
    Convert normalized bounding box coordinates to absolute pixel coordinates.

    Args:
        bbox: Normalized coordinates [0.0-1.0] in specified format
        img_width: Image width in pixels
        img_height: Image height in pixels
        format: Box format ('corners', 'topleft', or 'center')

    Returns:
        Tuple of absolute coordinates in same format as input
    """
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise TypeError("bbox must be a list/tuple of 4 elements")
    if not isinstance(img_width, (int, np.integer)) or not isinstance(img_height, (int, np.integer)):
        raise TypeError("Image dimensions must be integers")
    if img_width <= 0 or img_height <= 0:
        raise ValueError("Image dimensions must be positive")
    if not all(0 <= x <= 1 for x in bbox):
        raise ValueError("Normalized coordinates must be in range [0,1]")
    if not isinstance(format, BBoxFormat):
        raise TypeError("Format must be a  BBoxFormat")

    bbox = [float(x) for x in bbox]

    if format == BBoxFormat.CORNERS:
        x1, y1, x2, y2 = bbox
        if x1 > x2 or y1 > y2:
            raise ValueError("Invalid box: x1/y1 must be less than x2/y2")
        return tuple(map(int, (
            x1 * img_width,
            y1 * img_height,
            x2 * img_width,
            y2 * img_height
        )))

    elif format == BBoxFormat.TOPLEFT:
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            raise ValueError("Width and height must be positive")
        return tuple(map(int, (
            x * img_width,
            y * img_height,
            w * img_width,
            h * img_height
        )))

    elif format == BBoxFormat.CENTER:
        xc, yc, w, h = bbox
        if w <= 0 or h <= 0:
            raise ValueError("Width and height must be positive")
        if (xc - w/2) < 0 or (yc - h/2) < 0 or (xc + w/2) > 1 or (yc + h/2) > 1:
            raise ValueError("Box extends outside normalized bounds")
        return tuple(map(int, (
            xc * img_width,
            yc * img_height,
            w * img_width,
            h * img_height
        )))

    raise ValueError(f"Unsupported format: {format}. Must be 'corners', 'topleft', or 'center'")


def calculate_iou(box1: Union[List, Tuple], box2: Union[List, Tuple], format: BBoxFormat = BBoxFormat.CORNERS) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1: First bounding box coordinates
        box2: Second bounding box coordinates
        format: Format of the input boxes, either "corners" (x1,y1,x2,y2), "center" (xc,yc,w,h) or "topleft" (x1,y1,w,h)

    Returns:
        float: IoU value between 0 and 1

    Raises:
        ValueError: If format is not "corners", "center" or "topleft"
    """
    if not isinstance(format, BBoxFormat):
        raise TypeError("Format must be a  BBoxFormat")

    # Convert to corners format if necessary
    if format == BBoxFormat.CENTER:
        box1 = center_to_corners_format(box1)
        box2 = center_to_corners_format(box2)
    elif format == BBoxFormat.TOPLEFT:
        box1 = topleft_to_corners_format(box1)
        box2 = topleft_to_corners_format(box2)

    # Calculate intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calculate areas
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    # Avoid division by zero
    return intersection / (union + np.finfo(float).eps)


def plot_image_with_boxes(image_file: str, label_file: str):
    """Plot a image with its bounding boxes

    Args:
        image_file (str): path to the image
        label_file (str): path to the label
    """
    from saltup.ai.object_detection.dataset.yolo_darknet import read_label

    # Read image
    image = cv2.imread(image_file)
    # Read image dimensions
    image_height, image_width, _ = image.shape
    # Read YOLO label file
    boxes = read_label(label_file)
    # Draw bounding boxes
    for box in boxes:
        xc, yc, w, h, class_id = box
        x1, y1, x2, y2 = center_to_corners_format((xc, yc, w, h))

        x1 *= image_width
        y1 *= image_height
        x2 *= image_width
        y2 *= image_height

        # Define color based on class ID
        color = (0, 255, 255)  # Green color for bounding boxes
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    # Display image with bounding boxes using matplotlib
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

import json
import xml.etree.ElementTree as ET


# # examples:
# yolo_bboxes = BBox.from_yolo_file("path/to/yolo_annotation.txt", img_width=640, img_height=480)
# for bbox in yolo_bboxes:
#     print(bbox.to_coco())  # Converti in formato COCO
# # Carica bounding box da un file COCO
# coco_bboxes = BBox.from_coco_file("path/to/coco_annotation.json", image_id=42)
# for bbox in coco_bboxes:
#     print(bbox.to_pascal_voc())  # Converti in formato Pascal VOC

import json
import xml.etree.ElementTree as ET

class BBox:
    def __init__(self, coordinates: Union[List, Tuple] = None, format: BBoxFormat = BBoxFormat.CORNERS, img_width: int = None, img_height: int = None):
        """
        Initialize a BBox object with coordinates and format.

        Args:
            coordinates: The bounding box coordinates (optional).
            format: The format of the coordinates (CORNERS, CENTER, TOPLEFT).
            img_width: The width of the image (required for normalization).
            img_height: The height of the image (required for normalization).
        """
        self.coordinates = coordinates
        self.format = format
        self.img_width = img_width
        self.img_height = img_height

    @classmethod
    def from_yolo_file(cls, file_path: str, img_width: int, img_height: int):
        """
        Load bounding box from a YOLO format annotation file.

        Args:
            file_path: Path to the YOLO annotation file.
            img_width: Width of the image.
            img_height: Height of the image.

        Returns:
            A list of BBox objects (one for each annotation in the file).
        """
        bboxes = []
        with open(file_path, 'r') as file:
            for line in file:
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
                bbox = cls([x_center, y_center, width, height], format=BBoxFormat.CENTER, img_width=img_width, img_height=img_height)
                bboxes.append(bbox)
        return bboxes

    @classmethod
    def from_coco_file(cls, file_path: str, image_id: int):
        """
        Load bounding box from a COCO format annotation file.

        Args:
            file_path: Path to the COCO annotation file.
            image_id: ID of the image to load annotations for.

        Returns:
            A list of BBox objects (one for each annotation in the file).
        """
        with open(file_path, 'r') as file:
            data = json.load(file)

        bboxes = []
        for annotation in data['annotations']:
            if annotation['image_id'] == image_id:
                x_min, y_min, width, height = annotation['bbox']
                bbox = cls([x_min, y_min, width, height], format=BBoxFormat.TOPLEFT)
                bboxes.append(bbox)
        return bboxes

    @classmethod
    def from_pascal_voc_file(cls, file_path: str):
        """
        Load bounding box from a Pascal VOC format annotation file.

        Args:
            file_path: Path to the Pascal VOC annotation file.

        Returns:
            A list of BBox objects (one for each annotation in the file).
        """
        tree = ET.parse(file_path)
        root = tree.getroot()

        bboxes = []
        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            bbox = cls([xmin, ymin, xmax, ymax], format=BBoxFormat.CORNERS)
            bboxes.append(bbox)
        return bboxes

    def get_coordinates(self, format: BBoxFormat = None) -> Tuple[float, float, float, float]:
        """
        Get the bounding box coordinates in the specified format.

        Args:
            format: The desired format (CORNERS, CENTER, TOPLEFT). If None, returns in the current format.

        Returns:
            Tuple of coordinates in the specified format.
        """
        if format is None:
            return self.coordinates

        if self.format == format:
            return self.coordinates

        if self.format == BBoxFormat.CORNERS:
            if format == BBoxFormat.CENTER:
                return corners_to_center_format(self.coordinates)
            elif format == BBoxFormat.TOPLEFT:
                return corners_to_topleft_format(self.coordinates)
        elif self.format == BBoxFormat.CENTER:
            if format == BBoxFormat.CORNERS:
                return center_to_corners_format(self.coordinates)
            elif format == BBoxFormat.TOPLEFT:
                return center_to_topleft_format(self.coordinates)
        elif self.format == BBoxFormat.TOPLEFT:
            if format == BBoxFormat.CORNERS:
                return topleft_to_corners_format(self.coordinates)
            elif format == BBoxFormat.CENTER:
                return topleft_to_center_format(self.coordinates)

        raise ValueError(f"Unsupported format conversion: {self.format} to {format}")

    def set_coordinates(self, coordinates: Union[List, Tuple], format: BBoxFormat = None):
        """
        Set the bounding box coordinates.

        Args:
            coordinates: The new bounding box coordinates.
            format: The format of the new coordinates (CORNERS, CENTER, TOPLEFT). If None, assumes current format.
        """
        if format is None:
            format = self.format

        if format == BBoxFormat.CORNERS:
            x1, y1, x2, y2 = coordinates
            if x1 > x2 or y1 > y2:
                raise ValueError("Invalid box coordinates: x1/y1 must be less than x2/y2")
        elif format == BBoxFormat.CENTER:
            xc, yc, w, h = coordinates
            if w < 0 or h < 0:
                raise ValueError("Width and height must be non-negative")
        elif format == BBoxFormat.TOPLEFT:
            x1, y1, w, h = coordinates
            if w < 0 or h < 0:
                raise ValueError("Width and height must be non-negative")

        self.coordinates = coordinates
        self.format = format

    def normalize(self, img_width: int, img_height: int):
        """
        Normalize the bounding box coordinates relative to the image dimensions.

        Args:
            img_width: The width of the image.
            img_height: The height of the image.
        """
        self.coordinates = normalize_bbox(self.coordinates, img_width, img_height, self.format)
        self.img_width = img_width
        self.img_height = img_height

    def absolute(self):
        """
        Convert normalized bounding box coordinates to absolute pixel coordinates.

        Returns:
            Tuple of absolute coordinates in the current format.
        """
        if self.img_width is None or self.img_height is None:
            raise ValueError("Image dimensions must be set for absolute conversion")
        return absolute_bbox(self.coordinates, self.img_width, self.img_height, self.format)

    def to_yolo(self) -> Tuple[float, float, float, float]:
        """
        Convert the bounding box to YOLO format (x_center, y_center, width, height) normalized.

        Returns:
            Tuple of YOLO format coordinates.
        """
        if self.format == BBoxFormat.CORNERS:
            return pascalvoc_to_yolo_bbox(self.coordinates, self.img_width, self.img_height)
        elif self.format == BBoxFormat.CENTER:
            return self.coordinates
        elif self.format == BBoxFormat.TOPLEFT:
            return topleft_to_center_format(self.coordinates)

    def to_coco(self) -> Tuple[float, float, float, float]:
        """
        Convert the bounding box to COCO format (x_min, y_min, width, height) in pixels.

        Returns:
            Tuple of COCO format coordinates.
        """
        if self.format == BBoxFormat.CORNERS:
            return self.coordinates
        elif self.format == BBoxFormat.CENTER:
            return center_to_corners_format(self.coordinates)
        elif self.format == BBoxFormat.TOPLEFT:
            return topleft_to_corners_format(self.coordinates)

    def to_pascal_voc(self) -> Tuple[float, float, float, float]:
        """
        Convert the bounding box to Pascal VOC format (x_min, y_min, x_max, y_max) in pixels.

        Returns:
            Tuple of Pascal VOC format coordinates.
        """
        if self.format == BBoxFormat.CORNERS:
            return self.coordinates
        elif self.format == BBoxFormat.CENTER:
            return center_to_corners_format(self.coordinates)
        elif self.format == BBoxFormat.TOPLEFT:
            return topleft_to_corners_format(self.coordinates)

    def calculate_iou(self, other: 'BBox') -> float:
        """
        Calculate Intersection over Union (IoU) with another bounding box.

        Args:
            other: Another BBox object to calculate IoU with.

        Returns:
            float: IoU value between 0 and 1.
        """
        return calculate_iou(self.get_coordinates(BBoxFormat.CORNERS), other.get_coordinates(BBoxFormat.CORNERS))

    def __repr__(self):
        return f"BBox(coordinates={self.coordinates}, format={self.format}, img_width={self.img_width}, img_height={self.img_height})"