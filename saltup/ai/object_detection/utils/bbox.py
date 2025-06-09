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

 Notation Name    Coordinate Fromat   Scale Fromat   Notation           Example          
---------------  ------------------- -------------- ------------------ ----------------- 
 YOLO             Center              Normalized     (xc, yc, w, h)     0.5 0.5 0.2 0.3  
 COCO             TopLeft             Absolute       (x1, y1, w, h)     100 200 50 60    
 Pascal VOC       Corners             Absolute       (x1, y1, x2, y2)   100 200 150 260  

Coordinate Fromats:
- Center:  (xc, yc, w, h)    center point + dimensions
- TopLeft: (x1, y1, w, h)    top-left point + dimensions
- Corners: (x1, y1, x2, y2)  top-left + bottom-right points

Scale Fromats:
- Normalized: all values in range [0,1]
- Absolute:   all values in pixels

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

import json
import xml.etree.ElementTree as ET
import cv2
import os
import numpy as np
from copy import deepcopy
from enum import auto, IntEnum
from typing import Iterator, List, Tuple, Dict, Optional, Union

from saltup.saltup_env import SaltupEnv
from saltup.utils.data.image.image_utils import ColorMode
from saltup.utils.data.image.image_utils import Image


class CoordinateFormat(IntEnum):
    CENTER = auto()
    TOPLEFT = auto()
    CORNERS = auto()

class ScaleFormat(IntEnum):
    NORMALIZED = auto()
    ABSOLUTE = auto()

class BBoxFormat(IntEnum):
    CORNERS_NORMALIZED = 1
    CENTER_NORMALIZED = 2
    TOPLEFT_NORMALIZED = 3

    CORNERS_ABSOLUTE = 4
    CENTER_ABSOLUTE = 5
    TOPLEFT_ABSOLUTE = 6

    # Common Notations
    YOLO = 2
    COCO = 6
    PASCALVOC = 4
    
    def to_coordinate_format(self):
        """Convert the BBoxFormat enum to a CoordinateFormat enum."""
        if self in [BBoxFormat.CENTER_NORMALIZED, BBoxFormat.CENTER_ABSOLUTE]:
            return CoordinateFormat.CENTER
        elif self in [BBoxFormat.TOPLEFT_NORMALIZED, BBoxFormat.TOPLEFT_ABSOLUTE]:
            return CoordinateFormat.TOPLEFT
        elif self in [BBoxFormat.CORNERS_NORMALIZED, BBoxFormat.CORNERS_ABSOLUTE]:
            return CoordinateFormat.CORNERS
        else:
            raise ValueError(f"Unknown BBoxFormat: {self}")
        
    def to_scale_format(self):
        """Convert the BBoxFormat enum to a ScaleFormat enum."""
        if self in [BBoxFormat.CORNERS_NORMALIZED, BBoxFormat.CENTER_NORMALIZED, BBoxFormat.TOPLEFT_NORMALIZED]:
            return ScaleFormat.NORMALIZED
        elif self in [BBoxFormat.CORNERS_ABSOLUTE, BBoxFormat.CENTER_ABSOLUTE, BBoxFormat.TOPLEFT_ABSOLUTE]:
            return ScaleFormat.ABSOLUTE
        else:
            raise ValueError(f"Unknown BBoxFormat: {self}")

    def to_string(self):
        """Convert the BBoxFormat enum to a human-readable string."""
        if self == BBoxFormat.CORNERS_NORMALIZED:
            return "Corners Normalized (x1, y1, x2, y2)"
        elif self == BBoxFormat.CENTER_NORMALIZED:
            return "Center Normalized (center_x, center_y, width, height)"
        elif self == BBoxFormat.TOPLEFT_NORMALIZED:
            return "Top-left Normalized (x, y, width, height)"
        elif self == BBoxFormat.CORNERS_ABSOLUTE:
            return "Corners Absolute (x1, y1, x2, y2)"
        elif self == BBoxFormat.CENTER_ABSOLUTE:
            return "Center Absolute (center_x, center_y, width, height)"
        elif self == BBoxFormat.TOPLEFT_ABSOLUTE:
            return "Top-left Absolute (x, y, width, height)"
        elif self == BBoxFormat.YOLO:
            return "YOLO (normalized center-x, center-y, width, height)"
        elif self == BBoxFormat.COCO:
            return "COCO (absolute x1, y1, width, height)"
        elif self == BBoxFormat.PASCALVOC:
            return "PASCAL VOC (absolute x1, y1, x2, y2)"
        else:
            raise ValueError(f"Unknown BBoxFormat: {self}")
        
BBOX_INNER_FORMAT = BBoxFormat(SaltupEnv.SALTUP_BBOX_INNER_FORMAT)
FLOAT_PRECISION = SaltupEnv.SALTUP_BBOX_FLOAT_PRECISION

class IoUType(IntEnum):
    IOU = auto()
    DIOU = auto()
    CIOU = auto()
    GIOU = auto()


def convert_matrix_boxes(box_xy, box_wh):
    """
    Convert bounding boxes from center format to corner format.

    Args:
        box_xy (numpy.ndarray): Array containing the center coordinates of the boxes (x_center, y_center).
        box_wh (numpy.ndarray): Array containing the width and height of the boxes (width, height).

    Returns:
        corners (numpy.ndarray): Array containing the corner coordinates of the boxes in (xmin, ymin, xmax, ymax) format.
        centers (numpy.ndarray): Array containing the center coordinates and width and height of the boxes in (x, y, w, h) format.
    """
    # Calculate box corners
    box_mins = box_xy - (box_wh / 2.0)  # (xmin, ymin)
    box_maxes = box_xy + (box_wh / 2.0)  # (xmax, ymax)

    # Concatenate to get corners in (xmin, ymin, xmax, ymax) format
    corners = np.concatenate([
        box_mins[..., 0:1],  # x_min
        box_mins[..., 1:2],  # y_min
        box_maxes[..., 0:1],  # x_max
        box_maxes[..., 1:2]  # y_max
    ], axis=-1)

    # Concatenate to get centers in (x, y, w, h) format
    centers = np.concatenate([
        box_xy[..., 0:1],  # x
        box_xy[..., 1:2],  # y
        box_wh[..., 0:1],  # w
        box_wh[..., 1:2]   # h
    ], axis=-1)

    return corners, centers


class BBox:
    def __init__(
        self,
        coordinates: Union[List, Tuple, np.ndarray] = None,
        fmt: BBoxFormat = BBoxFormat.YOLO,
        img_height: int = None,
        img_width: int = None
    ):
        """
        Initialize a BBox object with coordinates and format.

        Args:
            coordinates: The bounding box coordinates.
            fmt: Input coordinates format (BBoxFormat enum). Defaults to BBoxFormat.YOLO.
            img_width: The width of the image (required for normalization).
            img_height: The height of the image (required for normalization).

        Raises:
            ValueError: If the coordinates are invalid or not in the correct format.

        Note:
            Internally, the BBox class always stores coordinates in CORNERS_ABSOLUTE format.
        """
        if coordinates is None:
            self.__coordinates = []
        else:
            self.set_coordinates(
                coordinates=coordinates,
                fmt=fmt,
                img_height=img_height,
                img_width=img_width
            )

    def copy(self):
        """Create a deep copy of the BoundingBox object."""
        return deepcopy(self)

    @classmethod
    def is_normalized(cls, coordinates: Union[List, Tuple, 'BBox'], eps: float = 5e-3) -> bool:
        """
        Check if the bounding box coordinates are normalized.
        Args:
            coordinates: The bounding box coordinates or a BBox object.
            eps: Tolerance for floating point comparison (default: 5e-3).

        Returns:
            bool: True if the coordinates are normalized, False otherwise.
        """
        if isinstance(coordinates, BBox):
            coordinates = coordinates.get_coordinates()
        
        # For normalized formats, check if all coordinates are between 0 and 1 (with tolerance)
        return all(-eps <= x <= 1.0 + eps for x in coordinates)
    
    @classmethod
    def clamp_normalized_coordinates(cls, coordinates: Union[List, Tuple]) -> List[float]:
        """
        Clamp normalized coordinates to ensure they are within [0, 1].

        Args:
            coordinates: List or tuple of normalized coordinates.

        Returns:
            List of clipped coordinates.
        """
        if not isinstance(coordinates, (list, tuple)):
            raise TypeError("Coordinates must be a list or tuple.")
        if not all(isinstance(x, (int, float)) for x in coordinates):
            raise ValueError("All coordinates must be numeric.")
        return [min(max(x, 0.0), 1.0) for x in coordinates]

    @classmethod
    def corners_to_center_format(cls, box: Union[List, Tuple]) -> Tuple[float, float, float, float]:
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
        
        if cls.is_normalized([xc, yc, w, h]):
            # Clip normalized coordinates to [0, 1]
            xc, yc, w, h = cls.clamp_normalized_coordinates([xc, yc, w, h])
        
        return xc, yc, w, h

    @classmethod
    def corners_to_topleft_format(cls, box: Union[List, Tuple]) -> Tuple[float, float, float, float]:
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
        
        if cls.is_normalized([x1, y1, w, h]):
            # Clip normalized coordinates to [0, 1]
            x1, y1, w, h = cls.clamp_normalized_coordinates([x1, y1, w, h])
        
        return x1, y1, w, h

    @classmethod
    def center_to_corners_format(cls, box: Union[List, Tuple]) -> Tuple[float, float, float, float]:
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

        # Clamp coordinates
        x1 = max(0, xc - w / 2)
        y1 = max(0, yc - h / 2)
        x2 = max(0, xc + w / 2)
        y2 = max(0, yc + h / 2)
        
        if cls.is_normalized([x1, y1, x2, y2]):
            # Clip normalized coordinates to [0, 1]
            x1, y1, x2, y2 = cls.clamp_normalized_coordinates([x1, y1, x2, y2])
        
        return x1, y1, x2, y2

    @classmethod
    def center_to_topleft_format(cls, box: Union[List, Tuple]) -> Tuple[float, float, float, float]:
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
        
        if cls.is_normalized([x1, y1, w, h]):
            # Clip normalized coordinates to [0, 1]
            x1, y1, w, h = cls.clamp_normalized_coordinates([x1, y1, w, h])

        return x1, y1, w, h

    @classmethod
    def topleft_to_center_format(cls, box: Union[List, Tuple]) -> Tuple[float, float, float, float]:
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
        
        if cls.is_normalized([xc, yc, w, h]):
            # Clip normalized coordinates to [0, 1]
            xc, yc, w, h = cls.clamp_normalized_coordinates([xc, yc, w, h])
        
        return xc, yc, w, h

    @classmethod
    def topleft_to_corners_format(cls, box: Union[List, Tuple]) -> Tuple[float, float, float, float]:
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
        
        if cls.is_normalized([x1, y1, x2, y2]):
            # Clip normalized coordinates to [0, 1]
            x1, y1, x2, y2 = cls.clamp_normalized_coordinates([x1, y1, x2, y2])
        
        return x1, y1, x2, y2

    @classmethod
    def normalize(
        cls,
        bbox: Union[List, Tuple],
        img_width: int,
        img_height: int,
        fmt: BBoxFormat
    ) -> Tuple[float, float, float, float]:
        """
        Normalize bounding box coordinates relative to image dimensions.

        Args:
            bbox: Bounding box coordinates in one of the absolute formats
            img_width: Width of the image in pixels
            img_height: Height of the image in pixels
            fmt: Format of input bbox (one of the absolute formats)

        Returns:
            Tuple of normalized coordinates in corresponding normalized format
            All values are in range [0.0, 1.0]

        Raises:
            ValueError: If coordinates are invalid (e.g., negative width/height)
            TypeError: If input types are incorrect
        """
        # Input validation
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise TypeError(f"bbox must be a list or tuple of 4 elements. Passed {type(bbox)}.")
        if not isinstance(img_width, (int, np.integer)) or not isinstance(img_height, (int, np.integer)):
            raise TypeError(f"Image dimensions (width, height) must be integers. Passed {type(img_width)}, {type(img_height)}.")
        if img_width <= 0 or img_height <= 0:
            raise ValueError("Image dimensions must be positive")
        if not isinstance(fmt, BBoxFormat):
            raise TypeError("Format must be a BBoxFormat")

        # Convert to float for calculations
        bbox = [float(x) for x in bbox]

        # Check if format is already normalized
        if fmt in [BBoxFormat.CORNERS_NORMALIZED, BBoxFormat.CENTER_NORMALIZED,
                     BBoxFormat.TOPLEFT_NORMALIZED, BBoxFormat.YOLO]:
            return bbox

        if fmt == BBoxFormat.CORNERS_ABSOLUTE or fmt == BBoxFormat.PASCALVOC:
            x1, y1, x2, y2 = bbox

            # Validate coordinates
            if x1 > x2 or y1 > y2:
                raise ValueError("Invalid box coordinates: x1/y1 must be less than x2/y2")
            if any(c < 0 for c in [x1, y1, x2, y2]):
                raise ValueError("Coordinates must be non-negative")

            # Clip coordinates to image boundaries
            x1 = max(0, min(x1, img_width))
            y1 = max(0, min(y1, img_height))
            x2 = max(0, min(x2, img_width))
            y2 = max(0, min(y2, img_height))

            # Normalize and clamp to [0, 1]
            x1_norm, y1_norm, x2_norm, y2_norm = cls.clamp_normalized_coordinates([
                x1 / img_width,
                y1 / img_height,
                x2 / img_width,
                y2 / img_height
            ])

            return (
                round(x1_norm, FLOAT_PRECISION),
                round(y1_norm, FLOAT_PRECISION),
                round(x2_norm, FLOAT_PRECISION),
                round(y2_norm, FLOAT_PRECISION)
            )

        elif fmt == BBoxFormat.TOPLEFT_ABSOLUTE or fmt == BBoxFormat.COCO:
            x1, y1, w, h = bbox

            # Validate coordinates and dimensions
            if w <= 0 or h <= 0:
                raise ValueError("Width and height must be positive")
            if x1 < 0 or y1 < 0:
                raise ValueError("Coordinates must be non-negative")

            # Clip coordinates to image boundaries
            x1 = max(0, min(x1, img_width))
            y1 = max(0, min(y1, img_height))
            w = min(w, img_width - x1)
            h = min(h, img_height - y1)

            # Normalize and clamp to [0, 1]
            x1_norm, y1_norm, w_norm, h_norm = cls.clamp_normalized_coordinates([
                x1 / img_width,
                y1 / img_height,
                w / img_width,
                h / img_height
            ])

            return (
                round(x1_norm, FLOAT_PRECISION),
                round(y1_norm, FLOAT_PRECISION),
                round(w_norm, FLOAT_PRECISION),
                round(h_norm, FLOAT_PRECISION)
            )

        elif fmt == BBoxFormat.CENTER_ABSOLUTE:
            xc, yc, w, h = bbox

            # Validate coordinates and dimensions
            if w <= 0 or h <= 0:
                raise ValueError("Width and height must be positive")

            # Calculate the corners of the bounding box
            x1 = xc - w / 2
            y1 = yc - h / 2
            x2 = xc + w / 2
            y2 = yc + h / 2

            # Clip coordinates to image boundaries
            x1 = max(0, min(x1, img_width))
            y1 = max(0, min(y1, img_height))
            x2 = max(0, min(x2, img_width))
            y2 = max(0, min(y2, img_height))

            # Recalculate center coordinates and dimensions after clipping
            xc = (x1 + x2) / 2
            yc = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1

            # Normalize and clamp to [0, 1]
            xc_norm, yc_norm, w_norm, h_norm = cls.clamp_normalized_coordinates([
                xc / img_width,
                yc / img_height,
                w / img_width,
                h / img_height
            ])

            return (
                round(xc_norm, FLOAT_PRECISION),
                round(yc_norm, FLOAT_PRECISION),
                round(w_norm, FLOAT_PRECISION),
                round(h_norm, FLOAT_PRECISION)
            )

        raise ValueError(f"Unsupported format: {fmt}")

    @classmethod
    def absolute(
        cls,
        bbox: Union[List, Tuple],
        img_width: int,
        img_height: int,
        fmt: BBoxFormat
    ) -> Tuple[float, float, float, float]:
        """
        Convert normalized bounding box coordinates to absolute pixel coordinates.

        Args:
            bbox: Normalized coordinates [0.0-1.0] in specified format
            img_width: Image width in pixels
            img_height: Image height in pixels
            fmt: Box format (one of the normalized formats)

        Returns:
            Tuple of absolute coordinates in corresponding absolute format
        """
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise TypeError("bbox must be a list/tuple of 4 elements")
        if not isinstance(img_width, (int, np.integer)) or not isinstance(img_height, (int, np.integer)):
            raise TypeError("Image dimensions must be integers")
        if img_width <= 0 or img_height <= 0:
            raise ValueError("Image dimensions must be positive")
        if not all(0 <= x <= 1 for x in bbox):
            raise ValueError("Normalized coordinates must be in range [0,1]. Passed:\n" + str(bbox))
        if not isinstance(fmt, BBoxFormat):
            raise TypeError("Format must be a BBoxFormat")

        # Check if format is already absolute
        if fmt in [BBoxFormat.CORNERS_ABSOLUTE, BBoxFormat.CENTER_ABSOLUTE,
                     BBoxFormat.TOPLEFT_ABSOLUTE, BBoxFormat.COCO, BBoxFormat.PASCALVOC]:
            return bbox

        bbox = [float(x) for x in bbox]

        if fmt == BBoxFormat.CORNERS_NORMALIZED:
            x1, y1, x2, y2 = bbox
            if x1 > x2 or y1 > y2:
                raise ValueError("Invalid box: x1/y1 must be less than x2/y2")
            return tuple(map(int, (
                x1 * img_width,
                y1 * img_height,
                x2 * img_width,
                y2 * img_height
            )))

        elif fmt == BBoxFormat.TOPLEFT_NORMALIZED:
            x, y, w, h = bbox
            if w <= 0 or h <= 0:
                raise ValueError("Width and height must be positive")
            return tuple(map(int, (
                x * img_width,
                y * img_height,
                w * img_width,
                h * img_height
            )))

        elif fmt == BBoxFormat.CENTER_NORMALIZED or fmt == BBoxFormat.YOLO:
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

        raise ValueError(f"Unsupported format: {fmt}")

    @staticmethod
    def _compute_iou(
        box1: Union[List, Tuple, 'BBox'],
        box2: Union[List, Tuple, 'BBox'],
        fmt: BBoxFormat = BBoxFormat.CORNERS_ABSOLUTE,
        img_shape: Tuple[int, int] = None,
        iou_type: IoUType = IoUType.IOU
    ) -> float:
        """
        Compute the Intersection over Union (IoU) between two bounding boxes.

        Args:
            box1: The first bounding box, either as a BBox object or a list/tuple of coordinates.
            box2: The second bounding box, either as a BBox object or a list/tuple of coordinates.
            fmt: The format of the bounding boxes. Defaults to BBoxFormat.CORNERS_ABSOLUTE.
            img_shape: The shape of the image as (height, width). Defaults to None.
            iou_type: The type of IoU to compute. Defaults to IoUType.IOU.

        Returns:
            float: The computed IoU value.

        Raises:
            TypeError: If the format is not a BBoxFormat or the IoU type is not an IoUType.
            TypeError: If the boxes are not BBox objects or lists/tuples of coordinates.
            ValueError: If an invalid IoU type is provided.
        """
        if not isinstance(fmt, BBoxFormat):
            raise TypeError("Format must be a BBoxFormat")
        if not isinstance(iou_type, IoUType):
            raise TypeError("IoU type must be an IoUType")

        if all(isinstance(b, BBox) for b in [box1, box2]):
            if box1.get_img_shape() != box2.get_img_shape():
                raise ValueError("Bounding boxes must have the same image shape")
            
            box1_coords = box1.get_coordinates(fmt=BBoxFormat.CORNERS_ABSOLUTE)
            box2_coords = box2.get_coordinates(fmt=BBoxFormat.CORNERS_ABSOLUTE)
            box1, box2 = box1_coords, box2_coords
        elif all(isinstance(b, (list, tuple)) for b in [box1, box2]):
            if fmt != BBoxFormat.CORNERS_ABSOLUTE:
                img_height, img_width = img_shape if img_shape else (None, None)   
                # Create BBox objects from the raw coordinates
                bbox1 = BBox(coordinates=box1, fmt=fmt, img_height=img_height, img_width=img_width)
                bbox2 = BBox(coordinates=box2, fmt=fmt, img_height=img_height, img_width=img_width)
                
                # Convert to absolute corners format
                box1 = bbox1.get_coordinates(fmt=BBoxFormat.CORNERS_ABSOLUTE)
                box2 = bbox2.get_coordinates(fmt=BBoxFormat.CORNERS_ABSOLUTE)
        else:
            raise TypeError(f"Boxes must be BBox objects or lists/tuples of coordinates. Passed: {type(box1)}, {type(box2)}")
                
        # Extract coordinates
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Calculate intersection coordinates
        x1 = max(x1_1, x1_2)
        y1 = max(y1_1, y1_2)
        x2 = min(x2_1, x2_2)
        y2 = min(y2_1, y2_2)

        # Calculate intersection area
        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        # Calculate areas of the bounding boxes
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        # Calculate IoU
        iou = intersection / (union + np.finfo(float).eps)

        if iou_type == IoUType.IOU:
            return iou

        # Calculate DIoU, CIoU, or GIoU
        elif iou_type in [IoUType.DIOU, IoUType.CIOU, IoUType.GIOU]:
            # Calculate the coordinates of the smallest enclosing box
            x1_c = min(x1_1, x1_2)
            y1_c = min(y1_1, y1_2)
            x2_c = max(x2_1, x2_2)
            y2_c = max(y2_1, y2_2)

            # Calculate the diagonal distance of the smallest enclosing box
            c = (x2_c - x1_c) ** 2 + (y2_c - y1_c) ** 2

            # Calculate the distance between the centers of the two boxes
            x1_m = (x1_1 + x2_1) / 2
            y1_m = (y1_1 + y2_1) / 2
            x2_m = (x1_2 + x2_2) / 2
            y2_m = (y1_2 + y2_2) / 2
            d = (x1_m - x2_m) ** 2 + (y1_m - y2_m) ** 2

            # Calculate DIoU
            diou = iou - d / (c + np.finfo(float).eps)

            if iou_type == IoUType.DIOU:
                return diou

            # Calculate CIoU
            elif iou_type == IoUType.CIOU:
                # Calculate aspect ratio consistency
                v = (4 / (np.pi ** 2)) * (np.arctan((x2_1 - x1_1) / (y2_1 - y1_1 + np.finfo(float).eps)) - np.arctan((x2_2 - x1_2) / (y2_2 - y1_2 + np.finfo(float).eps))) ** 2
                alpha = v / (1 - iou + v + np.finfo(float).eps)
                ciou = diou - alpha * v
                return ciou

            # Calculate GIoU
            elif iou_type == IoUType.GIOU:
                # Calculate the area of the smallest enclosing box
                area_c = (x2_c - x1_c) * (y2_c - y1_c)
                giou = iou - (area_c - union) / (area_c + np.finfo(float).eps)
                return giou

        else:
            raise ValueError("Invalid IoU type")

    @classmethod
    def from_yolo_file(cls, file_path: str, img_height: int = None, img_width: int = None) -> Tuple[List['BBox'], List[int]]:
        """
        Load bounding boxes from a YOLO format annotation file.

        Args:
            file_path: Path to the YOLO annotation file.
            img_width: Width of the image.
            img_height: Height of the image.

        Returns:
            Tuple[List[BBox], List[int]]: A tuple containing:
                - A list of BBox objects (one for each annotation in the file).
                - A list of class IDs (integers) corresponding to each bounding box.

        Raises:
            ValueError: If a line in the file does not contain exactly 5 values.
        """
        bboxes: List[BBox] = []
        class_ids: List[int] = []
        with open(file_path, 'r') as file:
            for line in file:
                # Split the line into components
                components = line.strip().split()
                if len(components) != 5:
                    raise ValueError(f"Invalid YOLO format: expected 5 values, got {len(components)} in line: {line}")

                # Parse the components
                class_id, x_center, y_center, width, height = map(float, components)
                class_id = int(class_id)

                # Create a BBox object in CENTER format
                bbox = cls(
                    coordinates=[x_center, y_center, width, height],
                    fmt=BBoxFormat.YOLO,
                    img_height=img_height,
                    img_width=img_width
                )
                bboxes.append(bbox)
                class_ids.append(class_id)

        return bboxes, class_ids

    @classmethod
    def from_coco_file(cls, file_path: str, image_id: int, img_height: int = None, img_width: int = None) -> List['BBox']:
        """
        Load bounding box from a COCO format annotation file.

        Args:
            file_path: Path to the COCO annotation file.
            image_id: ID of the image to load annotations for.
            img_width: Width of the image.
            img_height: Height of the image.

        Returns:
            A list of BBox objects (one for each annotation in the file).
        """
        with open(file_path, 'r') as file:
            data = json.load(file)

        bboxes = []
        for annotation in data['annotations']:
            if annotation['image_id'] == image_id:
                x_min, y_min, width, height = annotation['bbox']
                bbox = cls(
                    coordinates=[x_min, y_min, width, height],
                    fmt=BBoxFormat.COCO,
                    img_height=img_height,
                    img_width=img_width
                )
                bboxes.append(bbox)
        return bboxes

    @classmethod
    def from_pascal_voc_file(cls, file_path: str, img_height: int = None, img_width: int = None):
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
            bbox = cls(
                coordinates=[xmin, ymin, xmax, ymax],
                fmt=BBoxFormat.PASCALVOC,
                img_height=img_height,
                img_width=img_width
            )
            bboxes.append(bbox)
        return bboxes

    def get_img_shape(self) -> Tuple[int, int]:
        return self.img_height, self.img_width
    
    @staticmethod
    def converter(
        coordinates: Union[List, Tuple],
        from_fmt: BBoxFormat,
        to_fmt: BBoxFormat,
        img_shape: tuple = None
    ) -> Tuple[float, float, float, float]:
        """
        Converts bounding box coordinates from one format and scale to another.
        
        Args:
            coordinates (Union[List, Tuple]): The bounding box coordinates to be converted.
                The format of the coordinates depends on the `from_fmt` parameter.
            from_fmt (BBoxFormat): The current format of the bounding box coordinates.
            to_fmt (BBoxFormat): The desired format to convert the bounding box coordinates to.
            img_shape (tuple, optional): The shape of the image as (height, width). Required
                if scale conversion is needed (e.g., from absolute to normalized or vice versa).
        
        Returns:
            tuple: The converted bounding box coordinates in the desired format and scale.
        
        Raises:
            ValueError: If the target coordinate format is unsupported.
            ValueError: If image dimensions are required for scale conversion but not provided.
        """
        new_coordinates = list(coordinates)
        from_coord_type, from_scale_type = from_fmt.to_coordinate_format(), from_fmt.to_scale_format()
        to_coord_type, to_scale_type = to_fmt.to_coordinate_format(), to_fmt.to_scale_format()
        
        # Convert format
        if from_coord_type == CoordinateFormat.CORNERS:
            if to_coord_type == CoordinateFormat.CENTER:
                new_coordinates = BBox.corners_to_center_format(coordinates)
            elif to_coord_type == CoordinateFormat.TOPLEFT:
                new_coordinates = BBox.corners_to_topleft_format(coordinates)
        elif from_coord_type == CoordinateFormat.CENTER:
            if to_coord_type == CoordinateFormat.CORNERS:
                new_coordinates = BBox.center_to_corners_format(coordinates)
            elif to_coord_type == CoordinateFormat.TOPLEFT:
                new_coordinates = BBox.center_to_topleft_format(coordinates)
        elif from_coord_type == CoordinateFormat.TOPLEFT:
            if to_coord_type == CoordinateFormat.CORNERS:
                new_coordinates = BBox.topleft_to_corners_format(coordinates)
            elif to_coord_type == CoordinateFormat.CENTER:
                new_coordinates = BBox.topleft_to_center_format(coordinates)
        else:
            raise ValueError(f"Unsupported format: {from_coord_type}")
        
        # Convert scale if necessary
        if to_scale_type != from_scale_type:
            if img_shape is not None:
                img_height, img_width = img_shape
            else:
                raise ValueError("Image dimensions are required for scale conversion")
            
            # Create a format that represents the current coordinate format but with the current scale
            current_format = None
            if to_coord_type == CoordinateFormat.CORNERS:
                current_format = BBoxFormat.CORNERS_ABSOLUTE if from_scale_type == ScaleFormat.ABSOLUTE else BBoxFormat.CORNERS_NORMALIZED
            elif to_coord_type == CoordinateFormat.CENTER:
                current_format = BBoxFormat.CENTER_ABSOLUTE if from_scale_type == ScaleFormat.ABSOLUTE else BBoxFormat.CENTER_NORMALIZED
            elif to_coord_type == CoordinateFormat.TOPLEFT:
                current_format = BBoxFormat.TOPLEFT_ABSOLUTE if from_scale_type == ScaleFormat.ABSOLUTE else BBoxFormat.TOPLEFT_NORMALIZED
            
            if to_scale_type == ScaleFormat.NORMALIZED:
                new_coordinates = BBox.normalize(new_coordinates, img_width, img_height, current_format)
            else:
                new_coordinates = BBox.absolute(new_coordinates, img_width, img_height, current_format)
        
        return tuple(new_coordinates)

    def get_coordinates(self, fmt: BBoxFormat = BBOX_INNER_FORMAT) -> Tuple[float, float, float, float]:
        """
        Get the bounding box coordinates in the specified format.

        Args:
            fmt (BBoxFormat, optional): The desired format for the bounding box 
                coordinates. If None, the coordinates are returned in the 
                internal format (BBOX_INNER_FORMAT).

        Returns:
            Tuple[float, float, float, float]: The bounding box coordinates in the specified format.

        Raises:
            ValueError: If the specified format is not supported.
            ValueError: If required parameters for the conversion are missing.

        Note:
            Internally, the BBox class always stores coordinates in CORNERS_ABSOLUTE format.
        """
        if fmt is None or fmt == BBOX_INNER_FORMAT:
            return tuple(self.__coordinates)
        
        return BBox.converter(
            self.__coordinates,
            from_fmt = BBOX_INNER_FORMAT,
            to_fmt = fmt,
            img_shape = (self.img_height, self.img_width)
        )
        
    def set_coordinates(
        self,
        coordinates: Union[List, Tuple, np.ndarray],
        fmt: BBoxFormat,
        img_height: int = None,
        img_width: int = None
    ):
        """
        Set the bounding box coordinates.

        Args:
            coordinates: Input bounding box coordinates.
            fmt: Input format of the coordinates (BBoxFormat enum).
            img_height: The height of the image (required for normalization).
            img_width: The width of the image (required for normalization).
        """
        # If the input is a NumPy array, convert it to a list
        if isinstance(coordinates, np.ndarray):
            coordinates = coordinates.tolist()

        # Check that the co-ordinates are a list or tuple of 4 elements
        if not isinstance(coordinates, (list, tuple)) or len(coordinates) != 4:
            raise ValueError(f"coordinates must be a list or tuple of 4 elements. Passed {type(coordinates)}.")

        # Check if the format is normalized and image dimensions are provided
        if fmt in [BBoxFormat.CORNERS_NORMALIZED, BBoxFormat.CENTER_NORMALIZED,
                  BBoxFormat.TOPLEFT_NORMALIZED, BBoxFormat.YOLO]:
            if img_height is None or img_width is None:
                raise ValueError("Image dimensions (img_height and img_width) are required for normalized formats")
            if not BBox.is_normalized(coordinates):
                raise ValueError("Coordinates must be normalized for the specified format")
        
        self.__coordinates = BBox.converter(
            coordinates, 
            from_fmt = fmt, 
            to_fmt = BBOX_INNER_FORMAT, 
            img_shape = (img_height, img_width)
        )
        self.img_width = img_width
        self.img_height = img_height

    def compute_iou(
        self,
        other: 'BBox',
        iou_type: IoUType = IoUType.IOU
    ) -> float:
        """
        Compute the Intersection over Union (IoU) between two bounding boxes.

        Args:
            other: The second bounding box as a BBox object.
            iou_type: The type of IoU to compute. It can be IoUType.IOU, IoUType.DIOU, IoUType.CIOU, 
                     or IoUType.GIOU. Default is IoUType.IOU.

        Returns:
            float: The computed IoU value.

        Raises:
            TypeError: If other is not a BBox object or if the iou_type is not an IoUType.
            ValueError: If the iou_type is invalid.
        """
        # Validate that other is a BBox instance
        if not isinstance(other, BBox):
            raise TypeError("other must be a BBox object")
            
        if not isinstance(iou_type, IoUType):
            raise TypeError("iou_type must be an IoUType")
            
        # Call the internal computation method with BBoxFormat.CORNERS_ABSOLUTE 
        # Both BBox objects should independently handle conversion to absolute coordinates
        # using their own internal image dimensions
        return BBox._compute_iou(self, other, BBoxFormat.CORNERS_ABSOLUTE, None, iou_type)
    
    def __repr__(self):
        return f"BBox(coordinates={self.get_coordinates()}, fmt={BBOX_INNER_FORMAT.to_string()}, img_height={self.img_height}, img_width={self.img_width})"


class BBoxClassId(BBox):
    """
    A bounding box class that includes class identification information.

    This class extends the base BBox class by adding class identification capabilities
    through a class ID and optional class name. It supports various coordinate formats
    and provides methods for data retrieval and manipulation.

    Attributes:
        img_height (int): Height of the reference image in pixels.
        img_width (int): Width of the reference image in pixels.
        coordinates (Union[List, Tuple]): Bounding box coordinates in the specified format.
        class_id (int): Numeric identifier for the object class.
        class_name (Optional[str]): Human-readable name for the object class.
        fmt (BBoxFormat): Format specification for the coordinates. Defaults to BBoxFormat.YOLO.
    """

    def __init__(
        self,
        coordinates: Union[List, Tuple],
        class_id: int,
        class_name: Optional[str] = None,
        fmt: BBoxFormat = BBoxFormat.YOLO,
        img_height: int = None,
        img_width: int = None
    ):
        """
        Initializes a bounding box object with image dimensions, coordinates, class ID, and class name.

        Args:
            img_height (int): The height of the image.
            img_width (int): The width of the image.
            coordinates (Union[List, Tuple]): The coordinates of the bounding box.
            class_id (int): The ID of the class to which the bounding box belongs.
            class_name (Optional[str]): The name of the class to which the bounding box belongs.
            fmt (BBoxFormat, optional): The format of the bounding box coordinates. Defaults to BBoxFormat.YOLO.
        """
        super().__init__(coordinates, fmt, img_height, img_width)
        self._class_id = class_id
        self._class_name = class_name

    @property
    def class_id(self) -> int:
        return self._class_id

    @class_id.setter
    def class_id(self, value: int):
        self._class_id = value

    @property
    def class_name(self) -> Optional[str]:
        return self._class_name

    @class_name.setter
    def class_name(self, value: Optional[str]):
        self._class_name = value

    def get_data(self, fmt: Optional[BBoxFormat] = None, img_shape: tuple = None) -> Tuple[Union[List, Tuple], Union[int, str]]:
        """
        Retrieve the bounding box data including coordinates and class information.

        Args:
            fmt (Optional[BBoxFormat]): The format in which to return the coordinates.
                                        If None, the default format is used.

        Returns:
            tuple: A tuple containing the coordinates of the bounding box
                   and the class information (either class ID or class name).
        """
        return (self.get_coordinates(fmt, img_shape), self.class_id if not self.class_name else self.class_name)

    def __getitem__(self, idx):
        """
        Enable numpy-style indexing for the bounding box coordinates and class ID.

        This method allows accessing the bounding box data (coordinates + class_id)
        using index notation. The coordinates are concatenated with the class ID
        to form a single array that can be indexed.

        Args:
            idx: Index or slice to access the data. Valid indices are:
                0-3: Access the bounding box coordinates
                4: Access the class ID

        Returns:
            float: The requested coordinate value or class ID.

        Example:
            >>> bbox = BBoxClassId(480, 640, [0.1, 0.2, 0.3, 0.4], class_id=1)
            >>> bbox[0]  # Returns first coordinate (0.1)
            >>> bbox[4]  # Returns class_id (1)
            >>> bbox[1:3]  # Returns array([0.2, 0.3])
        """
        return np.array(list(self.get_coordinates()) + [self.class_id])[idx]
    
    def __iter__(self) -> Iterator[Union['BBox', int]]:
        """
        Allow unpacking: (BBox, class_id)
        """
        yield self
        yield self.class_id

    def __repr__(self):
        """
        Returns a string representation of the BBoxClassId object.

        The string includes the image height, image width, normalized coordinates,
        class ID, and class name of the bounding box.

        Returns:
            str: A formatted string representing the BBoxClassId object.
        """
        return f"""BBoxClassId(
            img_height={self.img_height},
            img_width={self.img_width},
            coordinates={self.get_coordinates()},
            class_id={self.class_id},
            class_name={self.class_name})"""

    @classmethod
    def from_yolo_file(cls, file_path: str, img_height: int, img_width: int) -> Tuple[List['BBoxClassId']]:
        bbox, class_id = super().from_yolo_file(file_path, img_height, img_width)
        return cls(
            img_height = bbox.get_img_height(),
            img_width = bbox.get_img_width(),
            coordinates = bbox.get_coordinates(),
            class_id = class_id
        )


class BBoxClassIdScore(BBoxClassId):
    """
    BBoxClassIdScore is a class that extends BBoxClassId to include a score attribute.
    Attributes:
        img_height (int): The height of the image.
        img_width (int): The width of the image.
        coordinates (Union[List, Tuple]): The coordinates of the bounding box.
        class_id (int): The class ID of the object.
        class_name (Optional[str]): The class name of the object.
        score (float): The confidence score of the detection.
        fmt (BBoxFormat): The format of the bounding box coordinates.
    Methods:
        score: Getter and setter for the confidence score of the detection.
    """

    def __init__(
        self,
        coordinates: Union[List, Tuple],
        class_id: int,
        class_name: Optional[str],
        score: float,
        fmt: BBoxFormat = BBoxFormat.YOLO,
        img_height: int = None,
        img_width: int = None
    ):
        """
        Initializes a bounding box object with the given parameters.

        Args:
            coordinates (Union[List, Tuple]): The coordinates of the bounding box.
            class_id (int): The ID of the class.
            class_name (Optional[str]): The name of the class.
            score (float): The confidence score of the bounding box.
            fmt (BBoxFormat, optional): The format of the bounding box coordinates. Defaults to BBoxFormat.YOLO.
            img_height (int, optional): The height of the image.
            img_width (int, optional): The width of the image.
        """
        super().__init__(coordinates, class_id, class_name, fmt, img_height, img_width)
        self._score = score

    @property
    def score(self) -> float:
        return self._score

    @score.setter
    def score(self, value: float):
        self._score = value

    def get_data(self, fmt: Optional[BBoxFormat] = None, img_shape: tuple = None) -> Tuple[Tuple[Union[List, Tuple], Union[int, str]], float]:
        """
        Retrieve the data of the bounding box along with its score.

        Args:
            fmt (Optional[BBoxFormat]): The format for the bounding box data.
                                 If None, the default format will be used.

        Returns:
            tuple: A tuple containing the bounding box data and the score.
        """
        return super().get_data(fmt, img_shape), self.score
    
    def __iter__(self) -> Iterator[Union['BBox', int, float]]:
        """
        Allow unpacking: (BBox, class_id, score)
        """
        yield self
        yield self.class_id
        yield self.score

    def __repr__(self):
        """
        Return a string representation of the BBoxClassIdScore object.

        The string includes the image height, image width, normalized coordinates,
        class ID, class name, and score of the bounding box.

        Returns:
            str: A string representation of the BBoxClassIdScore object.
        """
        return f"""BBoxClassIdScore(
            img_height={self.img_height},
            img_width={self.img_width},
            coordinates={self.get_coordinates()},
            class_id={self.class_id},
            class_name={self.class_name},
            score={self.score})"""


def nms(bboxes: List[BBox], scores: List[float], iou_threshold: float, max_boxes: int = None) -> List[BBox]:
    """
    Perform Non-Maximum Suppression (NMS) on a list of BBox objects.

    Args:
        bboxes: List of BBox objects.
        scores: List of confidence scores corresponding to the bounding boxes.
        iou_threshold: IoU threshold for suppression.
        max_boxes: Maximum number of boxes to keep (optional).

    Returns:
        List of BBox objects after applying NMS.
    """

    # Pair bounding boxes with their scores and sort them by scores in descending order
    boxes_with_scores = sorted(zip(bboxes, scores), key=lambda x: x[1], reverse=True)

    selected_bboxes = []
    while boxes_with_scores:
        # Select the box with the highest score
        current_box, current_score = boxes_with_scores.pop(0)
        selected_bboxes.append(current_box)

        # Remove boxes that have IoU greater than the threshold with the current box
        remaining_boxes = []
        for other_box, other_score in boxes_with_scores:
            iou = current_box.compute_iou(other_box)
            if iou <= iou_threshold:
                remaining_boxes.append((other_box, other_score))

        boxes_with_scores = remaining_boxes

        # Stop if we've reached the max number of boxes
        if max_boxes is not None and len(selected_bboxes) >= max_boxes:
            break

    return selected_bboxes


def draw_boxes_on_image(image: Image, bboxes: List[BBox], color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> Image:
    """
    Draw bounding boxes on an image.

    Args:
        image: Input image as an instance of the Image class.
        bboxes: List of BBox objects to draw on the image.
        color: Color of the bounding boxes in BGR format (default is green).
        thickness: Thickness of the bounding box lines (default is 2).

    Returns:
        A new instance of the Image class with bounding boxes drawn.
    """
    # Get the image data as a numpy array
    image_data = image.get_data()

    # Create a copy of the image to avoid modifying the original
    image_with_boxes = image_data.copy()

    # Determine the color mode of the input image
    if len(image_data.shape) == 2 or (len(image_data.shape) == 3 and image_data.shape[-1] == 1):
        input_color_mode = ColorMode.GRAY
    elif image_data.shape[2] == 3:
        # Check if the image is in RGB or BGR format (OpenCV uses BGR by default)
        # Here we assume the input is BGR unless explicitly converted
        input_color_mode = ColorMode.BGR
    else:
        raise ValueError("Unsupported image format. Expected grayscale (1 channel) or BGR/RGB (3 channels).")

    # Convert grayscale images to BGR (3 channels) to support colored bounding boxes
    if input_color_mode == ColorMode.GRAY:
        tmpimg = image.copy()
        tmpimg.convert_color_mode(ColorMode.BGR)
        image_with_boxes = tmpimg.get_data()
        output_color_mode = ColorMode.BGR
    else:
        image_with_boxes = image_data.copy()
        output_color_mode = input_color_mode

    for bbox in bboxes:
        # Check if the bbox is an instance of the BBox class
        if not isinstance(bbox, BBox):
            raise TypeError(f"Expected an instance of BBox, but got {type(bbox)}")

        # Convert the bounding box to corners format (x1, y1, x2, y2)
        x1, y1, x2, y2 = bbox.get_coordinates(fmt=BBoxFormat.CORNERS_ABSOLUTE)

        # Draw the bounding box on the image
        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), color, thickness)

    # Create a new instance of the Image class with the modified image data
    new_image = Image(image_with_boxes, color_mode=output_color_mode)

    return new_image


def print_bbox_info(boxes: List[BBox], class_ids: List[int], scores: List[float], class_labels_dict: Dict[int, str]):
    """
    Print information about the detected bounding boxes.

    Args:
        boxes: List of bounding boxes.
        class_ids: List of class IDs.
        scores: List of confidence scores.
        class_labels_dict: Dictionary mapping class IDs to class names.
    """
    print("\nDetected bounding boxes:")
    for i, (bbox, class_id, score) in enumerate(zip(boxes, class_ids, scores)):
        # Get bounding box coordinates
        x1, y1, x2, y2 = bbox.get_coordinates()

        # Get class name
        class_name = class_labels_dict.get(class_id, f"class_{class_id}")

        # Get the coordinate format as a string
        coordinate_format = bbox.fmt.to_string()

        # Print information
        print(f"Box {i + 1}:")
        print(f"  Class: {class_name} (ID: {class_id})")
        print(f"  Confidence: {score:.4f}")
        print(f"  Coordinate Format: {coordinate_format}")
        print(f"  Coordinates: (x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2})")
        print("-" * 40)


def draw_boxes_on_image_with_labels_score(
    image: Image,
    bboxes_with_labels_score: List[Tuple[BBox, int, float]],
    class_colors_bgr: Optional[Dict[int, Tuple[int, int, int]]] = None,
    class_labels: Optional[Dict[int, str]] = None,
    thickness: int = 2,
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 0.6,
    font_thickness: int = 1,
    text_color: Tuple[int, int, int] = (0, 0, 0),  # Black text by default
    text_background_color: Tuple[int, int, int] = (255, 255, 255),  # White background by default
) -> Image:
    """
    Draw bounding boxes on an image with class labels and scores.

    Args:
        image: The image on which to draw the bounding boxes, as an instance of the Image class.
        bboxes_with_labels_score: A list of tuples containing the bounding box, class ID, and score.
        class_colors_bgr: Optional dictionary mapping class IDs to colors (BGR format). Default is None.
        class_labels: Optional dictionary mapping class IDs to label strings. Default is None.
        thickness: Thickness of the bounding box lines.
        font: Font type for the text.
        font_scale: Font scale for the text.
        font_thickness: Thickness of the text.
        text_color: Color of the text. Default is black.
        text_background_color: Color of the text background rectangle. Default is white.

    Returns:
        A new instance of the Image class with bounding boxes and labels drawn.
    """
    # Get the image data as a numpy array
    image_data = image.get_data()

    # Determine the color mode of the input image
    if len(image_data.shape) == 2 or (len(image_data.shape) == 3 and image_data.shape[-1] == 1):
        input_color_mode = ColorMode.GRAY
    elif image_data.shape[2] == 3:
        # Check if the image is in RGB or BGR format (OpenCV uses BGR by default)
        # Here we assume the input is BGR unless explicitly converted
        input_color_mode = ColorMode.BGR
    else:
        raise ValueError("Unsupported image format. Expected grayscale (1 channel) or BGR/RGB (3 channels).")

    # Convert grayscale images to BGR (3 channels) to support colored bounding boxes
    if input_color_mode == ColorMode.GRAY:
        tmpimg = image.copy()
        tmpimg.convert_color_mode(ColorMode.BGR)
        image_with_boxes = tmpimg.get_data()
        output_color_mode = ColorMode.BGR
    else:
        image_with_boxes = image_data.copy()
        output_color_mode = input_color_mode

    img_height = image.get_height()
    img_width = image.get_width()

    # Default color (white) if class_colors_bgr is not provided
    default_color = (255, 255, 255)  # White in BGR format

    for bbox, class_id, score in bboxes_with_labels_score:
        # Get the color for the current class_id
        if class_colors_bgr is not None:
            # Use default color if class_id not found
            color = class_colors_bgr.get(class_id, default_color)
        else:
            color = default_color  # Use default color if class_colors_bgr is not provided

        # Draw the bounding box on the image using the draw_boxes_on_image function
        image_with_boxes = draw_boxes_on_image(Image(image_with_boxes, output_color_mode), [bbox], color, thickness).get_data()

        # Get the label for the current class_id
        if class_labels is not None:
            # Use default label if class_id not found
            label = class_labels.get(class_id, f"Class {class_id}")
        else:
            # Use default label if class_labels is not provided
            label = f"Class {class_id}"

        score_text = f"{score:.2f}"
        text = f"{label} - {score_text}"

        # Get the bounding box corners in absolute coordinates
        corners = bbox.get_coordinates(fmt=BBoxFormat.CORNERS_ABSOLUTE)
        x1, y1, x2, y2 = corners

        # Calculate text size and position
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_x = x1
        text_y = y1 - 10 if y1 - \
            10 > 10 else y1 + 20

        # Ensure text is within image bounds
        if text_y - text_height < 0:
            # Move text below the box if it goes above the image
            text_y = y1 + 20
        if text_x + text_width > img_width:
            text_x = img_width - text_width  # Move text left if it goes beyond the image width

        # Draw a background rectangle for the text
        cv2.rectangle(
            image_with_boxes,
            (text_x, text_y - text_height),
            (text_x + text_width, text_y),
            text_background_color,
            -1,  # Fill the rectangle
        )

        # Draw the text on the image
        cv2.putText(
            image_with_boxes,
            text,
            (text_x, text_y),
            font,
            font_scale,
            text_color,
            font_thickness,
            lineType=cv2.LINE_AA,
        )

    # Create a new instance of the Image class with the modified image data
    new_image = Image(image_with_boxes, color_mode=output_color_mode)

    return new_image
