import numpy as np
from typing import Optional, Union, Callable, Dict, Any, List, Tuple
import time
import cv2

from saltup.ai.object_detection.yolo.yolo import BaseYolo, YoloType, YoloOutput
from saltup.ai.object_detection.utils.anchor_based_model import (
    postprocess_decode, postprocess_filter_boxes, tiny_anchors_based_nms
)
from saltup.ai.object_detection.utils.bbox import BBox, BBoxFormat


class YoloAnchorsBased(BaseYolo):
    """
    A class that extends BaseYolo to handle anchor-based YOLO models.
    This class adds functionality to manage anchor boxes and adjust predictions based on them.
    """

    def __init__(
        self,
        yolot: YoloType,
        model_path: str,
        number_class: int,
        anchors: np.ndarray,
        max_output_boxes: int = 10
    ):
        """
        Initialize the AnchorsBased YOLO model.

        :param yolot: Type of YOLO model (e.g., YoloType.YOLOv3, YoloType.YOLOv4).
        :param model_path: Path to the model file.
        :param anchors: A numpy array of anchor boxes with shape (N, 2), where N is the number of anchors.
                       Each anchor is represented as (width, height).
        """
        super().__init__(yolot, model_path, number_class)  # Initialize the BaseYolo class
        self.anchors = anchors  # Store the anchor boxes
        self.num_anchors = anchors.shape[0]  # Number of anchor boxes
        self.max_output_boxes = max_output_boxes

    def _validate_input(self, img: np.ndarray) -> None:
        """Validate input image format and channel structure.

        Extends base validation with specific checks for channel dimensions
        and supported formats.

        Args:
            img: Input image to validate (single or multiple channels)

        Raises:
            ValueError: For invalid dimensions or unsupported channel counts
            TypeError: For non-numpy array inputs (from parent class)
        """
        super()._validate_input_preprocessing_image(img)

        if len(img.shape) not in [2, 3]:
            raise ValueError(
                "Input must be either 2D (single channel) or 3D (multiple channels) array")

        if len(img.shape) == 3 and img.shape[2] not in [1, 3]:
            raise ValueError(
                "Only 1 or 3 channels are supported for multi-channel images")

    def preprocess(
        self,
        image: np.ndarray,
        target_height: int,
        target_width: int,
        normalize_method: callable = lambda x: x.astype(np.float32) / 255.0,
        apply_padding: bool = True
    ) -> np.ndarray:
        """Process image for anchor-based object detection.

        Processing pipeline:
        1. Input validation and channel verification
        2. Channel dimension handling
        3. Square padding (optional, for square targets)
        4. Target size resizing
        5. Pixel normalization
        6. Batch dimension preparation

        Args:
            img: Source image (single or multi-channel)
            target_shape: Required output size as (height, width)

        Returns:
            np.ndarray: Processed image tensor with shape (1, height, width, channels)
            - First dimension: batch size
            - Last dimension: number of channels matches input (1 for single channel, 3 for RGB)
            - Normalized values

        Raises:
            ValueError: For invalid image formats
            TypeError: For incorrect input types
        """
        # Validate input format
        self._validate_input(image)

        # Determine channel configuration
        is_single_channel = len(image.shape) == 2

        # Extract dimensions
        if is_single_channel:
            height, width = image.shape
            channels = 1
        else:
            height, width, channels = image.shape

        # Handle square padding if needed
        if target_height == target_width and apply_padding:
            max_dim = max(height, width)
            if is_single_channel:
                padded = np.full((max_dim, max_dim), 114, dtype=np.uint8)
                padded[0:height, 0:width] = image
            else:
                padded = np.full((max_dim, max_dim, channels),
                                 114, dtype=np.uint8)
                padded[0:height, 0:width, :] = image
            image = padded

        # Scale to target dimensions
        image = cv2.resize(image, (target_width, target_height),
                           interpolation=cv2.INTER_LINEAR)

        # Normalize pixel values
        image = normalize_method(image)

        # Ensure proper channel dimension
        if is_single_channel:
            image = np.expand_dims(image, axis=-1)

        # Prepare batch dimension
        image = np.expand_dims(image, axis=0)

        return image.astype(np.float32)

    def postprocess(
        self,
        raw_output: np.ndarray,
        image_height: int,
        image_width: int,
        confidence_thr: float = 0.5,
        iou_threshold: float = 0.5
    ) -> List[Tuple[BBox, int, float]]:
        """postprocess output from AnchorsBased YOLO model

        Args:
            raw_output (np.ndarray): output matrix of the model
            num_classes (int): number of classes
            anchors (list[float]): anchors representting your dataset in normalized format
            model_input_height (int): input height of the model
            model_input_width (int): input width of the model
            image_height (int): input height of the current inferenced image
            image_width (int): input width of the current inferenced image
            max_output_boxes (int): maximum number of bounding box to be considered for non-max suppression
            confidence_thr (float): the threshold of the confidence score
            iou_threshold (float): the threshold of the Intersection over Union for NMS

        Returns:
            list[list]: list of the preicted bounding box in the image
        """
        anchors = np.array(self.anchors).reshape(-1, 2)
        input_shape = (self.img_input_height, self.img_input_width)
        preds_decoded = postprocess_decode(
            raw_output, anchors, self.number_class, input_shape, calc_loss=False)
        input_image_shape = [image_height, image_width]

        boxes, scores, classes, my_boxes = tiny_anchors_based_nms(
            yolo_outputs=preds_decoded,
            image_shape=input_image_shape,
            max_boxes=self.max_output_boxes,
            score_threshold=confidence_thr,
            iou_threshold=iou_threshold,
            classes_ids=list(range(0, self.number_class))
        )

        result = []
        for i, c in reversed(list(enumerate(classes))):
            box = boxes[i]
            score = scores[i]
            box_object = BBox(box, format=BBoxFormat.CORNERS,
                              img_width=image_width, img_height=image_height)
            result.append((box_object, int(c), score))

        return result
