import numpy as np
from typing import Union, Any, List, Tuple
import time
import cv2

from saltup.ai.object_detection.yolo.yolo import BaseYolo, YoloType, YoloOutput
from saltup.ai.object_detection.utils.anchor_based_model import (
    postprocess_decode, postprocess_filter_boxes, tiny_anchors_based_nms
)
from saltup.ai.object_detection.utils.bbox import BBox, BBoxFormat
from saltup.utils.data.image.image_utils import  ColorMode ,ImageFormat
from saltup.utils.data.image.image_utils import Image


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

    def get_input_info(self) -> Tuple[tuple, ColorMode, ImageFormat]:
        input_shape = self.model_input_shape[1:]  # Rimuove il batch size
        return (
            input_shape,  # Shape: (480, 640,1)
            ColorMode.RGB,
            ImageFormat.HWC
        )
    
    @staticmethod
    def preprocess(
        image: Union[np.ndarray, Image],
        target_height: int,
        target_width: int,
        normalize_method: callable = lambda x: x.astype(np.float32) / 255.0,
        apply_padding: bool = True,
        **kwargs: Any
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
        if isinstance(image, Image):
            raw_img = image.get_data()
            num_channel = image.get_number_channel()
        elif isinstance(image, np.ndarray):
            raw_img = image
            num_channel = 1 if len(image.shape) < 3 else image.shape[2]
        else:
            raise TypeError(f"Invalid type {type(image)} for image: should be 'np.ndarray' or 'saltup.Image'.")
        
        # Validate input format
        if num_channel not in [1, 3]:
            raise ValueError("Only 1 or 3 channels are supported for multi-channel images")

        # Extract dimensions
        if isinstance(image, Image):
            height = image.get_height()
            width = image.get_width()
        else:
            height = image.shape[1]
            width = image.shape[0]

        # Handle square padding if needed
        if target_height == target_width and apply_padding:
            max_dim = max(height, width)
            if num_channel == 1:
                padded = np.full((max_dim, max_dim), 114, dtype=np.uint8)
                padded[0:height, 0:width] = raw_img
            else:
                padded = np.full((max_dim, max_dim, num_channel),
                                 114, dtype=np.uint8)
                padded[0:height, 0:width, :] = raw_img
            raw_img = padded

        # Scale to target dimensions
        raw_img = cv2.resize(raw_img, (target_width, target_height),
                           interpolation=cv2.INTER_LINEAR)

        # Normalize pixel values
        raw_img = normalize_method(raw_img)

        # Ensure proper channel dimension
        if num_channel == 1:
            raw_img = np.expand_dims(raw_img, axis=-1)

        # Prepare batch dimension
        raw_img = np.expand_dims(raw_img, axis=0)

        return raw_img.astype(np.float32)

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
            image_height (int): input height of the current inferenced image
            image_width (int): input width of the current inferenced image
            confidence_thr (float): the threshold of the confidence score
            iou_threshold (float): the threshold of the Intersection over Union for NMS

        Returns:
            List[Tuple[BBox, int, float]]: List of predicted bounding boxes in the image with their respective score and class_id.
        """
        anchors = np.array(self.anchors).reshape(-1, 2)
        input_shape = (self.input_model_height, self.input_model_width)
        
        if isinstance(raw_output, list):
            raw_output = raw_output[0]
            
        preds_decoded = postprocess_decode(
            raw_output, anchors, self.number_class, input_shape, calc_loss=False)
        input_image_shape = [image_height, image_width]

        corners_boxes, scores, classes, centers_boxes = tiny_anchors_based_nms(
            yolo_outputs=preds_decoded,
            image_shape=input_image_shape,
            max_boxes=self.max_output_boxes,
            score_threshold=confidence_thr,
            iou_threshold=iou_threshold,
            classes_ids=list(range(0, self.number_class))
        )

        result = []
        for i, c in reversed(list(enumerate(classes))):
            box = corners_boxes[i]
            score = scores[i]

            if all(coord >= 0 for coord in box):
                box_object = BBox(
                    coordinates=box, 
                    fmt=BBoxFormat.CORNERS_ABSOLUTE,
                    img_height=image_height, 
                    img_width=image_width
                )
                result.append((box_object, int(c), score))
            else:
                print(f"Warning: Box {box} contains negative values and will be ignored.")

        return result
