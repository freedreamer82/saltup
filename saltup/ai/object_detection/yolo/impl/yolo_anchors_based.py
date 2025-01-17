import numpy as np
from typing import Optional, Union, Callable, Dict, Any, List, Tuple
import time
from saltup.ai.object_detection.yolo.yolo import BaseYolo, YoloType, YoloOutput
from saltup.ai.object_detection.yolo.preprocessing.anchors_based_preprocess import AnchorsBasedPreprocess
from saltup.ai.object_detection.yolo.postprocessing.anchors_based_postprocessing import AnchorsBasedPostprocess
from saltup.ai.object_detection.utils.bbox import BBox, BBoxFormat

class YoloAnchorsBased(BaseYolo):
    """
    A class that extends BaseYolo to handle anchor-based YOLO models.
    This class adds functionality to manage anchor boxes and adjust predictions based on them.
    """
    def __init__(self, yolot: YoloType, model_path: str, number_class:int, anchors: np.ndarray, max_output_boxes:int=10):
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

    def preprocess(self, image: np.ndarray, target_height:int, target_width:int) -> np.ndarray:
        """
        Preprocess the image before model inference.
        This method can be overridden to include anchor-specific preprocessing if needed.

        :param image: Input image to preprocess.
        :return: Preprocessed image.
        """
        preprocessor = AnchorsBasedPreprocess()
        return preprocessor(image, (target_height, target_width))

    
    def postprocess(self, raw_output: np.ndarray, 
                    image_height:int, image_width:int, confidence_thr:float=0.5, 
                            iou_threshold:float=0.5) -> List[Tuple[BBox, int, float]]:
        """
        Postprocess the raw output from the model to produce structured results.
        This method adjusts the predictions based on the anchor boxes.

        :param raw_output: Raw results from the model.
        :return: A YoloOutput object containing the adjusted predictions.
        """
        postprocessor = AnchorsBasedPostprocess()
        return postprocessor(raw_output, self.number_class, self.anchors, self.img_input_height, self.img_input_width, image_height, 
                                image_width, self.max_output_boxes, confidence_thr, iou_threshold)
 