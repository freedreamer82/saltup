
import numpy as np
from typing import (
    Optional, Callable, Union,
    Any, Dict, List ,Tuple,
)
import time
import cv2
from collections import defaultdict
from abc import ABC, abstractmethod

from saltup.ai.object_detection.utils.bbox  import BBox,BBoxFormat
from saltup.ai.object_detection.utils.metrics  import compute_ap, compute_map_50_95 , compute_ap_for_threshold
from saltup.ai.object_detection.yolo.yolo_type  import YoloType
from saltup.utils.data.image.image_utils import load_image, ColorMode ,ImageFormat
from saltup.ai.nn_manager import NeuralNetworkManager
from saltup.utils.data.image.image_utils import Image


class YoloOutput:
    """Class to represent the results of the YOLO model."""
    def __init__(
        self,
        boxes: List[Tuple[BBox, int, float]],
        image: Optional[np.ndarray] = None,
    ):
        """
        Initialize YoloOutput with bounding boxes, scores, and an optional image.

        Args:
            boxes: List of BBox objects representing the bounding boxes.
            scores: List of confidence scores.
            image: Optional image associated with the results.
        """
        self._boxes = boxes  # List of BBox objects
        self._image = image  # Optional image associated with the results

        self._inference_time_ms = 0.0  
        self._preprocessing_time_ms = 0.0 
        self._postprocessing_time_ms = 0.0 

    tprocessing_time_ms = 0.0 

    def get_boxes(self, format: Optional[BBoxFormat] = None) -> List[Tuple['BBox', int, float]]:
        """
        Get the list of bounding boxes in the specified format.

        Args:
            format: The desired format (CORNERS, CENTER, TOPLEFT). If None, leaves the format unchanged.

        Returns:
            List of tuples containing:
                - BBox objects with coordinates in the specified format (or current format if None).
                - Class ID (int).
                - Confidence score (float).
        """
        formatted_boxes = []
        for bbox, class_id, confidence in self._boxes:
            if format is None or bbox.get_format() == format:
                # If format is None or already matches, reuse the existing BBox object
                formatted_boxes.append((bbox, class_id, confidence))
            else:
                # If the format is different, create a new BBox object in the desired format
                new_bbox = BBox(
                    coordinates=bbox.get_coordinates(format),
                    format=format,
                    img_width=bbox.img_width,
                    img_height=bbox.img_height
                )
                formatted_boxes.append((new_bbox, class_id, confidence))
        return formatted_boxes
    
    def set_boxes(self, boxes: List[BBox]):
        """
        Set the list of bounding boxes.

        Args:
            boxes: List of BBox objects.
        """
        self._boxes = boxes
        
    def get_image(self) -> Optional[np.ndarray]:
        """Get the associated image."""
        return self._image

    def set_image(self, image: Optional[np.ndarray]):
        """Set the associated image."""
        self._image = image

    def get_inference_time(self) -> float:
        """Get the inference time in milliseconds."""
        return self._inference_time_ms

    def set_inference_time(self, value: float):
        """Set the inference time in milliseconds."""
        self._inference_time_ms = value

    def get_preprocessing_time(self) -> float:
        """Get the pre-processing time in milliseconds."""
        return self._preprocessing_time_ms

    def set_preprocessing_time(self, value: float):
        """Set the pre-processing time in milliseconds."""
        self._preprocessing_time_ms = value

    def get_postprocessing_time(self) -> float:
        """Get the post-processing time in milliseconds."""
        return self._postprocessing_time_ms

    def set_postprocessing_time(self, value: float):
        """Set the post-processing time in milliseconds."""
        self._postprocessing_time_ms = value

    def get_total_processing_time(self) -> float:
        """
        Get the total processing time (pre-processing + inference + post-processing) in milliseconds.

        Returns:
            Total processing time in milliseconds.
        """
        return (
            self._preprocessing_time_ms
            + self._inference_time_ms
            + self._postprocessing_time_ms
        )

    def get_property(self, property_name: str) -> Any:
        """
        Get a property by name.

        Args:
            property_name: Name of the property ("boxes", "image",
                          "inference_time", "preprocessing_time", "postprocessing_time").

        Returns:
            The value of the property.

        Raises:
            AttributeError: If the property does not exist.
        """
        if property_name == "boxes":
            return self._boxes
        elif property_name == "image":
            return self._image
        elif property_name == "inference_time":
            return self._inference_time_ms
        elif property_name == "preprocessing_time":
            return self._preprocessing_time_ms
        elif property_name == "postprocessing_time":
            return self._postprocessing_time_ms
        else:
            raise AttributeError(f"Property '{property_name}' does not exist.")

    def set_property(self, property_name: str, value: Any):
        """
        Set a property by name.

        Args:
            property_name: Name of the property ("boxes", "image",
                          "inference_time", "preprocessing_time", "postprocessing_time").
            value: The new value for the property.

        Raises:
            AttributeError: If the property does not exist.
        """
        if property_name == "boxes":
            self._boxes = value
        elif property_name == "image":
            self._image = value
        elif property_name == "inference_time":
            self._inference_time_ms = value
        elif property_name == "preprocessing_time":
            self._preprocessing_time_ms = value
        elif property_name == "postprocessing_time":
            self._postprocessing_time_ms = value
        else:
            raise AttributeError(f"Property '{property_name}' does not exist.")

    def __repr__(self):
        return (
            f"YoloOutput(boxes={self._boxes}"
            f"image={self._image is not None}, inference_time={self._inference_time_ms} ms, "
            f"preprocessing_time={self._preprocessing_time_ms} ms, "
            f"postprocessing_time={self._postprocessing_time_ms} ms)"
        )
   

class BaseYolo(NeuralNetworkManager):
    """Base class for implementing a generic YOLO model."""
    def __init__(self, yolot: YoloType, model_path: str, number_class:int):
        super().__init__()  # Initialize NeuralNetworkManager
        self.model_path = model_path
        self.number_class = number_class
        self.model, self.model_input_shape, self.model_output_shape = self.load_model(model_path)  # Load model using inherited method
        if yolot == YoloType.ANCHORS_BASED:
            self.img_input_height  = self.model_input_shape[1]
            self.img_input_width  = self.model_input_shape[2]
        else:
            self.img_input_height  = self.model_input_shape[2]
            self.img_input_width  = self.model_input_shape[3]
        self.yolotype = yolot

    def getYoloType(self) -> YoloType:
        return self.yolotype
    
    def _validate_input_preprocessing_image(self, img: np.ndarray) -> None:
        """Validate the input image format and type.
        
        Args:
            img: Input image to validate

        Raises:
            ValueError: If image is None
            TypeError: If image is not a numpy array
        """
        if img is None:
            raise ValueError("Input image cannot be None")
        if not isinstance(img, np.ndarray):
            raise TypeError("Input must be numpy array")
    

    @abstractmethod
    def get_input_info(self) -> Tuple[tuple, ColorMode, ImageFormat]:
        """Abstract method to get input information (shape, color mode, and color format).

        Returns:
            Tuple[tuple, ColorMode, ImageFormat]: A tuple containing:
                - Shape: A tuple representing the input shape (e.g., (H, W, C) or (C, H, W)).
                - ColorMode: An enum value from ColorMode (RGB, BGR, or GRAY).
                - ImageFormat: An enum value from ImageFormat (HWC or CHW).
        """
        pass

    @staticmethod
    def load_anchors(anchors_path:str) -> np.ndarray:
        anchors_data = np.loadtxt(anchors_path, delimiter=",")
        anchors_list = [row for row in anchors_data]
        anchors_array = np.array(anchors_list).reshape(-1, 2)
        
        return anchors_array
    
    @staticmethod
    def load_image(image_path: str, color_mode: ColorMode = ColorMode.BGR, format = ImageFormat.HWC) ->  Image:
        """Load and convert image to specified color mode.

        Args:
            image_path: Path to the image file
            color_mode: Target color mode ("RGB", "BGR", or "GRAY")

        Returns:
            np.ndarray: Image in specified color mode

        Raises:
            FileNotFoundError: If image file does not exist or cannot be loaded
            ValueError: If color conversion fails
        """
        return Image(image_path, color_mode , image_format=format)
    
    def get_number_image_channel(self) -> int:
        return self.model_input_shape[-1]
    
    def run(
        self,
        image: Image,
        img_height: int, 
        img_width: int,
        confidence_thr: float=0.5,
        iou_threshold:float = 0.5,
        preprocess: Optional[Union[Callable, bool]] = None,
        postprocess: Optional[Union[Callable, bool]] = None,
    ) -> YoloOutput:
        """
        Perform inference using the YOLO model.
        This method allows for custom preprocess and postprocess functions,
        and the option to disable preprocessing or postprocessing entirely.

        :param image: Input image for the model.
        :param preprocess: Optional custom preprocessing function. If None, uses the native `preprocess` method.
                           If False, skips preprocessing entirely.
        :param postprocess: Optional custom postprocessing function. If None, uses the native `postprocess` method.
                            If False, skips postprocessing entirely.
        :return: A YoloOutput object containing the results and timing information.
        """
        # Measure preprocessing time
        start_preprocess = time.time()
        if preprocess is False:
            # Skip preprocessing entirely
            preprocessed_image = image
        else:
            # Use custom preprocessing if provided, otherwise use the native method
            if callable(preprocess):
                preprocessed_image = preprocess(image, self.img_input_height, self.img_input_width)
            else:
                preprocessed_image = self.preprocess(image, self.img_input_height, self.img_input_width)
        end_preprocess = time.time()
        preprocessing_time_ms = (end_preprocess - start_preprocess) * 1000

        # Measure inference time
        start_inference = time.time()
        raw_output = self.model_inference(preprocessed_image)
        end_inference = time.time()
        inference_time_ms = (end_inference - start_inference) * 1000

        # Measure postprocessing time
        start_postprocess = time.time()
        if postprocess is False:
            # Skip postprocessing entirely
            postprocessed_output = raw_output
        else:
            # Use custom postprocessing if provided, otherwise use the native method
            if callable(postprocess):
                postprocessed_output = postprocess(raw_output, img_height, img_width, confidence_thr, 
                            iou_threshold)
            else:
                postprocessed_output = self.postprocess(raw_output, img_height, img_width, confidence_thr, 
                            iou_threshold)
        end_postprocess = time.time()
        postprocessing_time_ms = (end_postprocess - start_postprocess) * 1000
        
        yoloOut = YoloOutput(postprocessed_output, image=preprocessed_image)
        yoloOut.set_preprocessing_time(preprocessing_time_ms)
        yoloOut.set_inference_time(inference_time_ms)
        yoloOut.set_postprocessing_time(postprocessing_time_ms)
            
        
        return yoloOut

    @classmethod
    def evaluate(cls,predictions: YoloOutput, ground_truth: List[Tuple[BBox, int]], threshold_iou: float = 0.5) -> Dict[str, float]:
        """
        Compute evaluation metrics (precision, recall, F1-score, mAP, mAP@50-95) and TP, FP, FN.

        Args:
            predictions: YoloOutput object containing predicted bounding boxes, scores, and class IDs.
            ground_truth: List of tuples (BBox, class_id) representing ground truth.
            threshold_iou: IoU threshold to consider a detection as a true positive.

        Returns:
            Dictionary containing evaluation metrics and counts of TP, FP, FN.
        """
        if not ground_truth:  # No ground truth
            return {metric: 0.0 for metric in ["precision", "recall", "f1", "mAP", "mAP@50-95", "tp", "fp", "fn"]}

        if not predictions.get_boxes():  # No predictions
            return {metric: 0.0 for metric in ["precision", "recall", "f1", "mAP", "mAP@50-95", "tp", "fp", "fn"]}

        # Group ground truth and predictions by class ID
        gt_by_class = defaultdict(list)
        for gt_bbox, class_id in ground_truth:
            gt_by_class[class_id].append(gt_bbox)

        pred_by_class = defaultdict(list)
        for pred_bbox, class_id, score in predictions.get_boxes():
            pred_by_class[class_id].append((pred_bbox, score))

        # Initialize global counters and AP lists
        global_tp, global_fp, global_fn = 0, 0, 0
        aps = []

        for class_id in gt_by_class:
            gt_bboxes = gt_by_class[class_id]
            pred_bboxes_scores = pred_by_class.get(class_id, [])
            
            # Calculate AP for the class at the given threshold
            ap = compute_ap_for_threshold(gt_bboxes, pred_bboxes_scores, threshold_iou)
            aps.append(ap)

            # Match predictions to ground truth
            pred_bboxes_scores.sort(key=lambda x: x[1], reverse=True)
            pred_bboxes = [x[0] for x in pred_bboxes_scores]

            # Initialize TP, FP and matched ground truths
            tp = np.zeros(len(pred_bboxes))
            fp = np.zeros(len(pred_bboxes))
            gt_matched = [False] * len(gt_bboxes)

            for i, pred_bbox in enumerate(pred_bboxes):
                max_iou = 0
                best_match_idx = -1
                for j, gt_bbox in enumerate(gt_bboxes):
                    if gt_matched[j]:
                        continue
                    iou = pred_bbox.compute_iou(gt_bbox)
                    if iou > max_iou:
                        max_iou = iou
                        best_match_idx = j

                if max_iou >= threshold_iou and best_match_idx != -1:
                    tp[i] = 1
                    gt_matched[best_match_idx] = True
                else:
                    fp[i] = 1

            # Update global counters
            global_tp += np.sum(tp)
            global_fp += np.sum(fp)
            global_fn += len(gt_bboxes) - np.sum(tp)

        # Calculate global precision, recall, and F1-score
        precision = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0.0
        recall = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Calculate mAP using compute_ap_range (includes AP@50-95)
        mAP_50_95 = compute_map_50_95(
            [gt for gt_class in gt_by_class.values() for gt in gt_class],
            [(pred, score) for preds in pred_by_class.values() for pred, score in preds],
        )

        # Calculate mAP (average AP across classes at the given threshold)
        mAP = np.mean(aps) if aps else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mAP": mAP,
            "mAP@50-95": mAP_50_95,
            "tp": int(global_tp),  # True Positives
            "fp": int(global_fp),  # False Positives
            "fn": int(global_fn),  # False Negatives
        }
    def preprocess(self, 
                   image: Image,
                   target_height:int, 
                   target_width:int,        
                   normalize_method: callable = lambda x: x.astype(np.float32) / 255.0,
                   apply_padding: bool = True,
                   **kwargs: Any
                   ) -> np.ndarray:
        """
        Preprocess the image before model inference.

        :param image: Input image to preprocess.
        :return: Preprocessed image.
        """
        raise NotImplementedError("The preprocess method must be overridden in the derived class.")

    def postprocess(self, raw_output: np.ndarray,
                    image_height:int, image_width:int, confidence_thr:float=0.5, 
                            iou_threshold:float=0.5) -> List[Tuple[BBox, int, float]]:
        """
        Postprocess the raw output from the model to produce structured results.

        :param raw_output: Raw results from the model.
        :return: A YoloOutput object.
        """
        raise NotImplementedError("The postprocess method must be overridden in the derived class.")