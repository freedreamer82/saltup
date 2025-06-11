import numpy as np
from typing import (
    Optional, Callable, Union,
    Any, Dict, List, Tuple,
)
import time
from collections import defaultdict
from abc import ABC, abstractmethod
import sys, io

from saltup.ai.object_detection.utils.bbox import BBoxFormat, BBox, BBoxClassId, BBoxClassIdScore
from saltup.ai.object_detection.utils.metrics import compute_ap, compute_map_50_95, compute_ap_for_threshold
from saltup.ai.object_detection.yolo.yolo_type import YoloType
from saltup.ai.object_detection.utils.metrics import Metric
from saltup.utils.data.image.image_utils import ColorMode, ImageFormat, Image
from saltup.ai.nn_model import NeuralNetworkModel
from saltup.ai.base_dataformat.base_dataloader import BaseDataloader


class YoloOutput:
    """Class to represent the results of the YOLO model."""
    def __init__(
        self,
        boxes: List[Tuple[BBox, int, float]],
        image: Optional[Image] = None,
    ):
        """
        Initialize YoloOutput with bounding boxes and an optional image.

        Args:
            boxes: List of tuples (BBox, class_id, score) representing the bounding boxes, class IDs, and confidence scores.
            image: Optional image associated with the results.
        """
        self._boxes = boxes  # List of BBox objects
        self._image = image  # Optional image associated with the results

        self._inference_time_ms = 0.0  
        self._preprocessing_time_ms = 0.0 
        self._postprocessing_time_ms = 0.0 

    tprocessing_time_ms = 0.0 

    def get_boxes(self) -> List[BBoxClassIdScore]:
        """
        Get the list of bounding boxes.

        Returns:
            List of tuples containing:
            - BBox object.
            - Class ID (int).
            - Confidence score (float).
        """
        return self._boxes
    
    def set_boxes(self, boxes: List[BBoxClassIdScore]):
        """
        Set the list of bounding boxes.

        Args:
            boxes: List of tuples containing (BBox, class_id, score).
        """
        self._boxes = boxes
        
    def get_image(self) -> Optional[np.ndarray]:
        """Get the associated image."""
        return self._image

    def set_image(self, image: Optional[Image]):
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
   

class BaseYolo():
    """Base class for implementing a generic YOLO model."""
    def __init__(self, yolot: YoloType, model: NeuralNetworkModel, number_class:int):

        self._number_class = number_class
        self._model = model
        _, self._model_input_shape, self._model_output_shape = model.load()
        
        self._yolotype = yolot
        self._input_model_format = self.get_input_info()[2]
        self._input_model_color = self.get_input_info()[1]
        if self._input_model_format == ImageFormat.HWC:
            self._input_model_height = self.get_input_info()[0][0]
            self._input_model_width = self.get_input_info()[0][1]
            self._input_model_channel = self.get_input_info()[0][2]
        else:   
            self._input_model_height = self.get_input_info()[0][1]
            self._input_model_width = self.get_input_info()[0][2]  
            self._input_model_channel = self.get_input_info()[0][0]

    def get_input_model_height(self) -> int:
        return self._input_model_height

    def get_input_model_width(self) -> int:
        return self._input_model_width

    def get_input_model_channel(self) -> int:
        return self._input_model_channel

    def get_input_model_format(self) -> ImageFormat:
        return self._input_model_format

    def get_input_model_color(self) -> ColorMode:
        return self._input_model_color
    
    def getYoloType(self) -> YoloType:
        return self._yolotype
    
    def get_number_class(self) -> int:
        """Get the number of classes in the YOLO model."""
        return self._number_class

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
    def load_image(image_path: str, color_mode: ColorMode = ColorMode.BGR) ->  Image:
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
        return Image(image_path, color_mode)
    
    def get_number_image_channel(self) -> int:
        info = self.get_input_info()
        if info[2] == ImageFormat.HWC:
            return info[0][2]
        else:
            return info[0][0]
    
    def _convert_image_color_for_input_model(self ,image : Image)-> Image:

        #copy image and transorm according to input model
        img = image.copy() 
        # Get input information from the YOLO model
        model_color = self.get_input_model_color()
        channels = self.get_input_model_channel()

        if channels != img.get_number_channel() and model_color != img.get_color_mode():
             img.convert_color_mode(model_color if channels > 1 else ColorMode.GRAY)
        return img     

    def run(
        self,
        image: Image,
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

        img = self._convert_image_color_for_input_model(image)
        # Measure preprocessing time
        start_preprocess = time.time()
        if preprocess is False:
            # Skip preprocessing entirely
            preprocessed_image = img
        else:
            # Use custom preprocessing if provided, otherwise use the native method
            if callable(preprocess):
                preprocessed_image = preprocess(img, self._input_model_height, self._input_model_width)
            else:
                preprocessed_image = self.preprocess(img, self._input_model_height, self._input_model_width)
        end_preprocess = time.time()
        preprocessing_time_ms = (end_preprocess - start_preprocess) * 1000

        # Measure inference time
        start_inference = time.time()
        raw_output = self._model.model_inference(preprocessed_image)
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
                postprocessed_output = postprocess(raw_output, img.get_height(), 
                                                   img.get_width(), confidence_thr,  
                                                   iou_threshold)
            else:
                postprocessed_output = self.postprocess(raw_output, img.get_height(),
                                                        img.get_width(), confidence_thr, 
                                                        iou_threshold)
        end_postprocess = time.time()
        postprocessing_time_ms = (end_postprocess - start_postprocess) * 1000
        
        yoloOut = YoloOutput(postprocessed_output, image=preprocessed_image)
        yoloOut.set_preprocessing_time(preprocessing_time_ms)
        yoloOut.set_inference_time(inference_time_ms)
        yoloOut.set_postprocessing_time(postprocessing_time_ms)
        yoloOut.set_image(image=image) # input image
        
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
    
    @staticmethod
    @abstractmethod
    def preprocess(image: Image,
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
    
    def postprocess(
        self,
        raw_output: np.ndarray,
        image_height:int, image_width:int, confidence_thr:float=0.5, 
        iou_threshold:float=0.5
    ) -> List[BBoxClassIdScore]:
        """
        Postprocess the raw output from the model to produce structured results.

        :param raw_output: Raw results from the model.
        :return: A YoloOutput object.
        """
        raise NotImplementedError("The postprocess method must be overridden in the derived class.")


def evaluate_image(
    predictions: YoloOutput, 
    ground_truth: List[BBoxClassId], 
    iou_thres: float
) -> Dict[int, Metric]:
    """
    Evaluate predictions against ground truth and compute per-class metrics.

    Args:
        predictions: YOLO model predictions.
        ground_truth: Ground truth bounding boxes and class IDs.
        iou_thres: IoU threshold for matching predictions to ground truth.

    Returns:
        Dict[int, Metric]: Dictionary mapping class_id to Metric object.
    """
    pred_boxes_with_info = predictions.get_boxes()
    # Organize predictions and ground truth by class
    preds_by_class = {}
    for bbox, class_id, score in pred_boxes_with_info:
        preds_by_class.setdefault(class_id, []).append((bbox, score))

    gt_by_class = {}
    for bbox, class_id in ground_truth:
        gt_by_class.setdefault(class_id, []).append(bbox)

    all_classes = set(preds_by_class.keys()) | set(gt_by_class.keys())
    metrics_per_class = {class_id: Metric() for class_id in all_classes}

    for class_id in all_classes:
        preds = preds_by_class.get(class_id, [])
        gts = gt_by_class.get(class_id, [])
        matched_gt = [False] * len(gts)
        tp = 0
        fp = 0

        # Sort predictions by score descending if available
        preds = sorted(preds, key=lambda x: x[1], reverse=True)

        for pred_bbox, _ in preds:
            best_iou = 0
            best_match_idx = -1
            for j, gt_bbox in enumerate(gts):
                if matched_gt[j]:
                    continue
                iou = pred_bbox.compute_iou(gt_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_match_idx = j
            if best_iou >= iou_thres and best_match_idx != -1:
                tp += 1
                matched_gt[best_match_idx] = True
            else:
                fp += 1

        fn = matched_gt.count(False)
        metric = metrics_per_class[class_id]
        metric.addTP(tp)
        metric.addFP(fp)
        metric.addFN(fn)

    return metrics_per_class


def evaluate(
    yolo: BaseYolo,
    dataloader: BaseDataloader,
    iou_threshold: float = 0.5,
    confidence_threshold: float = 0.5,
    output_streams: Optional[List[io.IOBase]] = None,
) -> Tuple[Dict[int, Metric], Metric]:
    """
    Evaluate a YOLO model on a dataset using a dataloader.

    Args:
        yolo (BaseYolo): The YOLO model to evaluate.
        dataloader (BaseDataloader): Dataloader yielding (image, label) pairs.
        iou_threshold (float, optional): IoU threshold for matching. Defaults to 0.5.
        confidence_threshold (float, optional): Confidence threshold for predictions. Defaults to 0.5.
        stdout (Optional[List[io.IOBase]]): List of output streams (e.g., [sys.stdout, file]).

    Returns:
        Tuple[Dict[int, Metric], Metric]: 
            - Dictionary mapping class IDs to their Metric objects.
            - Overall Metric object for all classes.
    """
    number_classes = yolo.get_number_class()
    metrics_per_class = {k: Metric() for k in range(number_classes)}
    image_count = 0
    for image, label in dataloader:
        yolo_out = yolo.run(
            image,
            confidence_thr=confidence_threshold,
            iou_threshold=iou_threshold
        )
        
        one_shot_metric = evaluate_image(
            yolo_out,
            label,
            iou_threshold
        )
        
        for class_id in metrics_per_class:
            if class_id in one_shot_metric:
                metrics_per_class[class_id].addTP(one_shot_metric[class_id].getTP())
                metrics_per_class[class_id].addFP(one_shot_metric[class_id].getFP())
                metrics_per_class[class_id].addFN(one_shot_metric[class_id].getFN())
        image_count += 1

    overall_metric = sum(metrics_per_class.values(), start=Metric())
    # Optional: print to all provided streams
    if output_streams:
        output_text = []
        output_text.append(f"{'Images processed:':<20} {image_count}\n")
        output_text.append(f"\nPer class:\n")
        output_text.append("+"*80 + "\n")
        for class_id in range(number_classes):
            if class_id == 0:
                output_text.append(f"     {'id':<12} | {'Precision':<12} {'':>12} {'Recall':<12} {'':>6}{'F1 Score':<12}\n")
                output_text.append("+"*80 + "\n")
            output_text.append(f"    {class_id:<12} | {metrics_per_class[class_id].getPrecision():.4f} {'':<12}| {metrics_per_class[class_id].getRecall():.4f} {'':<12}| {metrics_per_class[class_id].getF1Score():.4f}\n")
            output_text.append("-"*80 + "\n")
        output_text.append("\nOverall:\n")
        output_text.append(f"  - {'True Positives (TP):':<25} {overall_metric.getTP()}\n")
        output_text.append(f"  - {'False Positives (FP):':<25} {overall_metric.getFP()}\n")
        output_text.append(f"  - {'False Negatives (FN):':<25} {overall_metric.getFN()}\n")
        output_text.append(f"  - {'Overall Precision:':<25} {overall_metric.getPrecision():.4f}\n")
        output_text.append(f"  - {'Overall Recall:':<25} {overall_metric.getRecall():.4f}\n")
        output_text.append(f"  - {'Overall F1 Score:':<25} {overall_metric.getF1Score():.4f}\n")
        output_text.append("="*80 + "\n\n")
        output_str = ''.join(output_text)
        for stream in output_streams:
            if stream is not None and hasattr(stream, "write"):
                stream.write(output_str)
                if hasattr(stream, "flush"):
                    stream.flush()
    return metrics_per_class, overall_metric
