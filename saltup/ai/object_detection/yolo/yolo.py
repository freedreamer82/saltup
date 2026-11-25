import numpy as np
from typing import (
    Optional, Callable, Union,
    Any, Dict, List, Mapping, Tuple,
)
import time
from collections import defaultdict
from abc import ABC, abstractmethod
from enum import Enum
from types import MappingProxyType
import sys, io
from tqdm import tqdm

from saltup.ai.object_detection.utils.bbox import BBoxFormat, BBox, BBoxClassId
from saltup.ai.object_detection.utils.metrics import compute_ap, compute_map_50_95, compute_ap_for_threshold
from saltup.ai.object_detection.yolo.yolo_type import YoloType
from saltup.ai.object_detection.utils.metrics import Metric
from saltup.utils.data.image.image_utils import ColorMode, ImageFormat, Image
from saltup.ai.nn_model import NeuralNetworkModel
from saltup.ai.base_dataformat.base_dataloader import BaseDataloader
from saltup.ai.hooks import HookHandle, HookRegistry, merge_hook_results


class YoloHookEvent(str, Enum):
    """Points of `BaseYolo.run` where registered hooks are called.

    Every event calls `hook(model, inputs, outputs, extras)` with the input and
    output of the stage that just ended (see `YoloHook`).
    """
    POSTPROCESS_END = "postprocess_end"
    """After the post-process: inputs = raw model output, outputs = final boxes."""


YoloHook = Callable[["BaseYolo", Any, Any, Mapping[str, Any]], Optional[Dict[str, Any]]]
"""Signature of a YOLO hook: `hook(model, inputs, outputs, extras) -> dict | None`.

- model: the `BaseYolo` running the stage.
- inputs, outputs: the stage input and output (see `YoloHookEvent`). Read only.
- extras: internal values the stage exposed, read only. For POSTPROCESS_END they are
  the dict a post-process returns as `(boxes, extras)`; empty for one returning `boxes`.
- return: a dict of named values to add to `YoloOutput.get_metadata()`, or None.
"""


class YoloOutput:
    """Class to represent the results of the YOLO model."""
    def __init__(
        self,
        boxes: List[Tuple[BBox, int, float]],
        image: Optional[Image] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize YoloOutput with bounding boxes and an optional image.

        Args:
            boxes: List of tuples (BBox, class_id, score) representing the bounding boxes, class IDs, and confidence scores.
            image: Optional image associated with the results.
            metadata: Named values returned by the hooks of the run (empty dict if None).
        """
        self._boxes = boxes  # List of BBox objects
        self._image = image  # Optional image associated with the results
        self._metadata: Dict[str, Any] = metadata if metadata is not None else {}

        self._inference_time_ms = 0.0  
        self._preprocessing_time_ms = 0.0 
        self._postprocessing_time_ms = 0.0 

    tprocessing_time_ms = 0.0 

    def get_boxes(self) -> List[Tuple[BBox, int, float]]:
        """
        Get the list of bounding boxes.

        Returns:
            List of tuples containing:
            - BBox object.
            - Class ID (int).
            - Confidence score (float).
        """
        return self._boxes
    
    def set_boxes(self, boxes: List[Tuple[BBox, int, float]]):
        """
        Set the list of bounding boxes.

        Args:
            boxes: List of tuples containing (BBox, class_id, score).
        """
        self._boxes = boxes
        
    def get_metadata(self) -> Dict[str, Any]:
        """Named values returned by the hooks of the run; empty when no hook is registered."""
        return self._metadata

    def set_metadata(self, metadata: Dict[str, Any]):
        """Set the metadata dictionary."""
        self._metadata = metadata

    def get_image(self) -> Optional[Image]:
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
            property_name: Name of the property ("boxes", "image", "metadata",
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
        elif property_name == "metadata":
            return self._metadata
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
            property_name: Name of the property ("boxes", "image", "metadata",
                          "inference_time", "preprocessing_time", "postprocessing_time").
            value: The new value for the property.

        Raises:
            AttributeError: If the property does not exist.
        """
        if property_name == "boxes":
            self._boxes = value
        elif property_name == "image":
            self._image = value
        elif property_name == "metadata":
            self._metadata = value
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
        self._hooks = HookRegistry()
        self._model_input_shape = self._model.get_input_shape()
        self._model_output_shape = self._model.get_output_shape()
        
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

    def get_model(self) -> NeuralNetworkModel:
        """Get the underlying neural network model."""
        return self._model
    
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
                            If False, skips postprocessing entirely (and no POSTPROCESS_END hook runs).
                            A post-process returns `boxes`, or `(boxes, extras)` where `extras` is a dict
                            of internal values passed to the POSTPROCESS_END hooks.
        :return: A YoloOutput object containing the results, the hooks metadata and timing information.
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
        postprocessed_output: Any
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

        metadata: Dict[str, Any] = {}
        if postprocess is not False:
            postprocessed_output, extras = _split_postprocess_result(postprocessed_output)
            # Hooks run outside the measured post-processing time
            metadata = self._run_hooks(
                YoloHookEvent.POSTPROCESS_END, raw_output, postprocessed_output, extras
            )

        yoloOut = YoloOutput(postprocessed_output, image=image, metadata=metadata)  # input image
        yoloOut.set_preprocessing_time(preprocessing_time_ms)
        yoloOut.set_inference_time(inference_time_ms)
        yoloOut.set_postprocessing_time(postprocessing_time_ms)
        
        return yoloOut
    
    def register_hook(
        self,
        event: Union[YoloHookEvent, str],
        hook: YoloHook,
        *,
        prepend: bool = False,
    ) -> HookHandle:
        """
        Attach `hook` to `event`; it is called on every `run()` that reaches the event.

        Hooks of the same event run in registration order; `prepend=True` puts this one
        first. Each hook may return a dict of named values: the dicts of all hooks are
        merged into `YoloOutput.get_metadata()`. Hooks observe, they cannot change the
        boxes. See `saltup.ai.hooks` for how hooks differ from training callbacks.

        Example, exposing a value that a custom post-process returns in its extras:

            def objectness_hook(model, raw_output, boxes, extras):
                return {"objectness": extras["objectness"]}

            with yolo.register_hook(YoloHookEvent.POSTPROCESS_END, objectness_hook):
                out = yolo.run(image, postprocess=my_postprocess)
            out.get_metadata()["objectness"]

        :param event: A `YoloHookEvent` or its string value, e.g. "postprocess_end".
        :param hook: Callable with the `YoloHook` signature.
        :param prepend: If True, call this hook before the ones already registered.
        :return: A `HookHandle`; `handle.remove()` detaches the hook.
        :raises ValueError: If `event` is not a `YoloHookEvent`.
        """
        return self._hooks.register(YoloHookEvent(event), hook, prepend=prepend)

    def _run_hooks(
        self,
        event: YoloHookEvent,
        inputs: Any,
        outputs: Any,
        extras: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Call the hooks of `event` in order and merge the dicts they return."""
        hooks = self._hooks.get(event)
        if not hooks:
            return {}
        read_only_extras = MappingProxyType(extras)
        return merge_hook_results([hook(self, inputs, outputs, read_only_extras) for hook in hooks])

    @staticmethod
    @abstractmethod
    def preprocess(image: Image,
                   target_height:int, 
                   target_width:int,        
                   normalize_method: Callable = lambda x: x.astype(np.float32) / 255.0,
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
    ) -> List[Tuple[BBox, int, float]]:
        """
        Postprocess the raw output from the model to produce structured results.

        :param raw_output: Raw results from the model.
        :return: A YoloOutput object.
        """
        raise NotImplementedError("The postprocess method must be overridden in the derived class.")


def _split_postprocess_result(result: Any) -> Tuple[Any, Dict[str, Any]]:
    """Split a post-process result into `(boxes, extras)`.

    A post-process returns either `boxes` (a list) or the tuple `(boxes, extras)` with
    `extras` a dict of internal values for the hooks. A list of boxes is never a
    2-tuple ending in a dict, so the two forms cannot be confused.
    """
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
        return result[0], result[1]
    return result, {}


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
    mAP: bool = False,
    preprocess: Optional[Union[Callable, bool]] = None,
    postprocess: Optional[Union[Callable, bool]] = None,
) -> Tuple[Dict[int, Metric], Metric, float]:
    """
    Evaluate a YOLO model on a dataset using a dataloader.

    Args:
        yolo (BaseYolo): The YOLO model to evaluate.
        dataloader (BaseDataloader): Dataloader yielding (image, label) pairs.
        iou_threshold (float, optional): IoU threshold for matching. Defaults to 0.5.
        confidence_threshold (float, optional): Confidence threshold for predictions. Defaults to 0.5.
        stdout (Optional[List[io.IOBase]]): List of output streams (e.g., [sys.stdout, file]).
        mAP (bool, optional): Also compute mAP@50-95. Defaults to False.
        preprocess, postprocess (optional): Forwarded to `yolo.run()`: a custom callable,
            False to skip the stage, or None (default) for the model's native method.

    Returns:
        Tuple[Dict[int, Metric], Metric, float]: 
            - Dictionary mapping class IDs to their Metric objects.
            - Overall Metric object for all classes.
            - Overall mAP@50-95 score.
    """
    number_classes = yolo.get_number_class()
    metrics_per_class = {k: Metric() for k in range(number_classes)}
    image_count = 0
    
    # Collect all predictions and ground truth for global mAP@50-95 calculation
    all_pred_boxes = []
    all_gt_boxes = []

    start_time = time.time()
    pbar = tqdm(dataloader, desc="Inference", dynamic_ncols=True)
    for path, image, label in pbar:
        yolo_out = yolo.run(
            image,
            confidence_thr=confidence_threshold,
            iou_threshold=iou_threshold,
            preprocess=preprocess,
            postprocess=postprocess,
        )
        
        one_shot_metric = evaluate_image(
            yolo_out,
            label,
            iou_threshold
        )
        
        # Collect predictions and ground truth for global mAP@50-95 calculation
        pred_boxes = yolo_out.get_boxes()
        gt_boxes = label
        # Convert to required format for compute_map_50_95
        pred_boxes_flat = [(bbox, score) for bbox, class_id, score in pred_boxes]
        gt_boxes_flat = [bbox for bbox, class_id in gt_boxes]
        all_pred_boxes.extend(pred_boxes_flat)
        all_gt_boxes.extend(gt_boxes_flat)

        for class_id in metrics_per_class:
            if class_id in one_shot_metric:
                metrics_per_class[class_id].addTP(one_shot_metric[class_id].getTP())
                metrics_per_class[class_id].addFP(one_shot_metric[class_id].getFP())
                metrics_per_class[class_id].addFN(one_shot_metric[class_id].getFN())
        image_count += 1
        
        overall_metric = sum(metrics_per_class.values(), start=Metric())
        dict_tqdm = {
            "precision": overall_metric.getPrecision(),
            "recall": overall_metric.getRecall(),
            "f1": overall_metric.getF1Score()
        }

        pbar.set_postfix(dict_tqdm)
        
    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_image = (total_time / image_count * 1000) if image_count > 0 else 0.0

    overall_metric = sum(metrics_per_class.values(), start=Metric())
    # Calculate mAP@50-95 across all images at once
    overall_map_50_95 = 0.0
    if mAP:
        overall_map_50_95 = compute_map_50_95(all_gt_boxes, all_pred_boxes) if all_gt_boxes and all_pred_boxes else 0.0
    # Optional: print to all provided streams
    if output_streams:
        output_text = []

        output_text.append(f"Model type: {yolo.getYoloType().name}\n")
        print(f"Total execution time: {total_time:.2f} seconds")
        print(f"Average time per image: {avg_time_per_image:.2f} ms")

        num_params = yolo.get_model().get_num_parameters()
        if num_params >= 1_000_000:
            num_params_str = f"{num_params:,} ({num_params/1_000_000:.1f}M)"
        elif num_params >= 1_000:
            num_params_str = f"{num_params:,} ({num_params/1_000:.1f}k)"
        else:
            num_params_str = f"{num_params}"
        output_text.append(f"Number of model parameters : {num_params_str}\n")
        model_size_bytes = yolo.get_model().get_model_size_bytes()
        if model_size_bytes is not None:
            output_text.append(f"Size of model : {model_size_bytes / (1024 * 1024):.2f} MB\n")
        else:
            output_text.append("Size of model : N/A\n")

        output_text.append(f"{'IoU threshold:':<20} {iou_threshold}\n")
        output_text.append(f"{'Confidence threshold:':<20} {confidence_threshold}\n")
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
        if mAP:
            output_text.append(f"  - {'Overall mAP@50-95 :':<25} {overall_map_50_95:.4f}\n")
        output_text.append("="*80 + "\n\n")
        output_str = ''.join(output_text)
        for stream in output_streams:
            if stream is not None and hasattr(stream, "write"):
                stream.write(output_str)
                if hasattr(stream, "flush"):
                    stream.flush()
    return metrics_per_class, overall_metric, overall_map_50_95
