from typing import Any, Dict, List ,Tuple

import numpy as np
from typing import Optional, Callable, Union
from saltup.ai.object_detection.utils.bbox  import BBox,BBoxFormat
<<<<<<< Updated upstream
from saltup.ai.object_detection.yolo  import YoloType
from saltup.ai import NeuralNetworkManager
from typing import List, Dict, Any, Union
import time

=======
from saltup.ai.object_detection.utils.metrics  import compute_ap, compute_ap_range
from saltup.ai.object_detection.yolo.yolo_type  import YoloType
from saltup.ai.nn_manager import NeuralNetworkManager
from typing import List, Dict, Any, Union
import time
from typing import Dict, List, Tuple
import cv2
from collections import defaultdict
>>>>>>> Stashed changes



class YoloOutput:
    """Class to represent the results of the YOLO model."""
    def __init__(
        self,
<<<<<<< Updated upstream
        boxes: List[BBox],
        scores: List[float],
        labels: List[str],
=======
        boxes: List[Tuple[BBox, int, float]],
>>>>>>> Stashed changes
        image: Optional[np.ndarray] = None,
    ):
        """
        Initialize YoloOutput with bounding boxes, scores, labels, and an optional image.

        Args:
            boxes: List of BBox objects representing the bounding boxes.
            scores: List of confidence scores.
            labels: List of predicted labels.
            image: Optional image associated with the results.
        """
        self._boxes = boxes  # List of BBox objects
        self._scores = scores  # List of confidence scores
        self._labels = labels  # List of predicted labels
        self._image = image  # Optional image associated with the results

        self._inference_time_ms = 0.0  
        self._preprocessing_time_ms = 0.0 
        self._postprocessing_time_ms = 0.0 

<<<<<<< Updated upstream
    def get_labels(self) -> List[str]:
        """Get the list of predicted labels."""
        return self._labels

    def set_labels(self, value: List[str]):
        """Set the list of predicted labels."""
        self._labels = value

    def get_boxes(self, format: str = "CENTER") -> List[BBox]:
=======
    def get_boxes(self, format: str = "CENTER") -> List[Tuple[BBox, int, float]]:
>>>>>>> Stashed changes
        """
        Get the list of bounding boxes in the specified format.

        Args:
            format: The desired format (CORNERS, CENTER, TOPLEFT).

        Returns:
            List of bounding box coordinates in the specified format.
        """
        return self._boxes

    def set_boxes(self, boxes: List[BBox]):
        """
        Set the list of bounding boxes.

        Args:
            boxes: List of BBox objects.
        """
        self._boxes = boxes

    def get_scores(self) -> List[float]:
        """Get the list of confidence scores."""
        return self._scores

    def set_scores(self, value: List[float]):
        """Set the list of confidence scores."""
        self._scores = value

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
            property_name: Name of the property ("boxes", "scores", "labels", "image",
                          "inference_time", "preprocessing_time", "postprocessing_time").

        Returns:
            The value of the property.

        Raises:
            AttributeError: If the property does not exist.
        """
        if property_name == "boxes":
            return self._boxes
        elif property_name == "scores":
            return self._scores
        elif property_name == "labels":
            return self._labels
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
            property_name: Name of the property ("boxes", "scores", "labels", "image",
                          "inference_time", "preprocessing_time", "postprocessing_time").
            value: The new value for the property.

        Raises:
            AttributeError: If the property does not exist.
        """
        if property_name == "boxes":
            self._boxes = value
        elif property_name == "scores":
            self._scores = value
        elif property_name == "labels":
            self._labels = value
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
            f"YoloOutput(boxes={self._boxes}, scores={self._scores}, labels={self._labels}, "
            f"image={self._image is not None}, inference_time={self._inference_time_ms} ms, "
            f"preprocessing_time={self._preprocessing_time_ms} ms, "
            f"postprocessing_time={self._postprocessing_time_ms} ms)"
        )


from enum import IntEnum, auto
from pathlib import Path
class ColorMode(IntEnum):
    RGB = auto()
    BGR = auto()
    GRAY = auto()
    
    
class BaseYolo(NeuralNetworkManager):
    """Base class for implementing a generic YOLO model."""
    def __init__(self, yolot: YoloType, model_path: str):
        super().__init__()  # Initialize NeuralNetworkManager
        self.model_path = model_path
<<<<<<< Updated upstream
        self.model = self.load_model(model_path)  # Load model using inherited method
=======
        self.number_class = number_class
        self.model, self.model_input_shape, self.model_output_shape = self.load_model(model_path)  # Load model using inherited method
        self.img_input_height  = self.model_input_shape[1]
        self.img_input_width  = self.model_input_shape[2]
>>>>>>> Stashed changes
        self.yolotype = yolot

    def getYoloType(self) -> YoloType:
        return self.yolotype
    
<<<<<<< Updated upstream
    def _load_model(self, model_path: str) -> Any:
        """
        Load the YOLO model from the given path
         Example: return torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        """
        raise NotImplementedError("Model loading must be implemented.")

=======
    @staticmethod
    def load_anchors(anchors_path:str) -> np.ndarray:
        anchors_data = np.loadtxt(anchors_path, delimiter=",")
        anchors_list = [row for row in anchors_data]
        anchors_array = np.array(anchors_list).reshape(-1, 2)
        
        return anchors_array
    
    @staticmethod
    def load_image(image_path: str, color_mode: ColorMode = ColorMode.BGR) -> np.ndarray:
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
        # Verify file exists
        if not Path(image_path).is_file():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Load image in BGR (OpenCV default)
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Failed to load image: {image_path}")

        # Convert to desired color mode
        try:
            if color_mode == ColorMode.RGB:
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif color_mode == ColorMode.GRAY:
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return image  # BGR
        except cv2.error as e:
            raise ValueError(f"Error converting image color mode: {e}")
    
    def get_number_image_channel(self) -> int:
        return self.model_input_shape[-1]
    
>>>>>>> Stashed changes
    def run(
        self,
        image: np.ndarray,
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
<<<<<<< Updated upstream
                preprocessed_image = preprocess(image)
            else:
                preprocessed_image = self.preprocess(image)
=======
                preprocessed_image = preprocess(image, self.img_input_height, self.img_input_width)
            else:
                preprocessed_image = self.preprocess(image, self.img_input_height, self.img_input_width)
>>>>>>> Stashed changes
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
                postprocessed_output = postprocess(raw_output)
            else:
                postprocessed_output = self.postprocess(raw_output)
        end_postprocess = time.time()
        postprocessing_time_ms = (end_postprocess - start_postprocess) * 1000

        # Set execution times in the YoloOutput object
        if isinstance(postprocessed_output, YoloOutput):
            postprocessed_output.set_preprocessing_time(preprocessing_time_ms)
            postprocessed_output.set_inference_time(inference_time_ms)
            postprocessed_output.set_postprocessing_time(postprocessing_time_ms)

        return postprocessed_output

    @staticmethod
    def evaluate( predictions: YoloOutput,ground_truth: BBox,  threshold_iou :float ) -> Dict[str, float]:
        """
         Compute evaluation metrics (e.g., precision, recall, mAP)
         Example:
         metrics = {"precision": 0.95, "recall": 0.90, "mAP": 0.85}
         return metrics
        """
        raise NotImplementedError("Evaluation must be implemented.")


    def preprocess(self, image: np.array) -> Any:
        """
        Preprocess the image before model inference.

        :param image: Input image to preprocess.
        :return: Preprocessed image.
        """
        raise NotImplementedError("The preprocess method must be overridden in the derived class.")

<<<<<<< Updated upstream
    def postprocess(self, raw_output: np.array) -> YoloOutput:
=======
    def postprocess(self, raw_output: np.ndarray,
                    image_height:int, image_width:int, confidence_thr:float=0.5, 
                            iou_threshold:float=0.5) -> List[Tuple[BBox, int, float]]:
>>>>>>> Stashed changes
        """
        Postprocess the raw output from the model to produce structured results.

        :param raw_output: Raw results from the model.
        :return: A YoloOutput object.
        """
        raise NotImplementedError("The postprocess method must be overridden in the derived class.")