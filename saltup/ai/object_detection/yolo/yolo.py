from typing import Any, Dict, List
from enum import IntEnum
import numpy as np
from typing import Optional, Callable, Union
from saltup.ai.object_detection.dataset.bbox_utils import BBox

class YoloOutput:
    """Class to represent the results of the YOLO model."""
    def __init__(self, boxes: List[Dict[str, Any]], scores: List[float], labels: List[str]):
        self._boxes = boxes  # List of dictionaries with bounding box coordinates
        self._scores = scores  # List of confidence scores
        self._labels = labels  # List of predicted labels

    def getLabels(self) -> List[str]:
        return self._labels

    def setLabels(self, value: List[str]):
        self._labels = value

    def getProperty(self, property_name: str) -> Any:
        if property_name == "boxes":
            return self._boxes
        elif property_name == "scores":
            return self._scores
        elif property_name == "labels":
            return self._labels
        else:
            raise AttributeError(f"Property '{property_name}' does not exist.")

    def setProperty(self, property_name: str, value: Any):
        if property_name == "boxes":
            self._boxes = value
        elif property_name == "scores":
            self._scores = value
        elif property_name == "labels":
            self._labels = value
        else:
            raise AttributeError(f"Property '{property_name}' does not exist.")

    def __repr__(self):
        return f"YoloOutput(boxes={self._boxes}, scores={self._scores}, labels={self._labels})"


class YoloType(IntEnum):
    ANCHORS_BASED = 0
    ULTRALITICS = 1
    SUPERGRAD = 2
    DAMO = 3

class BaseYolo:
    """Base class for implementing a generic YOLO model."""
    def __init__(self, yolot : YoloType, model_path: str):
        super().__init__()
        self.model_path = model_path
        self.model = self._load_model(model_path)
        self.yolotype = yolot

    def getYoloType(self) -> YoloType:
        return self.yolotype
    
    def _load_model(self, model_path: str) -> Any:
        """
        Load the YOLO model from the given path
         Example: return torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        """
        raise NotImplementedError("Model loading must be implemented.")

    def run(
        self,
        image: np.array,
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
        :return: A YoloOutput object containing the results.
        """
        # Handle preprocessing
        if preprocess is False:
            # Skip preprocessing entirely
            preprocessed_image = image
        else:
            # Use custom preprocessing if provided, otherwise use the native method
            if callable(preprocess):
                preprocessed_image = preprocess(image)
            else:
                preprocessed_image = self.preprocess(image)

        # Perform model inference
        raw_output = self._model_inference(preprocessed_image)

        # Handle postprocessing
        if postprocess is False:
            # Skip postprocessing entirely
            postprocessed_output = raw_output
        else:
            # Use custom postprocessing if provided, otherwise use the native method
            if callable(postprocess):
                postprocessed_output = postprocess(raw_output)
            else:
                postprocessed_output = self.postprocess(raw_output)

        return postprocessed_output

    @staticmethod
    def evaluate( predictions: YoloOutput,ground_truth: BBox) -> Dict[str, float]:
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

    def postprocess(self, raw_output: np.array) -> YoloOutput:
        """
        Postprocess the raw output from the model to produce structured results.

        :param raw_output: Raw results from the model.
        :return: A YoloOutput object.
        """
        raise NotImplementedError("The postprocess method must be overridden in the derived class.")

    def _model_inference(self, preprocessed_image: Any) -> Any:
        """
        Simulate YOLO model inference. This method should be implemented
        in the derived class to connect to the actual model.

        :param preprocessed_image: Preprocessed input for the model.
        :return: Raw output from the model.
        """
        raise NotImplementedError("The _model_inference method must be implemented in the derived class.")
