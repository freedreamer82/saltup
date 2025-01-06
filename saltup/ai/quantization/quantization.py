from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
import os

from saltup.utils.configure_logging import get_logger


class Quantize(ABC):
    """Abstract base class for model quantization pipelines."""
    
    def _validate_input_model(self, model_path: str) -> None:
        """Validate input tensor.
        
        Args:
            model_path: model path to quantize

        Raises:
            ValueError: If model is None or invalid
        """
        if model_path is None:
            raise ValueError("model path cannot be None")
        if not isinstance(model_path.endswith("keras", "h5" ,"onnx")):
            raise TypeError("model extension must be keras or onnx")

    def _validate_calibration_data(self, calibration_data_path: str) -> None:
        """Validate input data for calibration.
        
        Args:
            calibration_data_path: data calibration path

        Raises:
            ValueError: If calibration data extension is invalid
        """
        if calibration_data_path is None:
            raise ValueError("data calibration path cannot be None")
        
        for elt in os.listdir(calibration_data_path):
    
            if not isinstance(elt.split('.')[-1], ("png", "jpeg", "jpg", "raw")):
                raise TypeError("data extension must be valid")
    
    @abstractmethod
    def __call__(self):
        """Execute quantization pipeline.

        Must be implemented by subclasses to define specific quantization steps.

        Raises:
            NotImplementedError: If called directly on base class
        """
        get_logger(__name__).error(
            "Abstract quantization method called. Must be implemented in subclass."
        )
        raise NotImplementedError(
            "quantization method must be implemented in subclass.")