from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf

from saltup.utils.configure_logging import get_logger


class Postprocessing(ABC):
    """Abstract base class for model output postprocessing pipelines."""
    
    def _validate_input(self, model_output: np.ndarray) -> None:
        """Validate input tensor.
        
        Args:
            model_output: model output to validate

        Raises:
            ValueError: If model output is None or invalid
        """
        if model_output is None:
            raise ValueError("model output cannot be None")
        if not isinstance(model_output, (np.ndarray, tf.Tensor)):
            raise TypeError("Input must be numpy or tensorflow array")

    @abstractmethod
    def __call__(self):
        """Execute postprocessing pipeline.

        Must be implemented by subclasses to define specific postprocessing steps.

        Raises:
            NotImplementedError: If called directly on base class
        """
        get_logger(__name__).error(
            "Abstract postprocessing method called. Must be implemented in subclass."
        )
        raise NotImplementedError(
            "Postprocessing method must be implemented in subclass.")