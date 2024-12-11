from abc import ABC, abstractmethod
import numpy as np

from saltup.utils.configure_logging import get_logger


class BasePreprocessing(ABC):
    """Abstract base class for image preprocessing pipelines."""
    
    def _validate_input(self, img: np.ndarray) -> None:
        """Validate input image.
        
        Args:
            img: Input image to validate

        Raises:
            ValueError: If image is None or invalid
            TypeError: If image is not numpy array
        """
        if img is None:
            raise ValueError("Input image cannot be None")
        if not isinstance(img, np.ndarray):
            raise TypeError("Input must be numpy array")

    @abstractmethod
    def process(self):
        """Execute preprocessing pipeline.

        Must be implemented by subclasses to define specific preprocessing steps.

        Raises:
            NotImplementedError: If called directly on base class
        """
        get_logger(__name__).error(
            "Abstract preprocessing method called. Must be implemented in subclass."
        )
        raise NotImplementedError(
            "Preprocessing method must be implemented in subclass.")
