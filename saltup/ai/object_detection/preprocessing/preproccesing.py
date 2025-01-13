from abc import ABC, abstractmethod
import numpy as np

from saltup.utils.configure_logging import get_logger


class Preprocessing(ABC):
    """Abstract base class for image preprocessing pipelines.
    
    Provides common validation and normalization functionality for 
    implementing custom preprocessing pipelines.
    """
        
    def _validate_input(self, img: np.ndarray) -> None:
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
    def __call__(self):
        """Execute the preprocessing pipeline.

        This abstract method must be implemented by subclasses to define
        their specific preprocessing steps and transformations.

        Raises:
            NotImplementedError: If called directly on base class
        """
        get_logger(__name__).error(
            "Abstract preprocessing method called. Must be implemented in subclass."
        )
        raise NotImplementedError(
            "Preprocessing method must be implemented in subclass.")
        
    @staticmethod
    def standard_normalize(img: np.ndarray) -> np.ndarray:
        """
        Standard image normalization to [0,1] range.

        Args:
            img: Input image array

        Returns:
            np.ndarray: Normalized image as float32 in range [0,1]
        """
        return img.astype(np.float32) / 255.0