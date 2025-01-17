from abc import ABC, abstractmethod
from pathlib import Path
from enum import IntEnum, auto
import cv2
import numpy as np


class ColorMode(IntEnum):
    RGB = auto()
    BGR = auto()
    GRAY = auto()


class BaseDatasetLoader(ABC):
    """Base interface for dataset loaders.

    Abstract base class that defines the interface for dataset loaders.
    Provides basic functionality for loading and iterating over image-label pairs.
    """

    @abstractmethod
    def __iter__(self):
        """Returns iterator over image and label paths."""
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        """Returns total number of samples in dataset."""
        raise NotImplementedError

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
