from abc import ABC, abstractmethod
import numpy as np

from saltup.utils.data.image.image_utils import Image, ColorMode, ImageFormat


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
    def load_image(image_path: str, color_mode: ColorMode = ColorMode.BGR, image_format: ImageFormat = ImageFormat.HWC) -> np.ndarray:
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
        return Image(image_path, color_mode, image_format)
