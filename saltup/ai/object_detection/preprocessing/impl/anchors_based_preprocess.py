import numpy as np
import cv2

from saltup.ai.object_detection.preprocessing.base_preproccesing import BasePreprocessing


class AnchorsBasedPreprocess(BasePreprocessing):

    def _validate_input(self, img: np.ndarray) -> None:
        """Validate input image.

        Args:
            img: Input image to validate

        Raises:
            ValueError: If image is None, invalid or not grayscale
            TypeError: If image is not numpy array
        """
        super()._validate_input(img)  # Call parent validation

        if len(img.shape) != 2:
            raise ValueError("Input must be grayscale (2D array)")

    def process(self, img: np.ndarray, target_shape: tuple) -> np.ndarray:
        """Preprocess grayscale image for model input.

        Applies the following steps:
        1. Input validation - checks image is valid grayscale array
        2. Square padding with 114 value (if target shape is square)
        3. Resize to target dimensions 
        4. Normalize to [0,1] range
        5. Add channel and batch dimensions

        Args:
            img: Grayscale input image
            target_shape: Model input size as (height, width)

        Returns:
            Preprocessed image tensor with shape (1, height, width, 1)
            - First dimension: batch size
            - Last dimension: single grayscale channel
            - Values normalized to [0,1] range

        Raises:
            ValueError: If input is not a valid grayscale image
        """
        # Validate input
        self._validate_input(img)

        # Get dimensions
        height, width = img.shape

        # Apply square padding if needed
        input_height, input_width = target_shape
        if input_height == input_width:
            max_dim = max(height, width)
            padded = np.full((max_dim, max_dim), 114, dtype=np.uint8)
            padded[0:height, 0:width] = img  # Place at top-left
            img = padded

        # Resize to target shape
        img = cv2.resize(img, (input_width, input_height),
                         interpolation=cv2.INTER_LINEAR)

        # Normalize and add dimensions
        img = img / 255.0  # Normalize to [0,1]
        img = np.expand_dims(img, axis=-1)  # Add channel dimension
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        return img.astype(np.float32)
