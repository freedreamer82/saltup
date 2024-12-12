import numpy as np
import cv2

from saltup.ai.object_detection.preprocessing import Preprocessing


class DamoPreprocessing(Preprocessing):
    """DAMO object detection models preprocessing pipeline.

    Implements image preprocessing steps for DAMO object detection models:
    - Color space conversion (BGR to RGB)
    - Image resizing to target dimensions
    - Optional normalization
    - Channel reordering (HWC to CHW)
    - Optional padding if needed
    """

    def pad_image(self, image: np.ndarray, target_size: tuple) -> np.ndarray:
        """Add padding if image dimensions are smaller than target size.

        Args:
            image: Input image in CHW format
            target_size: Target dimensions as (width, height)

        Returns:
            np.ndarray: Image padded to target size if needed, original image otherwise
        """
        c, h, w = image.shape
        target_h, target_w = target_size

        # Return original image if no padding needed
        if h >= target_h and w >= target_w:
            return image

        # Add padding only if necessary
        padded_img = np.zeros((c, target_h, target_w), dtype=np.float32)
        padded_img[:, :h, :w] = image
        return padded_img

    def __call__(
        self,
        img: np.ndarray,
        target_shape: tuple,
        normalize_method: callable = None
    ) -> np.ndarray:
        """Process input image according to DAMO model requirements.

        Args:
            img: Input image in BGR format (H, W, C)
            target_shape: Desired output shape as (width, height)
            normalize_method: Optional custom normalization function.
                            If None, applies standard [0,1] normalization

        Returns:
            np.ndarray: Processed image tensor in CHW format, 
                       normalized and padded if necessary

        Raises:
            ValueError: If input image is None or empty
        """
        self._validate_input(img)

        # Convert color space
        image_data = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to target dimensions
        image_data = cv2.resize(image_data, target_shape)

        # Apply normalization
        if normalize_method:
            image_data = normalize_method(image_data)
        else:
            image_data = image_data.astype(np.float32) / 255.0

        # Convert to CHW format
        image_data = np.transpose(image_data, (2, 0, 1))

        # Add padding if needed
        image_data = self.pad_image(image_data, target_shape)

        # Add batch dimension
        image_data = np.expand_dims(image_data, axis=0)

        return image_data
