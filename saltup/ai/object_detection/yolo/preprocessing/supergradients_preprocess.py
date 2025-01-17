from enum import IntEnum, auto
import numpy as np
import cv2

from saltup.ai.object_detection.preprocessing import Preprocessing


class SupergradPreprocessType(IntEnum):
    """Supergradient model preprocessing types.
    
    Available pipelines:
        BASE: Standard pipeline with image normalization
        QAT: Quantization-aware training pipeline without normalization
    """
    BASE = 0
    QAT = 1


class SupergradPreprocess(Preprocessing):
    """Preprocessing pipeline for Supergradient object detection models.
    
    Supports two preprocessing pipelines:
    - Standard (BASE): Includes resizing, padding, and normalization
    - Quantization-aware training (QAT): Similar to BASE but preserves original pixel values
    
    Both pipelines preserve aspect ratio during resizing and use consistent padding
    for spatial dimensions.
    """
    
    class ImagePosition(IntEnum):
        """Valid positions for image placement within container.
        
        Defines anchor points:
            TOP_LEFT: Place image at (0,0) coordinates
            CENTER: Place image at container center
        """
        TOP_LEFT = auto()  # Align to origin
        CENTER = auto()    # Align to center
        
    def __init__(
        self,
        preprocess_type: SupergradPreprocessType = SupergradPreprocessType.BASE, 
        normalize_method: callable = None
    ):
        """Initialize preprocessy

        Args:
            normalize_method: Optional custom normalization function 
                           (defaults to standard [0,1] normalization)
        """
        super().__init__()
        self.preprocess_type = preprocess_type
        self.normalize_method = normalize_method if normalize_method else super().standard_normalize
    
    def resize_image_and_black_container_rgb(
        self, 
        image: np.ndarray, 
        final_width: int, 
        final_height: int, 
        img_position: ImagePosition
    ) -> np.ndarray:
        """Resize image and place it on a grey background.

        Maintains aspect ratio during resize and provides consistent padding.
        Uses grey padding (value 114) which is optimal for neural networks.

        Args:
            image: Input image in BGR format
            final_width: Target width in pixels
            final_height: Target height in pixels
            img_position: Anchor point for image placement (TOP_LEFT or CENTER)

        Returns:
            np.ndarray: RGB image with dimensions (final_height, final_width, 3)

        Raises:
            ValueError: If img_position is invalid
        """
        # Convert to RGB colorspace
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]

        # Calculate size preserving aspect ratio
        aspect_ratio = min(final_width / w, final_height / h)
        new_width = int(w * aspect_ratio)
        new_height = int(h * aspect_ratio)

        # Resize with high-quality algorithm
        resized_image = cv2.resize(image_rgb, (new_width, new_height), 
                                 interpolation=cv2.INTER_AREA)

        # Create padded container with standardized grey
        grey_image = 114 * np.ones((final_height, final_width, 3), 
                                 dtype=np.uint8)
        
        # Compute placement offsets
        x_offset = (final_width - new_width) // 2
        y_offset = (final_height - new_height) // 2

        # Place image according to position
        if img_position == self.ImagePosition.TOP_LEFT:
            grey_image[0:new_height, 0:new_width] = resized_image
        elif img_position == self.ImagePosition.CENTER: 
            grey_image[y_offset:y_offset+new_height, 
                      x_offset:x_offset+new_width] = resized_image
        else:
            raise ValueError(f"Invalid image position: {img_position}")

        return grey_image

    def preprocess(self, img: np.ndarray, target_shape: tuple) -> np.ndarray:
        """Apply standard preprocessing pipeline.

        Steps:
        1. Converts BGR to RGB
        2. Resize and pad image
        3. Apply normalization. By default normalized to [0,1], otherwise you can provide a custom normalization method.
        4. Convert to NCHW format

        Args:
            img: Input image in BGR format
            target_shape: Desired (height, width)

        Returns:
            np.ndarray: Normalized tensor in NCHW format
        """
        image_data = self.resize_image_and_black_container_rgb(
            img,
            final_height=target_shape[0],
            final_width=target_shape[1],
            img_position=self.ImagePosition.TOP_LEFT
        )
        # Apply normalization
        image_data = self.normalize_method(image_data)

        # Reorder channels (HWC -> NCHW)
        image_data = np.transpose(image_data, (2, 0, 1))
        
        # Add batch dimension
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
        return image_data

    def preprocess_qat(self, img: np.ndarray, target_shape: tuple) -> np.ndarray:
        """Execute QAT preprocessing pipeline.
        
        Similar to standard pipeline but preserves original pixel values
        for quantization-aware training requirements.

        Args:
            img: Source image in BGR format
            target_shape: Desired dimensions as (height, width)
        
        Returns:
            np.ndarray: Image tensor preserving original [0,255] range
        """
        return self.resize_image_and_black_container_rgb(
            img,
            final_height=target_shape[0],
            final_width=target_shape[1],
            img_position=self.ImagePosition.TOP_LEFT
        )

    def __call__(
        self, 
        img: np.ndarray, 
        target_shape: tuple
    ) -> np.ndarray:
        """Execute preprocessing pipeline.

        Args:
            img: Input BGR image
            target_shape: Target dimensions as (height, width)

        Returns:
            np.ndarray: Processed image tensor ready for model input

        Raises:
            ValueError: For invalid input image
        """
        self._validate_input(img)
        
        if self.preprocess_type == SupergradPreprocessType.QAT:
            return self.preprocess_qat(img, target_shape)
        return self.preprocess(img, target_shape)
