from enum import IntEnum, auto
import numpy as np
import cv2

from saltup.ai.object_detection.preprocessing.base_preproccesing import BasePreprocessing


class SupergradPreprocessType(IntEnum):
    """Supergradient model preprocessing types.
    
    Defines preprocessing pipelines:
        BASE: Standard pipeline with image normalization
        QAT: Quantization-aware training without normalization
    """
    BASE = 0
    QAT = 1


class SupergradPreprocess(BasePreprocessing):
    """Preprocessing pipeline for Supergradient object detection models.
    
    Implements two preprocessing pipelines for different model types:
    - Standard pipeline (BASE): Includes resizing, padding, and normalization
    - Quantization-aware training (QAT): Similar to BASE but preserves original pixel values
    
    Each pipeline maintains aspect ratio during resizing and uses consistent padding
    for maintaining spatial dimensions.
    """
    
    class ImagePosition(IntEnum):
        """Valid positions for image placement within container.
        
        Defines anchor points:
            TOP_LEFT: Place image at (0,0) coordinates
            CENTER: Place image at container center
        """
        TOP_LEFT = auto()  # Align to origin
        CENTER = auto()    # Align to center
    
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
            image: Source image in BGR format
            final_width: Desired container width in pixels
            final_height: Desired container height in pixels
            img_position: Anchor point for image placement (TOP_LEFT or CENTER)

        Returns:
            np.ndarray: RGB image with dimensions (final_height, final_width, 3)

        Raises:
            ValueError: If img_position is invalid
        """
        # Convert color space
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]

        # Preserve aspect ratio
        aspect_ratio = min(final_width / w, final_height / h)
        new_width = int(w * aspect_ratio)
        new_height = int(h * aspect_ratio)

        # High-quality downsampling
        resized_image = cv2.resize(image_rgb, (new_width, new_height), 
                                 interpolation=cv2.INTER_AREA)

        # Initialize container with neural network optimized grey value
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
        """Execute standard preprocessing pipeline.
        
        Converts BGR to RGB, resizes with padding, and normalizes to [0,1] range.
        Produces tensor in NCHW format suitable for neural network input.

        Args:
            img: Source image in BGR format
            target_shape: Desired dimensions as (height, width)
        
        Returns:
            np.ndarray: Normalized tensor in NCHW format, range [0,1]
        """
        img = self.resize_image_and_black_container_rgb(
            img,
            final_height=target_shape[0],
            final_width=target_shape[1],
            img_position=self.ImagePosition.TOP_LEFT
        )
        img = np.array(img) / 255.0  # Normalize
        img = np.transpose(img, (2, 0, 1))  # HWC to CHW format (channel first)
        return np.expand_dims(img, axis=0).astype(np.float32)

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

    def process(
        self, 
        img: np.ndarray, 
        target_shape: tuple, 
        type: SupergradPreprocessType = SupergradPreprocessType.BASE
    ) -> np.ndarray:
        """Select and apply appropriate preprocessing pipeline.
        
        Main entry point that routes to either standard or QAT pipeline
        based on specified type.

        Args:
            img: Source image in BGR format
            target_shape: Desired dimensions as (height, width)
            type: Pipeline selection (BASE or QAT)
            
        Returns:
            np.ndarray: Processed image tensor in format matching selected pipeline
        """
        self._validate_input(img)
        
        if type == SupergradPreprocessType.QAT:
            return self.preprocess_qat(img, target_shape)
        return self.preprocess(img, target_shape)