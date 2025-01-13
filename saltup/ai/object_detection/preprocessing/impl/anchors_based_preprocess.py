import numpy as np
import cv2

from saltup.ai.object_detection.preprocessing.preproccesing import Preprocessing

class AnchorsBasedPreprocess(Preprocessing):
    """
    Preprocessing pipeline for anchor-based object detection models.
    
    Provides comprehensive image preprocessing including:
    - Channel structure preservation
    - Dynamic resizing and padding
    - Configurable normalization
    - Input format validation
    - Batch dimension handling
    """
    
    def __init__(
        self,
        normalize_method: callable = None, 
        apply_padding: bool = True
    ):
        """
        Initialize the preprocessor with custom settings.
        
        Args:
            normalize_method: Custom normalization function (defaults to standard [0,1] normalization)
            apply_padding: Enable square padding for square target shapes (default: True)
        """
        super().__init__()
        
        # Fix syntax error in the original
        self.normalize_method = normalize_method if normalize_method else super().standard_normalize
        self.apply_padding = apply_padding        

    def _validate_input(self, img: np.ndarray) -> None:
        """Validate input image format and channel structure.
        
        Extends base validation with specific checks for channel dimensions
        and supported formats.

        Args:
            img: Input image to validate (single or multiple channels)

        Raises:
            ValueError: For invalid dimensions or unsupported channel counts
            TypeError: For non-numpy array inputs (from parent class)
        """
        super()._validate_input(img)
        
        if len(img.shape) not in [2, 3]:
            raise ValueError("Input must be either 2D (single channel) or 3D (multiple channels) array")
            
        if len(img.shape) == 3 and img.shape[2] not in [1, 3]:
            raise ValueError("Only 1 or 3 channels are supported for multi-channel images")

    def __call__(
        self, 
        img: np.ndarray, 
        target_shape: tuple,
    ) -> np.ndarray:
        """Process image for anchor-based object detection.

        Processing pipeline:
        1. Input validation and channel verification
        2. Channel dimension handling
        3. Square padding (optional, for square targets)
        4. Target size resizing
        5. Pixel normalization
        6. Batch dimension preparation

        Args:
            img: Source image (single or multi-channel)
            target_shape: Required output size as (height, width)

        Returns:
            np.ndarray: Processed image tensor with shape (1, height, width, channels)
            - First dimension: batch size
            - Last dimension: number of channels matches input (1 for single channel, 3 for RGB)
            - Normalized values

        Raises:
            ValueError: For invalid image formats
            TypeError: For incorrect input types
        """
        # Validate input format
        self._validate_input(img)
        
        # Determine channel configuration
        is_single_channel = len(img.shape) == 2
        
        # Extract dimensions
        if is_single_channel:
            height, width = img.shape
            channels = 1
        else:
            height, width, channels = img.shape
            
        input_height, input_width = target_shape

        # Handle square padding if needed
        if input_height == input_width and self.apply_padding:
            max_dim = max(height, width)
            if is_single_channel:
                padded = np.full((max_dim, max_dim), 114, dtype=np.uint8)
                padded[0:height, 0:width] = img
            else:
                padded = np.full((max_dim, max_dim, channels), 114, dtype=np.uint8)
                padded[0:height, 0:width, :] = img
            img = padded

        # Scale to target dimensions
        img = cv2.resize(img, (input_width, input_height),
                       interpolation=cv2.INTER_LINEAR)

        # Normalize pixel values
        img = self.normalize_method(img)

        # Ensure proper channel dimension
        if is_single_channel:
            img = np.expand_dims(img, axis=-1)

        # Prepare batch dimension
        img = np.expand_dims(img, axis=0)

        return img.astype(np.float32)