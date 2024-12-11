from typing import Tuple, Dict, Optional, Union, Any
import numpy as np
import cv2

from saltup.ai.object_detection.preprocessing.base_preproccesing import BasePreprocessing


class UltraliticsPreprocess(BasePreprocessing):
    """Preprocessing pipeline for Ultralytic models with letterboxing."""
    
    def __init__(
        self,
        target_shape: Tuple[int, int] = (640, 640),
        auto: bool = False,
        scale_fill: bool = False,
        scale_up: bool = True,
        center: bool = False,
        stride: int = 32
    ):
        """Initialize preprocessing parameters.
        
        Args:
            target_shape (Tuple[int, int]): Target size (height, width) for the resized image.
            auto (bool): If True, use minimum rectangle to resize. If False, use new_shape directly.
            scale_fill (bool): If True, stretch the image to new_shape without padding.
            scale_up (bool): If True, allow scaling up. If False, only scale down.
            center (bool): If True, center the placed image. If False, place image in top-left corner.
            stride (int): Stride of the model (e.g., 32 for YOLOv5).
        """
        self.target_shape = target_shape
        self.auto = auto
        self.scale_fill = scale_fill
        self.scale_up = scale_up
        self.center = center
        self.stride = stride

    def letterbox(self, img: np.ndarray, shape_override: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, Dict]:
        """Resize image to target shape and adds padding if needed while preserving aspect ratio.

        Args:
            img: Input image in BGR format
            shape_override: Optional target shape override

        Returns:
            Tuple containing:
                - Processed image array
                - Dict with transformation parameters (ratio, padding)
        
        Example:
            >>> processor = UltraliticsPreprocess(target_shape=(416, 416))
            >>> result = processor.letterbox(my_image)
        """
        shape = img.shape[:2]  # Current shape [height, width]
        new_shape = shape_override or self.target_shape
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Calculate scale ratio
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scale_up:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Calculate padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)
        elif self.scale_fill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])

        if self.center:
            dw /= 2  # divide padding into 2 sides
            dh /= 2

        # Resize and pad image
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
            
        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114)  # Grey padding
        )

        return img

    def process(
        self,
        img: np.ndarray,
        **kwargs: Any
    ) -> Union[np.ndarray, Dict]:
        """Execute preprocessing pipeline with optional parameter overrides.
        
        Args:
            img: Input image in BGR format
            **kwargs: Override any class parameter (target_shape, auto, scale_fill, etc.)

        Returns:
            Processed image tensor
        """
        self._validate_input(img)
        
        # Override params temporarily if needed
        original_params = None
        if kwargs:
            # Store original parameters
            original_params = self.__dict__.copy()
            
            # Apply any provided overrides
            self.__dict__.update((k, v) for k, v in kwargs.items() 
                                if k in original_params)

        try:
            # Process image
            processed_img = self.letterbox(img)
            
            # Convert to model input format
            image_data = np.array(processed_img) / 255.0    # Normalize
            image_data = np.transpose(image_data, (2, 0, 1))    # HWC to CHW format (channel first)
            image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
            
            return image_data
            
        finally:
            # Restore original parameters
            if original_params:
                self.__dict__.update(original_params)