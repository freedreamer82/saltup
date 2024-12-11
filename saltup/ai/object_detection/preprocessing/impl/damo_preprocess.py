import numpy as np
import cv2

from saltup.ai.object_detection.preprocessing.base_preproccesing import BasePreprocessing


class DamoPreprocessing(BasePreprocessing):
   """Preprocessing pipeline for DAMO object detection models.
   
   Handles:
   - Color space conversion (BGR to RGB) 
   - Image resizing
   """

   def process(self, img: np.ndarray, target_shape: tuple) -> np.ndarray:
       """Convert color space and resize image.

       Args:
           img: Input image in BGR format
           target_shape: Desired output shape (width, height)

       Returns:
           Resized RGB image tensor matching target shape
       """
       self._validate_input(img)
       
       image_data = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
       image_data = cv2.resize(image_data, target_shape)  # Resize to target dimensions
       
       return image_data