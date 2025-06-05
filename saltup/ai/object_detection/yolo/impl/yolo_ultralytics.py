import numpy as np
import cv2
from typing import Optional, Union, Dict, Any, List, Tuple

from saltup.ai.object_detection.utils.bbox import BBox, BBoxFormat, nms
from saltup.utils.data.image.image_utils import  Image, ColorMode ,ImageFormat
from saltup.ai.object_detection.yolo.yolo import BaseYolo, YoloType
from saltup.ai.nn_model import NeuralNetworkModel

class YoloUltralytics(BaseYolo):
    """
    A class that extends BaseYolo to handle Ultralytics YOLO models.
    """
    def __init__(
        self,
        model:NeuralNetworkModel,  
        number_class: int
    ):
        """
        Initialize the Ultralytics YOLO model.

        :param yolot: Type of YOLO model (e.g., YoloType.YOLOv3, YoloType.YOLOv4).
        :param model_path: Path to the model file.
        :param number_class: The number of classes in the model.
        """
        super().__init__(YoloType.ULTRALYTICS, model, number_class)  # Initialize the BaseYolo class
    
    def get_input_info(self) -> Tuple[tuple, ColorMode, ImageFormat]:
        input_shape = self._model_input_shape[1:]  # Rimuove il batch size
        return (
            input_shape,  # Shape: (3, 480, 640)
            ColorMode.RGB,
            ImageFormat.CHW
        )
        
    @staticmethod    
    def letterbox(
            img: np.ndarray,
            target_shape: Tuple[int, int], 
            shape_override: Optional[Tuple[int, int]] = None,
            auto: bool = False,
            scale_fill: bool = False,
            scale_up: bool = True,
            center: bool = False,
            stride: int = 32
        ) -> Tuple[np.ndarray, Dict]:
        """Resize image to target shape and adds padding if needed while preserving aspect ratio.

        Args:
            img: Input image in BGR format
            target_shape: Desired output shape as (width, height)
            shape_override: Optional target shape override
            auto: If True, use minimum rectangle to resize. If False, use new_shape directly.
            scale_fill: If True, stretch the image to new_shape without padding.
            scale_up: If True, allow scaling up. If False, only scale down.
            center: If True, center the placed image. If False, place image in top-left corner.
            stride: Stride of the model (e.g., 32 for YOLOv5).

        Returns:
            Tuple containing:
                - Processed image array
                - Dict with transformation parameters (ratio, padding)
        """
        shape = img.shape[:2]  # Current shape [height, width]
        new_shape = shape_override or target_shape
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Calculate scale ratio
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scale_up:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Calculate padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)
        elif scale_fill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])

        if center:
            dw /= 2  # divide padding into 2 sides
            dh /= 2

        # Resize and pad image
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
            
        top, bottom = int(round(dh - 0.1)) if center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if center else 0, int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114)  # Grey padding
        )

        return img
    
    @staticmethod
    def preprocess(
        image: Image,
        target_height:int, 
        target_width:int,        
        normalize_method: callable = lambda x: x.astype(np.float32) / 255.0,
        apply_padding: bool = True,
        **kwargs: Any
    ) -> np.ndarray:
        """
        Execute complete preprocessing pipeline.
        
        Pipeline steps:
        1. Input validation
        2. Parameter override handling
        3. Letterboxing and resizing
        4. Pixel normalization
        5. Channel reordering (HWC → CHW)
        6. Batch dimension addition
        
        Args:
            img: Input BGR image
            **kwargs: Optional parameter overrides for this call

        Returns:
            np.ndarray: Processed tensor ready for model input
        """
        if isinstance(image, Image):
            raw_img = image.get_data() # get in hwc
            num_channel = image.get_number_channel()
        elif isinstance(image, np.ndarray):
            raw_img = image
            num_channel = 1 if len(image.shape) < 3 else image.shape[2]
        else:
            raise TypeError(f"Invalid type {type(image)} for image: should be 'np.ndarray' or 'saltup.Image'.")
        
        # Validate input format
        if num_channel not in [1, 3]:
            raise ValueError("Only 1 or 3 channels are supported for multi-channel images")
        
        # Override params temporarily if needed
        # original_params = None
        # if kwargs:
        #     # Store original parameters
        #     original_params = self.__dict__.copy()
            
        #     # Apply any provided overrides
        #     self.__dict__.update((k, v) for k, v in kwargs.items() 
        #                         if k in original_params)
        
        try:
            # Process image
            processed_img = YoloUltralytics.letterbox(raw_img, (target_height, target_width))#, **kwargs)
            
            # Normalize
            image_data = normalize_method(processed_img)
            
            # Convert to model input format
            image_data = np.transpose(image_data, (2, 0, 1))    # HWC to CHW format (channel first)
            image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
            
            return image_data
        except Exception as e:
            raise e
        
        # finally:
        #     # Restore original parameters
        #     if original_params:
        #         self.__dict__.update(original_params)

    def postprocess(
        self, 
        raw_output: np.ndarray,
        image_height:int, 
        image_width:int, 
        confidence_thr:float=0.5, 
        iou_threshold:float=0.5
    ) -> List[Tuple[BBox, int, float]]:
        """
        Postprocess the output from the Ultralytics-yolo model.

        Args:
            raw_output (np.ndarray): output matrix of the model
            image_height (int): input height of the current inferenced image
            image_width (int): input width of the current inferenced image
            confidence_thr (float): the threshold of the confidence score
            iou_threshold (float): the threshold of the Intersection over Union for NMS

        Returns:
            List[Tuple[BBox, int, float]]: List of predicted bounding boxes in the image with their respective score and class_id.
        """
        raw_output = raw_output[0].astype(float)
        raw_output = raw_output.transpose()

        boxes = []
        probs = []
        class_ids = []
        x_factor = image_width / self._input_model_width
        y_factor = image_height / self._input_model_height
        
        for row in raw_output:
            prob = row[4:].max()
            if prob < confidence_thr:  # Use the confidence threshold set in the class
                continue
            class_id = row[4:].argmax()
            xc, yc, w, h = row[:4]
            x1 = (xc - w/2) * x_factor
            y1 = (yc - h/2) * y_factor
            x2 = (xc + w/2) * x_factor
            y2 = (yc + h/2) * y_factor
            
            x1 = np.maximum(0, np.minimum(image_width, x1))  # x1
            y1 = np.maximum(0, np.minimum(image_height, y1))  # y1
            x2 = np.maximum(0, np.minimum(image_width, x2))  # x2
            y2 = np.maximum(0, np.minimum(image_height, y2))  # y2
            
            box_object = BBox(
                coordinates=(x1, y1, x2, y2), 
                fmt=BBoxFormat.CORNERS_ABSOLUTE,
                img_height=image_height, 
                img_width=image_width
            )
            boxes.append(box_object)
            probs.append(prob)
            class_ids.append(class_id)
                        
        selected_boxes = nms(boxes, probs, iou_threshold)
        selected_indices = [boxes.index(box) for box in selected_boxes]
        selected_scores = [probs[indices] for indices in selected_indices]
        selected_class_ids = [class_ids[indices] for indices in selected_indices]
        
        result = []
        for i, c in reversed(list(enumerate(selected_class_ids))):
            box = selected_boxes[i]
            score = selected_scores[i]
            result.append((box, int(c), score))

        return result
 