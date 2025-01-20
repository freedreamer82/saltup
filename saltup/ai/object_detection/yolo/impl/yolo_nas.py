import numpy as np
import cv2
from typing import Optional, Union, Callable, Dict, Any, List, Tuple

from saltup.utils.data.image.image_utils import ColorMode, ImageFormat, Image
from saltup.ai.object_detection.utils.bbox import BBox, BBoxFormat, nms
from saltup.ai.object_detection.yolo.yolo import BaseYolo, YoloType

class YoloNas(BaseYolo):
    """
    A class that extends BaseYolo to handle supergrad YOLO models.
    """
    def __init__(
        self, 
        yolot: YoloType, 
        model_path: str,  
        number_class: int
        ):
        """
        Initialize the supergrad YOLO model.

        :param yolot: Type of YOLO model (e.g., YoloType.YOLOv3, YoloType.YOLOv4).
        :param model_path: Path to the model file.
        :param number_class: The number of classes in the model.
        """
        super().__init__(yolot, model_path, number_class)  # Initialize the BaseYolo class
    
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
        super()._validate_input_preprocessing_image(img)

        if len(img.shape) not in [2, 3]:
            raise ValueError(
                "Input must be either 2D (single channel) or 3D (multiple channels) array")

        if len(img.shape) == 3 and img.shape[2] not in [1, 3]:
            raise ValueError(
                "Only 1 or 3 channels are supported for multi-channel images")
         

    def preprocess(self,
                   image: Image,
                   target_height:int, 
                   target_width:int,        
                   normalize_method: callable = lambda x: x.astype(np.float32) / 255.0,
                   apply_padding: bool = True,
                   **kwargs: Any
                   ) -> np.ndarray:
        """Process input image according to supergrad model requirements.

        Args:
            img: Input image in BGR format (H, W, C)
            target_shape: Desired output shape as (width, height)

        Returns:
            np.ndarray: Processed image tensor with:
                - NCHW format (batch, channels, height, width)
                - Normalized pixel values
                - Target spatial dimensions
                - float32 precision

        Raises:
            ValueError: For invalid or empty inputs
        """
        raw_image = image.get_data()
        self._validate_input(raw_image)
        
        h, w = raw_image.shape[:2]

        # Calculate size preserving aspect ratio
        aspect_ratio = min(target_width / w, target_height / h)
        new_width = int(w * aspect_ratio)
        new_height = int(h * aspect_ratio)

        # Resize with high-quality algorithm
        resized_image = cv2.resize(raw_image, (new_width, new_height), 
                                 interpolation=cv2.INTER_AREA)
        
        if apply_padding:
            image_data = Image.pad_image(resized_image, target_height, target_width, 
                               image_format=ImageFormat.HWC)
        else:
            image_data = resized_image
        # Apply normalization
        image_data = normalize_method(image_data)
        
        image_data = np.transpose(image_data, (2, 0, 1))    # HWC to CHW format (channel first)
        # Add batch dimension
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
        
        return image_data

    def postprocess(self, 
                    raw_output: np.ndarray,
                    image_height:int, 
                    image_width:int, 
                    confidence_thr:float=0.5, 
                    iou_threshold:float=0.5) -> List[Tuple[BBox, int, float]]:
        """
        Postprocess the output from the supergrad-yolo model.
        Args:
            raw_output (np.ndarray): output matrix of the model
            image_height (int): input height of the current inferenced image
            image_width (int): input width of the current inferenced image
            confidence_thr (float): the threshold of the confidence score
            iou_threshold (float): the threshold of the Intersection over Union for NMS

        Returns:
            List[Tuple[BBox, int, float]]: List of predicted bounding boxes in the image with their respective score and class_id.
        """
        class_score = raw_output[1].squeeze(0)
        bboxes = raw_output[0].squeeze(0)
        rows = class_score.shape[0]
        boxes = []
        probs = []
        class_ids = []
        x_factor = image_width / self.img_input_width
        y_factor = image_height / self.img_input_height
        
        for i in range(rows):
            # Extract the class scores from the current row
            classes_scores = class_score[i]
            # Find the maximum score among the class scores
            prob = np.max(classes_scores)
            
            # If the maximum score is above the confidence threshold
            if prob >= confidence_thr:
                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)
                
                # Extract the bounding box coordinates from the current row
                x1, y1, x2, y2 = bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3]
                
                x1 = np.maximum(0, np.minimum(image_width, x1))  # x1
                y1 = np.maximum(0, np.minimum(image_height, y1))  # y1
                x2 = np.maximum(0, np.minimum(image_width, x2))  # x2
                y2 = np.maximum(0, np.minimum(image_height, y2))  # y2
                
                box_object = BBox((x1, y1, x2, y2), format=BBoxFormat.CORNERS,
                              img_width=image_width, img_height=image_height)
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
 