import numpy as np
import cv2
from typing import Optional, Union, Callable, Dict, Any, List, Tuple


from saltup.ai.object_detection.utils.bbox import BBox, BBoxFormat, nms
from saltup.utils.data.image.image_utils import  Image, ColorMode ,ImageFormat
from saltup.ai.object_detection.yolo.yolo import BaseYolo, YoloType

class YoloDamo(BaseYolo):
    """
    A class that extends BaseYolo to handle damo YOLO models.
    """
    def __init__(
        self, 
        yolot: YoloType, 
        model_path: str,  
        number_class: int
        ):
        """
        Initialize the Damo YOLO model.

        :param yolot: Type of YOLO model (e.g., YoloType.YOLOv3, YoloType.YOLOv4).
        :param model_path: Path to the model file.
        :param number_class: The number of classes in the model.
        """
        super().__init__(yolot, model_path, number_class)  # Initialize the BaseYolo class
    
    def get_input_info(self) -> Tuple[tuple, ColorMode, ImageFormat]:
        input_shape = self.model_input_shape[1:]  # Rimuove il batch size
        return (
            input_shape,  # Shape: (3, 480, 640)
            ColorMode.RGB,
            ImageFormat.CHW
        )
         
    @staticmethod
    def preprocess(
                   image: Image,
                   target_height:int, 
                   target_width:int,        
                   normalize_method: callable = lambda x: x.astype(np.float32) / 255.0,
                   apply_padding: bool = True,
                   **kwargs: Any
                   ) -> np.ndarray:
        """Process input image according to DAMO model requirements.

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
        num_channel = image.get_number_channel()
        
        # Validate input format
        if num_channel not in [1, 3]:
            raise ValueError(
                "Only 1 or 3 channels are supported for multi-channel images")

        # Resize to target dimensions
        image_data = cv2.resize(raw_image, (target_width, target_height))
        
        # Convert to CHW format
        image_data = np.transpose(image_data, (2, 0, 1))

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
        Postprocess the output from the damo-yolo model.
        Args:
            raw_output (np.ndarray): output matrix of the model
            image_height (int): input height of the current inferenced image
            image_width (int): input width of the current inferenced image
            confidence_thr (float): the threshold of the confidence score
            iou_threshold (float): the threshold of the Intersection over Union for NMS

        Returns:
            List[Tuple[BBox, int, float]]: List of predicted bounding boxes in the image with their respective score and class_id.
        """
        class_scores = raw_output[0].squeeze(0)
        bboxes = raw_output[1].squeeze(0)
        rows = class_scores.shape[0]
        boxes = []
        probs = []
        class_ids = []
        
        x_factor = image_width / self.input_model_width
        y_factor = image_height / self.input_model_height

        for i in range(rows):
            # Extract the class scores and find the maximum score
            scores = class_scores[i]
            prob = np.max(scores)

            if prob >= confidence_thr:
                class_id = np.argmax(scores)

                # Scale bounding box coordinates
                x1, y1, x2, y2 = bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3]
                x1 *= x_factor
                y1 *= y_factor
                x2 *= x_factor
                y2 *= y_factor
                
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
 