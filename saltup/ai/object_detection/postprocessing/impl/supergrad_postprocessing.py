import numpy as np

from saltup.ai.object_detection.dataset.bbox_utils import calculate_iou
from saltup.ai.object_detection.postprocessing import Postprocessing


class DamoPostprocessing(Postprocessing):
    """
    Class to postprocess the output of a supergrad-yolo model.

    Steps:
    1. Extract class scores and bounding boxes from the model output.
    2. Scale bounding box coordinates based on the input and original image dimensions.
    3. Filter out boxes with confidence scores below the specified threshold.
    4. Apply non-maximum suppression (NMS) to remove overlapping boxes based on IoU.
    5. Return the list of final bounding boxes with labels and confidence scores.
    """

    def postprocess(self,
                    model_output:np.ndarray,
                    classes_name:list[str], 
                    model_input_height:int, 
                    model_input_width:int, 
                    image_height:int, 
                    image_width:int,
                    confidence_thr:float, 
                    iou_threshold:float) -> list[list]:       
        """
        Postprocess the output from the YOLO model.

        Args:
            model_output (np.ndarray): Output matrix from the model.
            image_height (int): Height of the input image.
            image_width (int): Width of the input image.
            classes_name (list[str]): List of class names.
            model_input_height (int): Height of the model input.
            model_input_width (int): Width of the model input.
            confidence_thr (float): Confidence threshold to filter predictions.
            iou_threshold (float): IoU threshold for non-max suppression.

        Returns:
            list[list]: List of predicted bounding boxes in the image.
        """
        class_score = model_output[1].squeeze(0)
        bboxes = model_output[0].squeeze(0)
        rows = class_score.shape[0]
        boxes = []
        x_factor = image_width / model_input_width
        y_factor = image_height / model_input_height
        for i in range(rows):
            # Extract the class scores from the current row
            classes_scores = class_score[i]
            # Find the maximum score among the class scores
            prob = np.max(classes_scores)
            
            # If the maximum score is above the confidence threshold
            if prob >= confidence_thr:
                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)
                label = classes_name[class_id]
                # Extract the bounding box coordinates from the current row
                x1, y1, x2, y2 = bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3]

            boxes.append([x1, y1, x2, y2, label, prob])

        boxes.sort(key=lambda x: x[5], reverse=True)  # Sort by confidence
        result = []
        # Apply the non-max supression
        while len(boxes) > 0:
            result.append(boxes[0])
            boxes = [box for box in boxes if calculate_iou(box, boxes[0]) < iou_threshold]
        return result

    def __call__(self,
                model_output:np.ndarray,
                classes_name:list[str], 
                model_input_height:int, 
                model_input_width:int, 
                image_height:int, 
                image_width:int,
                confidence_thr:float, 
                iou_threshold:float) -> list[list]:          
        """
        Directly invoking the postprocess method.

        Args:
            model_output (np.ndarray): Output matrix from the model.
            image_height (int): Height of the input image.
            image_width (int): Width of the input image.
            classes_name (list[str]): List of class names.
            model_input_height (int): Height of the model input.
            model_input_width (int): Width of the model input.
            confidence_thr (float): Confidence threshold for predictions.
            iou_threshold (float): IoU threshold for non-max suppression.

        Returns:
            list[list]: List of predicted bounding boxes in the image.
        """
        return self.postprocess(model_output, image_height, image_width, classes_name, model_input_height, model_input_width, confidence_thr, iou_threshold)

