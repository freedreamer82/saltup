import numpy as np

from saltup.ai.object_detection.dataset.bbox_utils import calculate_iou
from saltup.ai.object_detection.postprocessing import Postprocessing


class UltralyticsPostprocess(Postprocessing):
    """
    Class to postprocess the output of a YOLO model.

    Steps:
    1. Extract class scores and bounding boxes from the model output.
    2. Scale bounding box coordinates based on the input and original image dimensions.
    3. Filter out boxes with confidence scores below the specified threshold.
    4. Apply non-maximum suppression (NMS) to remove overlapping boxes based on IoU.
    5. Return the list of final bounding boxes with labels and confidence scores.
    """

    def postprocess(self,
                    model_output:np.ndarray,
                    classes_name:list[str]=['red', 'blue', 'green', 'yellow'], 
                    model_input_height:int=480, 
                    model_input_width:int=640, 
                    image_height:int=480, 
                    image_width:int=640,
                    confidence_thr:float=0.5, 
                    iou_threshold:float=0.5) -> list[list]:       
        """
        Postprocess the output from the ultralytics-yolo model.

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
        model_output = model_output[0].astype(float)
        model_output = model_output.transpose()

        boxes = []
        for row in model_output:
            prob = row[4:].max()
            if prob < confidence_thr:  # Use the confidence threshold set in the class
                continue
            class_id = row[4:].argmax()
            label = classes_name[class_id]
            xc, yc, w, h = row[:4]
            x1 = (xc - w/2) / model_input_width * image_width
            y1 = (yc - h/2) / model_input_height * image_height
            x2 = (xc + w/2) / model_input_width * image_width
            y2 = (yc + h/2) / model_input_height * image_height

            boxes.append([x1, y1, x2, y2, label, prob])

        boxes.sort(key=lambda x: x[5], reverse=True)  # Sort by confidence
        result = []
        # Apply the non-max supression
        while len(boxes) > 0 and len(result) < len(boxes):
            result.append(boxes[0])
            boxes = [box for box in boxes if calculate_iou(box[:4], boxes[0][:4]) < iou_threshold]
        return result

    def __call__(self,
                model_output:np.ndarray,
                classes_name:list[str]=['red', 'blue', 'green', 'yellow'], 
                model_input_height:int=480, 
                model_input_width:int=640, 
                image_height:int=480, 
                image_width:int=640,
                confidence_thr:float=0.5, 
                iou_threshold:float=0.5) -> list[list]:          
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
        return self.postprocess(model_output, classes_name, model_input_height, model_input_width,image_height, image_width,  confidence_thr, iou_threshold)




