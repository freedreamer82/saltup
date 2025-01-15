import numpy as np

from saltup.ai.object_detection.dataset.bbox_utils import calculate_iou
from saltup.ai.object_detection.postprocessing import Postprocessing


class DamoPostprocess(Postprocessing):
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
        Postprocess the output from the damo-yolo model.

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
        class_scores = model_output[0].squeeze(0)
        bboxes = model_output[1].squeeze(0)
        rows = class_scores.shape[0]
        boxes = []

        x_factor = image_width / model_input_width
        y_factor = image_height / model_input_height

        for i in range(rows):
            # Extract the class scores and find the maximum score
            scores = class_scores[i]
            prob = np.max(scores)

            if prob >= confidence_thr:
                class_id = np.argmax(scores)
                label = classes_name[class_id]

                # Scale bounding box coordinates
                x1, y1, x2, y2 = bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3]
                x1 *= x_factor
                y1 *= y_factor
                x2 *= x_factor
                y2 *= y_factor

                boxes.append([x1, y1, x2, y2, label, prob])

        # Sort boxes by confidence
        boxes.sort(key=lambda x: x[5], reverse=True)

        # Apply non-max suppression
        result = []
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
        return self.postprocess(model_output, classes_name, model_input_height, model_input_width, image_height, image_width, confidence_thr, iou_threshold)



