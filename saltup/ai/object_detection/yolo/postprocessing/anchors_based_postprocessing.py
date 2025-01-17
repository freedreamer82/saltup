import numpy as np
from saltup.ai.object_detection.utils.bbox import BBox, BBoxFormat, convert_matrix_boxes
from saltup.ai.object_detection.yolo.postprocessing.postprocessing import Postprocessing
from saltup.ai.object_detection.yolo.yolo import YoloOutput
from typing import List, Tuple

def decode(yolo_output, anchors, num_classes, input_shape, calc_loss=False):
    """
    Decode YOLO output derived from loss function logic.
    Returns box coordinates, dimensions, confidence, and class probabilities.
    """
    stride = input_shape[0] / yolo_output.shape[1]
    grid_h = int(input_shape[0] // stride)
    grid_w = int(input_shape[1] // stride)
    num_anchors = len(anchors)

    yolo_output = yolo_output.reshape(-1, grid_h, grid_w, num_anchors, 5 + num_classes).astype(np.float32)

    box_xy = 1 / (1 + np.exp(-yolo_output[..., 0:2]))
    box_wh = np.exp(yolo_output[..., 2:4])
    box_confidence = 1 / (1 + np.exp(-yolo_output[..., 4:5]))
    box_class_probs = 1 / (1 + np.exp(-yolo_output[..., 5:]))

    grid_y = np.arange(grid_h).reshape(-1, 1, 1, 1)
    grid_x = np.arange(grid_w).reshape(1, -1, 1, 1)
    grid_y = np.tile(grid_y, (1, grid_w, 1, 1))
    grid_x = np.tile(grid_x, (grid_h, 1, 1, 1))
    grid = np.concatenate([grid_x, grid_y], axis=-1)
    grid = np.tile(grid, (1, 1, num_anchors, 1))

    box_xy = (box_xy + grid) / np.array([grid_w, grid_h], dtype=np.float32)

    anchors_tensor = np.array(anchors, dtype=np.float32).reshape(1, 1, 1, num_anchors, 2)
    box_wh = box_wh * anchors_tensor

    if calc_loss:
        return grid, yolo_output, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs

def filter_boxes(my_boxes, boxes, box_confidence, box_class_probs, threshold=0.5):
    """
    Filters YOLO boxes based on object and class confidence.

    Args:
        my_boxes (numpy.ndarray): containing the coordinates of the boxes in the original image dimensions.
        boxes (numpy.ndarray): containing the coordinates of the boxes.
        box_confidence (numpy.ndarray): containing the object confidence scores.
        box_class_probs (numpy.ndarray): containing the class probabilities.
        threshold (float): threshold for box score to be considered as a detection.

    Returns:
        boxes (numpy.ndarray): containing the coordinates of the filtered boxes in corners format.
        scores (numpy.ndarray): containing the scores of the filtered boxes.
        classes (numpy.ndarray): containing the class IDs of the filtered boxes.
        my_boxes (numpy.ndarray): containing the coordinates of the filtered boxes in centroids format.
    """
    box_scores = box_confidence * box_class_probs
    box_classes = np.argmax(box_scores, axis=-1)  # Shape: (N, ...)
    box_class_scores = np.max(box_scores, axis=-1)  # Shape: (N, ...)

    # Create prediction mask
    prediction_mask = box_class_scores >= threshold
    
    # Apply boolean mask to filter boxes
    boxes = boxes[prediction_mask]
    my_boxes = my_boxes[prediction_mask]
    scores = box_class_scores[prediction_mask]
    classes = box_classes[prediction_mask]

    return boxes, scores, classes, my_boxes


from typing import List

def nms(bboxes: List[BBox], scores: List[float], iou_threshold: float, max_boxes: int = None) -> List[BBox]:
    """
    Perform Non-Maximum Suppression (NMS) on a list of BBox objects.

    Args:
        bboxes: List of BBox objects.
        scores: List of confidence scores corresponding to the bounding boxes.
        iou_threshold: IoU threshold for suppression.
        max_boxes: Maximum number of boxes to keep (optional).

    Returns:
        List of BBox objects after applying NMS.
    """
    
    # Pair bounding boxes with their scores and sort them by scores in descending order
    boxes_with_scores = sorted(zip(bboxes, scores), key=lambda x: x[1], reverse=True)

    selected_bboxes = []
    while boxes_with_scores:
        # Select the box with the highest score
        current_box, current_score = boxes_with_scores.pop(0)
        selected_bboxes.append(current_box)

        # Remove boxes that have IoU greater than the threshold with the current box
        remaining_boxes = []
        for other_box, other_score in boxes_with_scores:
            iou = current_box.compute_iou(other_box)
            if iou <= iou_threshold:
                remaining_boxes.append((other_box, other_score))

        boxes_with_scores = remaining_boxes

        # Stop if we've reached the max number of boxes
        if max_boxes is not None and len(selected_bboxes) >= max_boxes:
            break

    return selected_bboxes

def tiny_yolo_v2_nms(
    yolo_outputs, image_shape, max_boxes=30, score_threshold=0.5, iou_threshold=0.3, classes_ids=[0]
):
    """
    Applies non-max suppression to the output of Tiny YOLO v2 model.

    Args:
        yolo_outputs (list): Output of the Tiny YOLO v2 model.
        image_shape (tuple): Shape of the input image (height, width).
        max_boxes (int): Maximum number of boxes to be selected by non-max suppression.
        score_threshold (float): Threshold for box score to be considered as a detection.
        iou_threshold (float): Threshold for intersection over union to be considered as a duplicate detection.
        classes_ids (list): List of class IDs to perform non-max suppression on.

    Returns:
        List of selected boxes, scores, and classes.
    """
    box_xy, box_wh, box_confidence, box_class_probs = yolo_outputs
    boxes, my_boxes = convert_matrix_boxes(box_xy, box_wh)
    boxes, scores, classes, my_boxes = filter_boxes(
        my_boxes, boxes, box_confidence, box_class_probs, threshold=score_threshold
    )

    # Scale boxes to image dimensions
    height, width = image_shape
    image_dims = np.array([height, width, height, width], dtype=np.float32)
    boxes = boxes * image_dims  # Convert to original image dimensions
    
    # Wrap boxes into BBox objects
    bboxes = [BBox(box.tolist(), format=BBoxFormat.CORNERS, img_width=width, img_height=height) for box in boxes]

    total_boxes, total_scores, total_classes = [], [], []

    for c in classes_ids:
        # Filter by class ID
        mask = (classes == c)
        class_boxes = [bboxes[i] for i in np.where(mask)[0]]
        class_scores = scores[mask]

        if not class_boxes:
            continue

        # Apply NMS
        selected_boxes = nms(class_boxes, class_scores, iou_threshold, max_boxes)
        
        selected_indices = [class_boxes.index(box) for box in selected_boxes]

        # Collect results
        total_boxes.extend(selected_boxes)
        total_scores.extend(class_scores[selected_indices])
        total_classes.extend([c] * len(selected_boxes))

    # Convert results to numpy arrays
    if total_boxes:
        s_boxes = np.array([box.get_coordinates() for box in total_boxes], dtype=np.float32)
        s_scores = np.array(total_scores, dtype=np.float32)
        s_classes = np.array(total_classes, dtype=np.int32)
        s_my_boxes = np.array([box.to_yolo() for box in total_boxes], dtype=np.float32)
    else:
        s_boxes = np.empty((0, 4), dtype=np.float32)
        s_scores = np.empty((0,), dtype=np.float32)
        s_classes = np.empty((0,), dtype=np.int32)
        s_my_boxes = np.empty((0, 4), dtype=np.float32)

    return s_boxes, s_scores, s_classes, s_my_boxes

class AnchorsBasedPostprocess(Postprocessing):
    
    """
    Class to postprocess the output of a AnchorsBased-yolo model.

    Steps:
    1. Extract class scores and bounding boxes from the model output through the decode function.
    2. Filter out boxes with confidence scores below the specified threshold using the filter_boxes function.
    3. Apply non-maximum suppression (NMS) to remove overlapping boxes based on IoU.
    4. Return the list of final bounding boxes with labels and confidence scores.
    """
    
    def postprocess(self,
                    model_output:np.ndarray,
                    num_classes:int,
                    anchors_list:list[float],
                    model_input_height:int, 
                    model_input_width:int, 
                    image_height:int, 
                    image_width:int,
                    max_output_boxes:int=10, 
                    confidence_thr:float=0.5, 
                    iou_threshold:float=0.5) -> List[Tuple[BBox, int, float]]:
        
        """postprocess output from AnchorsBased YOLO model

        Args:
            model_output (np.ndarray): output matrix of the model
            num_classes (int): number of classes
            anchors_list (list[float]): anchors representting your dataset in normalized format
            model_input_height (int): input height of the model
            model_input_width (int): input width of the model
            image_height (int): input height of the current inferenced image
            image_width (int): input width of the current inferenced image
            max_output_boxes (int): maximum number of bounding box to be considered for non-max suppression
            confidence_thr (float): the threshold of the confidence score
            iou_threshold (float): the threshold of the Intersection over Union for NMS

        Returns:
            list[list]: list of the preicted bounding box in the image
        """
        
        anchors = np.array(anchors_list).reshape(-1, 2)
        input_shape = (model_input_height, model_input_width)
        preds_decoded = decode(model_output, anchors, num_classes, input_shape, calc_loss=False)
        input_image_shape = [image_height, image_width]

        boxes, scores, classes, my_boxes = tiny_yolo_v2_nms(yolo_outputs = preds_decoded,
                                                            image_shape = input_image_shape,
                                                            max_boxes=max_output_boxes,
                                                            score_threshold=confidence_thr,
                                                            iou_threshold=iou_threshold,
                                                            classes_ids=list(range(0, num_classes)))


        result = []
        for i, c in reversed(list(enumerate(classes))):
            box = boxes[i]
            score = scores[i]
            box_object = BBox(box, format=BBoxFormat.CORNERS, img_width=image_width, img_height=image_height)
            result.append((box_object, int(c), score))
            
        return result
    
    def __call__(self,
                model_output:np.ndarray,
                num_classes:int,
                anchors_list:list[float],
                model_input_height:int, 
                model_input_width:int, 
                image_height:int, 
                image_width:int,
                max_output_boxes:int=10, 
                confidence_thr:float=0.5, 
                iou_threshold:float=0.5) -> List[Tuple[BBox, int, float]]:         
        """
        Directly invoking the postprocess method.

        Args:
            model_output (np.ndarray): output matrix of the model
            num_classes (int):  num of the classes
            anchors (list[float]): anchors representting your dataset in normalized format
            model_input_height (int): input height of the model
            model_input_width (int): input width of the model
            image_height (int): input height of the current inferenced image
            image_width (int): input width of the current inferenced image
            max_output_boxes (int): maximum number of bounding box to be considered for non-max suppression
            confidence_thr (float): the threshold of the confidence score
            iou_threshold (float): the threshold of the Intersection over Union for NMS

        Returns:
            list[list]: List of predicted bounding boxes in the image.
        """
        return self.postprocess(model_output, num_classes, anchors_list, model_input_height, model_input_width, image_height, 
                                image_width, max_output_boxes, confidence_thr, iou_threshold)