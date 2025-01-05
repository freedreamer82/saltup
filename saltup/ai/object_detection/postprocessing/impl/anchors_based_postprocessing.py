import numpy as np
from tensorflow.keras import backend as K
import tensorflow as tf
from saltup.ai.object_detection.dataset.bbox_utils import calculate_iou, process_boxes
from saltup.ai.object_detection.postprocessing import Postprocessing

def decode(yolo_output, anchors, num_classes, input_shape, calc_loss=False):
    """
    Decode YOLO output derived from loss function logic.
    Returns box coordinates, dimensions, confidence and class probabilities.
    """
    # Calculate grid size based on input shape and network stride
    stride = input_shape[0] / yolo_output.shape[1]
    grid_h = int(input_shape[0] // stride)
    grid_w = int(input_shape[1] // stride)
    num_anchors = len(anchors)
    
    # Reshape output
    yolo_output = tf.reshape(yolo_output, 
                           [-1, grid_h, grid_w, num_anchors, 5 + num_classes])
    yolo_output = tf.cast(yolo_output, tf.float32)

    # Parse predictions
    box_xy = tf.sigmoid(yolo_output[..., 0:2])  # sigmoid per xy
    box_wh = tf.exp(yolo_output[..., 2:4])      # exp per wh perché erano in log
    box_confidence = tf.sigmoid(yolo_output[..., 4:5])
    box_class_probs = tf.sigmoid(yolo_output[..., 5:])
    
    # Create grid
    grid_y = tf.cast(tf.reshape(tf.range(grid_h), [-1, 1, 1, 1]), tf.float32)
    grid_x = tf.cast(tf.reshape(tf.range(grid_w), [1, -1, 1, 1]), tf.float32)
    grid_y = tf.tile(grid_y, [1, grid_w, 1, 1])
    grid_x = tf.tile(grid_x, [grid_h, 1, 1, 1])
    grid = tf.concat([grid_x, grid_y], axis=-1)
    grid = tf.tile(grid, [1, 1, num_anchors, 1])
    grid = tf.expand_dims(grid, axis=0)
    
    # Normalize predictions
    box_xy = (box_xy + grid) / tf.constant([grid_w, grid_h], dtype=tf.float32)
    
    # Prepare anchors
    anchors_tensor = tf.tile(
        tf.reshape(tf.cast(anchors, tf.float32), [1, 1, 1, num_anchors, 2]),
        [1, grid_h, grid_w, 1, 1]
    )
    
    # Scale box dimensions by anchors (già con exp applicato)
    box_wh = box_wh * anchors_tensor
    
    if calc_loss == True:
        return grid, yolo_output, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs

def filter_boxes(my_boxes,boxes, box_confidence, box_class_probs, threshold=0.5):
    """
    Filters YOLO boxes based on object and class confidence.

    Args:
        my_boxes (tensor): containing the coordinates of the boxes in the original image dimensions.
        boxes (tensor): containing the coordinates of the boxes.
        box_confidence (tensor): containing the object confidence scores.
        box_class_probs (tensor): containing the class probabilities.
        threshold (float): threshold for box score to be considered as a detection.

    Returns:
        boxes (tensor): containing the coordinates of the filtered boxes in corners format.
        scores (tensor): containing the scores of the filtered boxes.
        classes (tensor): containing the class IDs of the filtered boxes.
        my_boxes (tensor): containing the coordinates of the filtered boxes in centoids format.
    """
    box_scores =  box_confidence*box_class_probs
    box_classes = K.argmax(box_scores, axis=-1)
    box_class_scores = K.max(box_scores, axis=-1)
    prediction_mask = box_class_scores >= threshold
    boxes = tf.boolean_mask(boxes, prediction_mask)
    my_boxes = tf.boolean_mask(my_boxes, prediction_mask)
    scores = tf.boolean_mask(box_class_scores, prediction_mask)
    classes = tf.boolean_mask(box_classes, prediction_mask)
    return boxes, scores, classes,my_boxes


def tiny_yolo_v2_nms(yolo_outputs,image_shape,max_boxes=30,score_threshold=.5,iou_threshold=.3 , classes_ids=[0]):
    """
    Applies non-max suppression to the output of Tiny YOLO v2 model.

    Args:
        yolo_outputs (list): output of the Tiny YOLO v2 model.
        image_shape (tuple): shape of the input image (height, width).
        max_boxes (int): maximum number of boxes to be selected by non-max suppression.
        score_threshold (float): threshold for box score to be considered as a detection.
        iou_threshold (float): threshold for intersection over union to be considered as a duplicate detection.
        classes_ids (list): list of class IDs to perform non-max suppression on.

    Returns:
        s_boxes (tensor): tensor of shape (num_boxes, 4) containing the coordinates of the selected boxes in corners format.
        s_scores (tensor): tensor of shape (num_boxes,) containing the scores of the selected boxes.
        s_classes (tensor): tensor of shape (num_boxes,) containing the class IDs of the selected boxes.
        s_my_boxes (tensor): tensor of shape (num_boxes, 4) containing the coordinates of the selected boxes in centroids format.
    """
    box_xy, box_wh, box_confidence, box_class_probs = yolo_outputs
    boxes,my_boxes = process_boxes(box_xy, box_wh)
    boxes, scores, classes,my_boxes= filter_boxes(my_boxes,boxes, box_confidence, box_class_probs, threshold=score_threshold)
    height = image_shape[0]
    width = image_shape[1]
    image_dims = K.stack([height, width, height, width])
    image_dims = K.reshape(image_dims, [1, 4])
    image_dims = K.cast(image_dims, dtype='float32')
    boxes = boxes * image_dims
    max_boxes_tensor = K.variable(max_boxes, dtype='int32')
    total_boxes = []
    total_scores = []
    total_classes = []
    total_my_boxes = []
    #apply nms per class
    for c in classes_ids:
        mask =tf.equal(classes, c)
        s_classes = tf.boolean_mask(classes, mask)
        s_scores = tf.boolean_mask(scores, mask)
        s_boxes = tf.boolean_mask(boxes, mask)
        s_my_boxes = tf.boolean_mask(my_boxes, mask)

        nms_index = tf.image.non_max_suppression(
            s_boxes, s_scores, max_boxes_tensor, iou_threshold=iou_threshold)
        s_boxes = K.gather(s_boxes, nms_index)
        s_scores = K.gather(s_scores, nms_index)
        s_classes = K.gather(s_classes, nms_index)
        s_my_boxes = K.gather(s_my_boxes, nms_index)

        total_boxes.append(s_boxes)
        total_scores.append(s_scores)
        total_classes.append(s_classes)
        total_my_boxes.append(s_my_boxes)
    s_boxes = K.concatenate(total_boxes, axis=0)
    s_my_boxes = K.concatenate(total_my_boxes, axis=0)
    s_scores = K.concatenate(total_scores, axis=0)
    s_classes = K.concatenate(total_classes, axis=0)
    return s_boxes, s_scores, s_classes,s_my_boxes

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
                    classes_name:list[str],
                    anchors_list:list[float],
                    model_input_height:int, 
                    model_input_width:int, 
                    image_height:int, 
                    image_width:int,
                    max_output_boxes:int=10, 
                    confidence_thr:float=0.5, 
                    iou_threshold:float=0.5) -> list[list]:
        
        """postprocess output from AnchorsBased YOLO model

        Args:
            model_output (np.ndarray): output matrix of the model
            classes_name (list[str]): list of the name of the classes
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
        
        predictions_tensor = K.constant(model_output)

        num_classes = len(classes_name)
        
        input_shape = (model_input_height, model_input_width)
        preds_decoded = decode(predictions_tensor, anchors, num_classes, input_shape, calc_loss=False)
        input_image_shape = [image_height, image_width]
        
        boxes, scores, classes, my_boxes = tiny_yolo_v2_nms(yolo_outputs = preds_decoded,
                                                            image_shape = input_image_shape,
                                                            max_boxes=max_output_boxes,
                                                            score_threshold=confidence_thr,
                                                            iou_threshold=iou_threshold,
                                                            classes_ids=list(range(0, num_classes)))
        classes = classes.numpy()
        print(f'classes: {classes}')
        scores = scores.numpy()
        print(f'scores: {scores}')
        my_boxes = my_boxes.numpy()
        boxes = boxes.numpy()
        result = []
        for i, c in reversed(list(enumerate(classes))):
            box_c = my_boxes[i]
            score = scores[i]
            label = classes_name[int(c)]
            yyy, xxx, hhh, www = box_c
            x1 = int((xxx - www / 2) * image_width)
            y1 = int((yyy - hhh / 2) * image_height)
            x2 = int((xxx + www / 2) * image_width)
            y2 = int((yyy + hhh / 2) * image_height)
            result.append([x1, y1, x2, y2, label ,score])
        return result
    
    def __call__(self,
                model_output:np.ndarray,
                classes_name:list[str],
                anchors_list:list[float],
                model_input_height:int, 
                model_input_width:int, 
                image_height:int, 
                image_width:int,
                max_output_boxes:int=10, 
                confidence_thr:float=0.5, 
                iou_threshold:float=0.5) -> list[list]:         
        """
        Directly invoking the postprocess method.

        Args:
            model_output (np.ndarray): output matrix of the model
            classes_name (list[str]): list of the name of the classes
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
        return self.postprocess(model_output, classes_name, anchors_list, model_input_height, model_input_width, image_height, 
                                image_width, max_output_boxes, confidence_thr, iou_threshold)