import os
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from saltup.ai.object_detection.utils.bbox import BBox, BBoxFormat, convert_matrix_boxes, nms


def compute_anchors(boxes: np.ndarray, num_anchors: int) -> np.ndarray:
   """
   Compute optimal anchor boxes using K-means clustering.
   
   Args:
       boxes: Array of shape (N, 2) containing width and height of bounding boxes
       num_anchors: Number of anchor boxes to generate
       
   Returns:
       np.ndarray: Array of shape (num_anchors, 2) containing the computed anchor boxes
                  in format (width, height)
                  
   Raises:
       ValueError: If boxes is empty or has wrong shape
       ValueError: If num_anchors is less than 1 or greater than number of boxes
       TypeError: If boxes is not a numpy array
   """
   # Input validation
   if not isinstance(boxes, np.ndarray):
       raise TypeError("boxes must be a numpy array")
       
   if len(boxes.shape) != 2 or boxes.shape[1] != 2:
       raise ValueError("boxes must have shape (N, 2) where N is number of boxes")
       
   if boxes.shape[0] == 0:
       raise ValueError("boxes array is empty")
       
   if num_anchors < 1:
       raise ValueError("num_anchors must be greater than 0")
       
   if num_anchors > boxes.shape[0]:
       raise ValueError(f"num_anchors ({num_anchors}) cannot be greater than number of boxes ({boxes.shape[0]})")

   # Perform K-means clustering
   kmeans = KMeans(n_clusters=num_anchors, random_state=0)
   kmeans.fit(boxes)

   # Return cluster centroids (optimal anchors)
   return kmeans.cluster_centers_


def compute_anchor_iou(wh1: np.ndarray, wh2: np.ndarray) -> float:
   """
   Compute IoU between two anchor boxes using only width and height.
   Assumes boxes are centered at the same point.
   
   Args:
       wh1: Width and height of first box as (w,h)
       wh2: Width and height of second box as (w,h)
       
   Returns:
       float: IoU value between 0 and 1
       
   Note:
       This is a simplified IoU calculation specifically for comparing anchor
       boxes, where only width/height are considered and boxes are assumed
       to be centered.
   """
   # Get width and height
   w1, h1 = wh1
   w2, h2 = wh2
   
   # Calculate intersection area (assumes centered boxes)
   inter_w = min(w1, w2)
   inter_h = min(h1, h2)
   intersection = inter_w * inter_h
   
   # Calculate areas
   area1 = w1 * h1
   area2 = w2 * h2
   
   # Calculate union and IoU
   union = area1 + area2 - intersection
   return intersection / (union + np.finfo(float).eps)


def convert_to_grid_format(
   boxes: np.ndarray,
   class_labels: list[int],
   grid_size: tuple[int, int],
   anchors: list[tuple[float, float]],
   num_classes: int
) -> np.ndarray:
    """
    Convert bounding boxes to YOLO grid format.
    
    Args:
        boxes: array of shape (N, 4) containing bounding boxes in format 
              [x_center, y_center, width, height] normalized to [0,1]
        class_labels: list of class indices for each box
        grid_size: tuple of (height, width) for the grid
        anchors: list of tuples [(width, height), ...] for anchor boxes
        num_classes: number of classes
        
    Returns:
        np.ndarray: grid labels of shape (1, grid_h, grid_w, num_anchors, 5 + num_classes)
                   where 5 represents [x, y, w, h, objectness]
    """
    grid_labels = np.zeros((1, grid_size[0], grid_size[1],
                        len(anchors), 5 + num_classes), dtype=np.float32)
    
    for box, class_label in zip(boxes, class_labels):
        # Extract box coordinates
        x_center, y_center, width, height = box
        
        # Calculate grid cell location
        grid_x = int(x_center * grid_size[1])
        grid_y = int(y_center * grid_size[0])
        
        # Clip to ensure valid grid indices
        grid_x = np.clip(grid_x, 0, grid_size[1] - 1)
        grid_y = np.clip(grid_y, 0, grid_size[0] - 1)
        
        # Find best matching anchor box
        box_wh = np.array([width, height])
        best_iou = 0
        best_anchor = -1
        
        for i, anchor in enumerate(anchors):
            iou = compute_anchor_iou(box_wh, np.array(anchor))
            if iou > best_iou:
                best_iou = iou
                best_anchor = i
        
        if best_iou > 0:
            # Store box coordinates relative to grid cell
            grid_labels[0, grid_y, grid_x, best_anchor, 0] = x_center * grid_size[1] - grid_x
            grid_labels[0, grid_y, grid_x, best_anchor, 1] = y_center * grid_size[0] - grid_y
            
            # Store width and height as log scale relative to anchor box
            grid_labels[0, grid_y, grid_x, best_anchor, 2] = np.log(width / anchors[best_anchor][0])
            grid_labels[0, grid_y, grid_x, best_anchor, 3] = np.log(height / anchors[best_anchor][1])
            
            # Set objectness score
            grid_labels[0, grid_y, grid_x, best_anchor, 4] = 1.0
            
            # Set class one-hot encoding
            grid_labels[0, grid_y, grid_x, best_anchor, 5 + int(class_label)] = 1.0
    
    return grid_labels


def plot_anchors(
    anchors: np.ndarray, 
    image_size: tuple[int, int], 
    title: str = "Anchor Boxes"
) -> None:
   """
   Plot anchor boxes on an image with specified dimensions.
   
   Args:
       anchors: Array-like of normalized anchors in format (width, height)
       image_size: Image dimensions as (height, width)
       title: Plot title
       
   Returns:
       None: Displays the plot with matplotlib
   """
   # Extract dimensions
   image_height, image_width = image_size
   
   # Convert anchor data to numpy array
   anchors = np.array(anchors)
   
   # Scale anchors to image dimensions
   scaled_anchors = anchors * [image_width, image_height]

   # Create figure with empty canvas
   fig, ax = plt.subplots(figsize=(8, 8))
   ax.set_xlim(0, image_width)
   ax.set_ylim(0, image_height)
   ax.invert_yaxis()  # Match image coordinate system
   ax.set_title(title)
   ax.set_xlabel('Width (pixels)')
   ax.set_ylabel('Height (pixels)')

   # Draw anchor boxes
   for anchor in scaled_anchors:
       w, h = anchor
       rect = patches.Rectangle(
           ((image_width - w) / 2, (image_height - h) / 2),  # Center the anchor
           w, h,
           linewidth=2, edgecolor='red', facecolor='none'
       )
       ax.add_patch(rect)
   
   plt.grid(True)
   plt.show()
   

def postprocess_decode(yolo_output, anchors, num_classes, input_shape, calc_loss=False):
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
    #box_class_probs = 1 / (1 + np.exp(-yolo_output[..., 5:]))
    exp_class = np.exp(yolo_output[..., 5:])
    box_class_probs = exp_class / np.sum(exp_class, axis=-1, keepdims=True)

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


def postprocess_filter_boxes(centers_boxes, corners_boxes, box_confidence, box_class_probs, threshold=0.5):
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
    corners_boxes = corners_boxes[prediction_mask]
    centers_boxes = centers_boxes[prediction_mask]
    scores = box_class_scores[prediction_mask]
    classes = box_classes[prediction_mask]

    return corners_boxes, scores, classes, centers_boxes


def tiny_anchors_based_nms(
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
    corners_boxes, centers_boxes = convert_matrix_boxes(box_xy, box_wh)
    corners_boxes, scores, classes, centers_boxes = postprocess_filter_boxes(
        centers_boxes, corners_boxes, box_confidence, box_class_probs, threshold=score_threshold
    )

    # Scale boxes to image dimensions
    height, width = image_shape
    image_dims = np.array([width, height, width, height], dtype=np.float32)
    corners_boxes = corners_boxes * image_dims  # Convert to original image dimensions
    
    #to avoid invalid boxes (negative values)
    corners_boxes[:, 0] = np.maximum(0, np.minimum(width, corners_boxes[:, 0]))  # x1
    corners_boxes[:, 1] = np.maximum(0, np.minimum(height, corners_boxes[:, 1]))  # y1
    corners_boxes[:, 2] = np.maximum(0, np.minimum(width, corners_boxes[:, 2]))  # x2
    corners_boxes[:, 3] = np.maximum(0, np.minimum(height, corners_boxes[:, 3]))  # y2

    # Wrap boxes into BBox objects
    bboxes = [BBox(img_height=height, img_width=width, coordinates=box.tolist(), format=BBoxFormat.CORNERS) for box in corners_boxes]

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
        s_corners_boxes = np.array([box.get_coordinates() for box in total_boxes], dtype=np.float32)
        s_scores = np.array(total_scores, dtype=np.float32)
        s_classes = np.array(total_classes, dtype=np.int32)
        s_centers_boxes = np.array([box.to_yolo() for box in total_boxes], dtype=np.float32)
    else:
        s_corners_boxes = np.empty((0, 4), dtype=np.float32)
        s_scores = np.empty((0,), dtype=np.float32)
        s_classes = np.empty((0,), dtype=np.int32)
        s_centers_boxes = np.empty((0, 4), dtype=np.float32)

    return s_corners_boxes, s_scores, s_classes, s_centers_boxes