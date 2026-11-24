from typing import List, Sequence, Tuple, Union

import numpy as np
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
    # scikit-learn is only needed to compute anchors (a training-time step),
    # so import it lazily to keep inference-only installs light.
    try:
        from sklearn.cluster import KMeans
    except ImportError as e:
        raise ImportError(
            "scikit-learn is required by compute_anchors. "
            'Install it with: pip install "saltup[training]" or pip install scikit-learn'
        ) from e
    # sklearn is unannotated: pyright infers n_init as str from its "auto" default
    kmeans = KMeans(n_clusters=num_anchors, random_state=0, n_init=10)  # pyright: ignore[reportArgumentType]
    kmeans.fit(boxes)

    # Sort anchors by area (width * height) for deterministic order
    anchors = kmeans.cluster_centers_
    anchors = anchors[np.argsort(anchors[:, 0] * anchors[:, 1])]

    return anchors


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
    num_classes: int,
) -> np.ndarray:
    """
    Encode ground-truth boxes into the anchor grid used as training target.

    This is the inverse of `postprocess_decode`. Each box is assigned to exactly
    one slot (cell, anchor), called the responsible slot:

    - cell: the grid cell that contains the box center,
      `cell_x = floor(x_center * grid_w)`, `cell_y = floor(y_center * grid_h)`.
    - anchor: the anchor with the highest shape IoU against the box
      (width and height only, both centered at the same point).

    The responsible slot stores:

    - `[0:2]` center offset inside the cell, in [0, 1):
      `x_center * grid_w - cell_x`, `y_center * grid_h - cell_y`.
    - `[2:4]` log size ratio to the anchor:
      `log(width / anchor_w)`, `log(height / anchor_h)`.
    - `[4]` objectness target, always 1.0.
    - `[5 + class_id]` one-hot class target, 1.0.

    Every other slot stays all zeros, i.e. objectness target 0. Two boxes that
    fall in the same cell and pick the same anchor overwrite each other: only
    the last one is kept.

    Args:
        boxes: array of shape (N, 4), boxes as [x_center, y_center, width, height]
            normalized to [0, 1] of the image.
        class_labels: class index of each box.
        grid_size: grid shape as (grid_h, grid_w).
        anchors: anchor sizes as [(width, height), ...], normalized to [0, 1]
            of the image.
        num_classes: number of classes.

    Returns:
        np.ndarray: targets of shape (1, grid_h, grid_w, num_anchors, 5 + num_classes).
    """
    grid_labels = np.zeros((1, grid_size[0], grid_size[1],
                        len(anchors), 5 + num_classes), dtype=np.float32)
    
    for box, class_label in zip(boxes, class_labels):
        x_center, y_center, width, height = box

        # Cell that contains the box center; a center exactly on the right or
        # bottom edge (1.0) would index past the grid, hence the clip.
        grid_x = int(x_center * grid_size[1])
        grid_y = int(y_center * grid_size[0])
        grid_x = np.clip(grid_x, 0, grid_size[1] - 1)
        grid_y = np.clip(grid_y, 0, grid_size[0] - 1)

        # Anchor whose shape overlaps the box best (position ignored)
        box_wh = np.array([width, height])
        best_iou = 0
        best_anchor = -1
        
        for i, anchor in enumerate(anchors):
            iou = compute_anchor_iou(box_wh, np.array(anchor))
            if iou > best_iou:
                best_iou = iou
                best_anchor = i
        
        # A zero-size box overlaps no anchor and is skipped
        if best_iou > 0:
            # Center offset inside the cell, in [0, 1)
            grid_labels[0, grid_y, grid_x, best_anchor, 0] = x_center * grid_size[1] - grid_x
            grid_labels[0, grid_y, grid_x, best_anchor, 1] = y_center * grid_size[0] - grid_y

            # Log size ratio to the anchor: 0 when the box has the anchor's size
            grid_labels[0, grid_y, grid_x, best_anchor, 2] = np.log(width / anchors[best_anchor][0])
            grid_labels[0, grid_y, grid_x, best_anchor, 3] = np.log(height / anchors[best_anchor][1])

            # Objectness target: 1 only in the responsible slot
            grid_labels[0, grid_y, grid_x, best_anchor, 4] = 1.0

            grid_labels[0, grid_y, grid_x, best_anchor, 5 + int(class_label)] = 1.0
    
    return grid_labels


def plot_anchors(
    anchors: np.ndarray,
    image_size: tuple[int, int],
    title: str = "Anchor Boxes",
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
   

def postprocess_decode(
    yolo_output: np.ndarray,
    anchors: Union[np.ndarray, Sequence[Tuple[float, float]]],
    num_classes: int,
    input_shape: tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Decode the raw model output into boxes, objectness and class probabilities.

    The model predicts one vector of 5 + num_classes raw values (logits) for
    every slot, where a slot is one anchor in one grid cell:

        [t_x, t_y, t_w, t_h, t_obj, t_class_0, ..., t_class_{C-1}]

    Each slot is decoded as:

    - center: `cx = (sigmoid(t_x) + cell_x) / grid_w`,
      `cy = (sigmoid(t_y) + cell_y) / grid_h`.
      The sigmoid keeps the center inside the cell that predicted it.
    - size: `w = anchor_w * exp(t_w)`, `h = anchor_h * exp(t_h)`.
      A raw value of 0 gives exactly the anchor's size.
    - objectness: `sigmoid(t_obj)`, the score that the slot holds an object
      of any class. Training (see `convert_to_grid_format`) pushes it to 1 in
      the responsible slot of each box and to 0 everywhere else.
    - class probabilities: `softmax(t_class)`, the class distribution given
      that an object is there; it sums to 1 over the classes.

    Centers and sizes are normalized to [0, 1] of the model input, the same
    unit as the anchors.

    Args:
        yolo_output: raw model output. Axis 1 must be grid_h and the total size
            must reshape to (batch, grid_h, grid_w, num_anchors, 5 + num_classes).
        anchors: anchor sizes as [(width, height), ...], normalized to [0, 1]
            of the model input.
        num_classes: number of classes.
        input_shape: model input size as (height, width) in pixels.

    Returns:
        tuple: (centers, sizes, objectness, class_probs), arrays of shape
        (batch, grid_h, grid_w, num_anchors, K) with K equal to 2, 2, 1 and
        num_classes respectively.
    """
    # Same stride (pixels per cell) on both axes, derived from the height
    stride = input_shape[0] / yolo_output.shape[1]
    grid_h = int(input_shape[0] // stride)
    grid_w = int(input_shape[1] // stride)
    num_anchors = len(anchors)

    raw = yolo_output.reshape(-1, grid_h, grid_w, num_anchors, 5 + num_classes).astype(np.float32)

    offsets_in_cell = 1 / (1 + np.exp(-raw[..., 0:2]))
    size_ratios = np.exp(raw[..., 2:4])
    objectness = 1 / (1 + np.exp(-raw[..., 4:5]))
    exp_class = np.exp(raw[..., 5:])
    class_probs = exp_class / np.sum(exp_class, axis=-1, keepdims=True)

    # (grid_h, grid_w, num_anchors, 2) array of (cell_x, cell_y) indices
    cell_y = np.tile(np.arange(grid_h).reshape(-1, 1, 1, 1), (1, grid_w, 1, 1))
    cell_x = np.tile(np.arange(grid_w).reshape(1, -1, 1, 1), (grid_h, 1, 1, 1))
    cell_xy = np.tile(np.concatenate([cell_x, cell_y], axis=-1), (1, 1, num_anchors, 1))

    centers = (offsets_in_cell + cell_xy) / np.array([grid_w, grid_h], dtype=np.float32)

    anchor_sizes = np.array(anchors, dtype=np.float32).reshape(1, 1, 1, num_anchors, 2)
    sizes = size_ratios * anchor_sizes

    return centers, sizes, objectness, class_probs


def postprocess_filter_boxes(
    centers_boxes: np.ndarray,
    corners_boxes: np.ndarray,
    box_confidence: np.ndarray,
    box_class_probs: np.ndarray,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Keep the slots whose best class score reaches the threshold.

    The class score of a slot for class c is `objectness * class_probs[c]`:
    the score that the slot holds an object and that the object is of class c.
    Each slot is labelled with the class of its highest score, and kept when
    that score is >= threshold.

    Args:
        centers_boxes (numpy.ndarray): boxes as (cx, cy, w, h), shape (..., 4).
        corners_boxes (numpy.ndarray): the same boxes as (x1, y1, x2, y2), shape (..., 4).
        box_confidence (numpy.ndarray): objectness, shape (..., 1).
        box_class_probs (numpy.ndarray): class probabilities, shape (..., num_classes).
        threshold (float): minimum class score to keep a slot.

    Returns:
        tuple: (corners_boxes, scores, classes, centers_boxes) of the kept slots,
        flattened to shape (N, 4), (N,), (N,) and (N, 4).
    """
    class_scores = box_confidence * box_class_probs
    best_class = np.argmax(class_scores, axis=-1)
    best_score = np.max(class_scores, axis=-1)

    keep = best_score >= threshold

    corners_boxes = corners_boxes[keep]
    centers_boxes = centers_boxes[keep]
    scores = best_score[keep]
    classes = best_class[keep]

    return corners_boxes, scores, classes, centers_boxes


def tiny_anchors_based_nms(
    yolo_outputs: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    image_shape: Sequence[int],
    max_boxes: int = 30,
    score_threshold: float = 0.5,
    iou_threshold: float = 0.3,
    classes_ids: List[int] = [0],
    class_agnostic: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Applies non-max suppression to the output of Tiny YOLO v2 model.

    Args:
        yolo_outputs (list): Output of the Tiny YOLO v2 model.
        image_shape (tuple): Shape of the input image (height, width).
        max_boxes (int): Maximum number of boxes to be selected by non-max suppression.
        score_threshold (float): Threshold for box score to be considered as a detection.
        iou_threshold (float): Threshold for intersection over union to be considered as a duplicate detection.
        classes_ids (list): List of class IDs to perform non-max suppression on.
        class_agnostic (bool): If False (default), NMS runs separately for each class and
            max_boxes applies per class. If True, a single NMS runs over all classes, so
            overlapping boxes of different classes suppress each other, and max_boxes
            applies to the total.

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
    bboxes = [BBox(coordinates=box.tolist(), fmt=BBoxFormat.CORNERS_ABSOLUTE, img_height=height, img_width=width) for box in corners_boxes]

    total_boxes, total_scores, total_classes = [], [], []

    # One NMS group per class, or a single group with every class when agnostic
    groups = [list(classes_ids)] if class_agnostic else [[c] for c in classes_ids]

    for group in groups:
        # Filter by class ID
        mask = np.isin(classes, group)
        class_boxes = [bboxes[i] for i in np.where(mask)[0]]
        class_scores = scores[mask]
        group_classes = classes[mask]

        if not class_boxes:
            continue

        # Apply NMS
        selected_boxes = nms(class_boxes, class_scores, iou_threshold, max_boxes)
        
        selected_indices = [class_boxes.index(box) for box in selected_boxes]

        # Collect results
        total_boxes.extend(selected_boxes)
        total_scores.extend(class_scores[selected_indices])
        total_classes.extend(group_classes[selected_indices])

    # Convert results to numpy arrays
    if total_boxes:
        s_corners_boxes = np.array([box.get_coordinates() for box in total_boxes], dtype=np.float32)
        s_scores = np.array(total_scores, dtype=np.float32)
        s_classes = np.array(total_classes, dtype=np.int32)
        s_centers_boxes = np.array([box.get_coordinates(fmt=BBoxFormat.YOLO) for box in total_boxes], dtype=np.float32)
    else:
        s_corners_boxes = np.empty((0, 4), dtype=np.float32)
        s_scores = np.empty((0,), dtype=np.float32)
        s_classes = np.empty((0,), dtype=np.int32)
        s_centers_boxes = np.empty((0, 4), dtype=np.float32)

    return s_corners_boxes, s_scores, s_classes, s_centers_boxes