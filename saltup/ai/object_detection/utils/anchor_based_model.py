import os
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.patches as patches


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