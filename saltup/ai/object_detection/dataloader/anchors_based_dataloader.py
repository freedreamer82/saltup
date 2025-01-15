from typing import Tuple
import albumentations as A
import numpy as np
import os

from saltup.ai.object_detection.dataset.base_dataset_loader import BaseDatasetLoader
from saltup.ai.object_detection.preprocessing.impl.anchors_based_preprocess import AnchorsBasedPreprocess
from saltup.ai.object_detection.utils.anchor_based_model import convert_to_grid_format
from saltup.ai.object_detection.utils.bbox import (
    compute_iou,
    center_to_corners_format, 
    absolute_bbox,
    BBoxFormat
)
from saltup.utils.configure_logging import get_logger


class AnchorsBasedDataloader:
    def __init__(
        self, 
        dataset_loader: BaseDatasetLoader,
        anchors: np.ndarray,
        target_size: Tuple[int, int], 
        grid_size: Tuple[int, int],
        num_classes: int,
        batch_size: int = 1,
        preprocess: callable = None,
        transform: A.Compose = None
    ):
        self.dataset_loader = dataset_loader
        self.__indexes = np.arange(len(dataset_loader))
        
        self.anchors = anchors
        self.__num_anchors = len(anchors)
        
        self.batch_size = batch_size
        self.target_size = target_size
        self.grid_size = grid_size
        self.num_classes = num_classes
        
        self.transform = transform
        self.augment = True if self.transform else False
        
        self.preprocess = preprocess
        if not self.preprocess:
            self.preprocess = AnchorsBasedPreprocess(apply_padding=False)
            
        self.__logger = get_logger(__name__)
        self.__logger.info("Initializing AnchorsBasedDataloader")
        
    def __len__(self):
        return int(np.ceil(len(self.dataset_loader) / self.batch_size))
    
    def __iter__(self):
        for i in range(0, len(self), self.batch_size):
            yield self[i]
    
    def __getitem__(self, idx):
        batch_indexes = self.__indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_size = len(batch_indexes)
        
        images = np.zeros((batch_size, self.target_size[0], self.target_size[1], 1), dtype=np.float32)
        labels = np.zeros((batch_size, self.grid_size[0], self.grid_size[1],
                        self.__num_anchors, 5 + self.num_classes), dtype=np.float32)
        
        for i, idx in enumerate(batch_indexes):
            try:
                # Get image and labels from dataset loader
                image, annotation_data = next(self.dataset_loader)
                
                # Extract boxes and class labels
                boxes, class_labels = annotation_data[:, :4], annotation_data[:, 4]
                
                # Preprocess image
                image = self.preprocess(image, self.target_size)
                
                # Apply augmentations
                if len(boxes) > 0 and self.augment:
                    try:
                        transformed = self.transform(
                            image=image.squeeze(),
                            bboxes=boxes.tolist(),
                            class_labels=class_labels
                        )
                        
                        if transformed['bboxes']:
                            new_boxes = np.array(transformed['bboxes'])
                            
                            # Check that the transformed boxes are valid
                            valid_mask = (
                                (new_boxes[:, 0] >= 0) & (new_boxes[:, 0] <= 1) &  # x_center
                                (new_boxes[:, 1] >= 0) & (new_boxes[:, 1] <= 1) &  # y_center
                                (new_boxes[:, 2] > 0) & (new_boxes[:, 2] < 1) &    # width
                                (new_boxes[:, 3] > 0) & (new_boxes[:, 3] < 1)      # height
                            )
                            
                            if np.any(valid_mask):
                                image = np.expand_dims(transformed['image'], axis=-1)
                                boxes = new_boxes[valid_mask]
                                class_labels = np.array(transformed['class_labels'])[valid_mask]
                            else:
                                pass
                    except Exception as e:
                        self.__logger.error(f"Error in augmenting: {e}")
                
                # Convert labels to grid format
                if len(boxes) > 0:
                    grid_labels = convert_to_grid_format(
                        boxes, class_labels,
                        grid_size=self.grid_size,
                        anchors=self.anchors,
                        num_classes=self.num_classes
                    )
                    images[i] = image
                    labels[i] = grid_labels
                else:
                    images[i] = image
                
            except Exception as e:
                self.__logger.error(f"Failed to process batch item {idx}: {e}")
                continue
            
            return images, labels
        
    def on_epoch_end(self):
        np.random.shuffle(self.__indexes)

    def visualize_sample(self, idx, show_grid=True, show_anchors=False):
        
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        try:
            image, annotation_data = self.dataset_loader[idx]
            
            # Extract boxes and class labels
            boxes, class_labels = annotation_data[:, :4], annotation_data[:, 4]

            # Preprocess image
            processed_image = self.preprocess(image, self.target_size)
            
            # Create subplot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
            
            # Plot original image
            ax1.imshow(image, cmap='gray' if len(image.shape) == 2 else None)
            ax1.set_title(f'Original Image ({image.shape[1]}x{image.shape[0]})')
            
            # Draw original boxes
            for box, class_id in zip(boxes, class_labels):
                x_center, y_center, width, height = box
                
                # TODO: makes bbox indipendent from the dataset format
                # Convert normalized coordinates to pixels
                x1 = int((x_center - width/2) * image.shape[1])
                y1 = int((y_center - height/2) * image.shape[0])
                x2 = int((x_center + width/2) * image.shape[1])
                y2 = int((y_center + height/2) * image.shape[0])
                
                # Draw rectangle
                rect = patches.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    linewidth=2,
                    edgecolor='r',
                    facecolor='none'
                )
                ax1.add_patch(rect)
                
                # Add label
                ax1.text(
                    x1, y1-5,
                    f'Class {int(class_id)}',
                    color='red',
                    fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.7)
                )
            
            # Plot preprocessed image
            ax2.imshow(processed_image.squeeze(), cmap='gray' if len(processed_image.shape) == 3 else None)
            ax2.set_title(f'Preprocessed Image ({self.target_size[1]}x{self.target_size[0]})')
        
            # Show YOLO grid
            if show_grid and len(boxes) > 0:
                grid_size_h, grid_size_w = self.grid_size
                cell_size_h = self.target_size[0] // grid_size_h
                cell_size_w = self.target_size[1] // grid_size_w
                
                for x in range(grid_size_w):
                    for y in range(grid_size_h):
                        rect = patches.Rectangle(
                            (x * cell_size_w, y * cell_size_h),
                            cell_size_w, cell_size_h,
                            linewidth=1,
                            edgecolor='green',
                            facecolor='none',
                            linestyle=':'
                        )
                        ax2.add_patch(rect)
                        
            # Show anchor boxes
            if show_anchors and len(boxes) > 0:
                for box in boxes:
                    x_center, y_center, width, height = box
                    best_iou = 0
                    best_anchor = None
                    
                    # Find best matching anchor
                    for anchor in self.anchors:
                        iou = compute_iou(np.array([width, height]), np.array(anchor))
                        if iou > best_iou:
                            best_iou = iou
                            best_anchor = anchor
                    
                    if best_anchor is not None:
                        anchor_w, anchor_h = best_anchor
                        anchor_w_px = int(anchor_w * self.target_size[1])
                        anchor_h_px = int(anchor_h * self.target_size[0])
                        x_center_px = int(x_center * self.target_size[1])
                        y_center_px = int(y_center * self.target_size[0])
                        
                        rect_anchor = patches.Rectangle(
                            (x_center_px - anchor_w_px // 2, y_center_px - anchor_h_px // 2),
                            anchor_w_px, anchor_h_px,
                            linewidth=1,
                            edgecolor='blue',
                            facecolor='none',
                            linestyle='--'
                        )
                        ax2.add_patch(rect_anchor)
            
            plt.tight_layout()
            plt.show()
            
            # Print statistics
            print("\nStatistics:")
            print(f"Number of objects: {len(boxes)}")
            
            # Count objects per class
            for class_id in np.unique(class_labels):
                count = np.sum(class_labels == class_id)
                print(f"- Class {int(class_id)}: {count} objects")
            
            if len(boxes) > 0:
                boxes = np.array(boxes)
                widths = boxes[:, 2]
                heights = boxes[:, 3]
                print("\nBox dimensions (normalized):")
                print(f"Width  - min: {widths.min():.3f}, max: {widths.max():.3f}, mean: {widths.mean():.3f}")
                print(f"Height - min: {heights.min():.3f}, max: {heights.max():.3f}, mean: {heights.mean():.3f}")
                
        except Exception as e:
            self.__logger.error(f"Error visualizing sample: {e}")