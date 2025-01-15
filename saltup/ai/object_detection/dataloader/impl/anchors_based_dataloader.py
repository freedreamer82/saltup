from typing import Tuple
import albumentations as A
import numpy as np
import os

from saltup.ai.object_detection.dataset.base_dataset_loader import BaseDatasetLoader
from saltup.ai.object_detection.preprocessing.impl.anchors_based_preprocess import AnchorsBasedPreprocess
from saltup.ai.object_detection.utils.anchor_based_model import convert_to_grid_format
from saltup.ai.object_detection.utils.bbox import compute_iou
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
    