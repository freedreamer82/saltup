import numpy as np
import cv2

from typing import Tuple
import numpy as np
import os
from glob import glob

from saltup.ai.object_detection.dataset.base_dataset import BaseDataloader
from saltup.ai.object_detection.datagenerator.base_datagen import BasedDatagenerator
from saltup.utils.data.image.image_utils import Image
from saltup.utils.configure_logging import get_logger


class ClassificationDataloader(BaseDataloader):
    def __init__(self, root_dir:str, classes_dict:dict={}, img_size:Tuple=(224, 224, 3), extensions=('jpg', 'jpeg', 'png')):
        """
        Args:
            root_dir (str): Root directory containing subfolders per class
            img_size (tuple): Image size (H, W, C) for preallocation (optional)
            extensions (tuple): Allowed image extensions
        """
        self.root_dir = root_dir
        self.img_size = img_size
        self.extensions = extensions
        
        self.image_paths = []
        self.labels = []
        if classes_dict:
            self.class_to_idx = classes_dict
            self.idx_to_class = {idx: class_name for class_name, idx in classes_dict.items()}
        else:
            self.class_to_idx = {}
            self.idx_to_class = {}
        
        self._load_dataset()

    def _load_dataset(self):
        if self.class_to_idx:
            classes = list(self.class_to_idx.keys())
        else:
            classes = sorted(os.listdir(self.root_dir))
            self.class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
            self.idx_to_class = {idx: class_name for class_name, idx in self.class_to_idx.items()}
        
        for class_name in classes:
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for ext in self.extensions:
                files = glob(os.path.join(class_dir, f'*.{ext}'))
                self.image_paths.extend(files)
                self.labels.extend([self.class_to_idx[class_name]] * len(files))
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image(img_path)
        img = image.get_data()
        
        return img, label
    
    def get_num_classes(self):
        return len(set(self.labels))
    def __len__(self):
        return len(self.image_paths)
    
    def __iter__(self):
        self._iter_idx = 0
        return self
    
    def __next__(self):
        if self._iter_idx >= len(self):
            raise StopIteration
        item = self.__getitem__(self._iter_idx)
        self._iter_idx += 1
        return item

from tensorflow.keras.utils import to_categorical, Sequence
class keras_ClassificationDataGenerator(BasedDatagenerator, Sequence):
    def __init__(
        self,
        dataloader,
        target_size,
        num_classes,
        batch_size=1,
        preprocess=None,
        transform=None,
        seed=None
    ):
        super().__init__(dataloader, target_size, num_classes, batch_size, preprocess, transform, seed)
        self.on_epoch_end()
    
    def __len__(self):
        # Number of batches per epoch
        return int(np.ceil(len(self.dataloader) / self.batch_size))
    
    def __iter__(self):
        self.current_idx = 0
        return self
    
    def __next__(self):
        if self.current_idx >= len(self):
            self.on_epoch_end()
            raise StopIteration
        batch = self.__getitem__(self.current_idx)
        self.current_idx += 1
        return batch
    
    def __getitem__(self, idx):
        # Get batch indexes
        batch_indexes = self._indexes[idx * self.batch_size: (idx + 1) * self.batch_size]
        
        images = []
        labels = []
        
        for i in batch_indexes:
            img, label = self.dataloader[i]
            
            # Apply preprocessing
            if self.preprocess:
                img = self.preprocess(img, target_size=self.target_size)
            
            # Apply augmentations
            if self.do_augment and self.transform:
                augmented = self.transform(image=img)
                img = augmented['image']
            
            # One-hot encode the label
            label = to_categorical(label, num_classes=self.num_classes)
            
            images.append(img)
            labels.append(label)
        
        images = np.stack(images, axis=0)
        labels = np.stack(labels, axis=0)
        
        return images, labels
    
    def on_epoch_end(self):
        # Shuffle indexes at the end of each epoch
        self._rng.shuffle(self._indexes)

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
    
class pytorch_ClassificationDataGenerator(BasedDatagenerator, Dataset):
    def __init__(
        self,
        dataloader,
        target_size,
        num_classes,
        batch_size=1,
        preprocess=None,
        transform=None,
        seed=None
    ):
        self.dataloader = dataloader
        self.target_size = target_size
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.preprocess = preprocess
        self.transform = transform
        #self.do_augment = True if transform else False
        
        # Set random seed for reproducibility
        self._rng = np.random.RandomState(seed if seed is not None else 42)
        
        # Create indexes for all samples
        self._indexes = np.arange(len(self.dataloader))
        self._rng.shuffle(self._indexes)
    
    def on_epoch_end(self):
        # Shuffle indexes at the end of each epoch
        self._rng.shuffle(self._indexes)
        
    def __len__(self):
        # Number of batches per epoch
        return len(self.dataloader)
    
    def __iter__(self):
        self.current_idx = 0
        return self
    
    def __next__(self):
        if self.current_idx >= len(self):
            self.on_epoch_end()
            raise StopIteration
        batch = self.__getitem__(self.current_idx)
        self.current_idx += 1
        return batch

    def __getitem__(self, idx):
        """
        Get a single sample as PyTorch tensors.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (image, label) as PyTorch tensors
        """
        # Use the shuffled index
        actual_idx = self._indexes[idx]
        
        # Get the image and label
        img, label = self.dataloader[actual_idx]
        
        # Apply preprocessing
        if self.preprocess:
            img = self.preprocess(img, target_size=self.target_size)
        
        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']
        
        # Convert label to one-hot encoding
        label_tensor = F.one_hot(torch.tensor(label), num_classes=self.num_classes).float()
        
        # Convert image to torch tensor with proper channel ordering (C, H, W)
        img = np.transpose(img, (2, 0, 1))
        img_tensor = torch.tensor(img, dtype=torch.float32)
        
        return img_tensor, label_tensor
    
    def on_epoch_end(self):
        # Shuffle indexes at the end of each epoch
        self._rng.shuffle(self._indexes)        
    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function for PyTorch DataLoader.
        
        Handles batching of samples into a single tensor.
        
        Args:
            batch: List of (image, label) tuples from __getitem__
            
        Returns:
            Tuple of:
            - images tensor of shape (batch_size, C, H, W)
            - labels tensor of shape (batch_size, 1)
        """
        images, labels = zip(*batch)
        
        # Stack images and labels
        images = torch.stack(images)
        labels = torch.stack(labels)
        
        return images, labels
