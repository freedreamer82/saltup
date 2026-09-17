

from __future__ import annotations

from typing import List
import numpy as np

from saltup.ai.base_dataformat.base_datagen import BaseDatagenerator
from saltup.ai.classification.dataloader import ClassificationDataloader
from saltup.utils.data.image.image_utils import Image

# Optional framework wrappers.
# Keras and PyTorch are optional dependencies (``saltup[keras]`` / ``saltup[torch]``):
# the generators below are always importable, but instantiating one without its
# framework installed raises an ImportError explaining what to install.
try:
    from keras.utils import Sequence, to_categorical  # type: ignore
    _KERAS_AVAILABLE = True
except ImportError:
    _KERAS_AVAILABLE = False
    to_categorical = None

    class Sequence:  # minimal stand-in so the class below can still be defined
        pass


class keras_ClassificationDataGenerator(BaseDatagenerator, Sequence):
    def __init__(
        self,
        dataloader,
        target_size,
        num_classes,
        batch_size=1,
        preprocess=None,
        transform=None,
        apply_padding=False,
        seed=None
    ):
        if not _KERAS_AVAILABLE:
            raise ImportError(
                "keras_ClassificationDataGenerator requires Keras. "
                'Install it with: pip install "saltup[keras]"'
            )
        super().__init__(dataloader, target_size, num_classes, batch_size, preprocess, transform, apply_padding, seed)
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
    
    def split(self, ratios:List[float]=[0.2, 0.8]) -> List['keras_ClassificationDataGenerator']:
        list_output = []
        list_dataloaders = self.dataloader.split(ratios)
        for dl in list_dataloaders:
            current_datagen = keras_ClassificationDataGenerator(
                dataloader=dl,
                target_size=self.target_size,
                num_classes=self.num_classes,
                batch_size=self.batch_size,
                preprocess=self._preprocess,
                transform=self.transform,
                apply_padding=self.apply_padding
            )
            list_output.append(current_datagen)
        return list_output
    
    @staticmethod
    def merge(dg1, dg2) -> 'keras_ClassificationDataGenerator':
        dl_merged = ClassificationDataloader.merge(dg1.dataloader, dg2.dataloader)
        current_datagen = keras_ClassificationDataGenerator(
            dataloader=dl_merged,
            target_size=dg1.target_size,
            num_classes=dg1.num_classes,
            batch_size=dg1.batch_size,
            preprocess=dg1._preprocess,
            transform=dg1.transform,
            apply_padding=dg1.apply_padding and dg2.apply_padding   #apply padding only if both datagens have it enabled
        )
        return current_datagen
    
    def __getitem__(self, idx):
        # Get batch indexes
        batch_indexes = self._indexes[idx * self.batch_size: (idx + 1) * self.batch_size]
        
        images = []
        labels = []
        
        for i in batch_indexes:
            path, img, label = self.dataloader[i]
            
            # Apply preprocessing
            if self._preprocess:
                img = self._preprocess(img, self.target_height, self.target_width, apply_padding=self.apply_padding)

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


try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import Dataset
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    F = None
    _TORCH_AVAILABLE = False

    class Dataset:  # minimal stand-in so the class below can still be defined
        pass


class pytorch_ClassificationDataGenerator(BaseDatagenerator, Dataset):
    def __init__(
        self,
        dataloader,
        target_size,
        num_classes,
        batch_size=1,
        preprocess=None,
        transform=None,
        apply_padding=False,
        seed=None
    ):
        
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "pytorch_ClassificationDataGenerator requires PyTorch. "
                'Install it with: pip install "saltup[torch]"'
            )
        super().__init__(dataloader, target_size, num_classes, batch_size, preprocess, transform, apply_padding, seed)
        #self.do_augment = True if transform else False
        
        # Set random seed for reproducibility
        self._rng = np.random.RandomState(seed if seed is not None else 42)
        
        # Create indexes for all samples
        self._indexes = np.arange(len(self.dataloader))
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
    
    def split(self, ratios:List[float]=[0.2, 0.8]):
        list_output = []
        list_dataloaders = self.dataloader.split(ratios)
        for dl in list_dataloaders:
            current_datagen = pytorch_ClassificationDataGenerator(
                dataloader=dl,
                target_size=self.target_size,
                num_classes=self.num_classes,
                batch_size=self.batch_size,
                preprocess=self._preprocess,
                transform=self.transform,
                apply_padding=self.apply_padding
            )
            list_output.append(current_datagen)
        return list_output
    
    @staticmethod
    def merge(dg1, dg2):
        dl_merged = ClassificationDataloader.merge(dg1.dataloader, dg2.dataloader)
        current_datagen = pytorch_ClassificationDataGenerator(
            dataloader=dl_merged,
            target_size=dg1.target_size,
            num_classes=dg1.num_classes,
            batch_size=dg1.batch_size,
            preprocess=dg1._preprocess,
            transform=dg1.transform,
            apply_padding=dg1.apply_padding and dg2.apply_padding   #apply padding only if both datagens have it enabled
        )
        return current_datagen
        
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
        path, img, label = self.dataloader[actual_idx]
        
        # Apply preprocessing
        if self._preprocess:
            img = self._preprocess(img, target_height=self.target_height, target_width=self.target_width, apply_padding=self.apply_padding)
        
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