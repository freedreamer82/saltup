from abc import ABC, abstractmethod
from typing import Tuple
import albumentations as A
import numpy as np

from saltup.ai.object_detection.dataset.base_dataset import BaseDataloader

class BasedDatagenerator(ABC):
    def __init__(
        self, 
        dataloader: BaseDataloader,
        target_size: Tuple[int, int], 
        num_classes: int,
        batch_size: int = 1,
        preprocess: callable = None,
        transform: A.Compose = None,
        seed: int = None
    ):
        """
        Initialize the dataloader.

        Args:
            dataloader: Base dataset loader providing image-label pairs
            target_size: Model input size as (height, width)
            num_classes: Number of object classes
            batch_size: Number of samples per batch
            preprocess: Optional custom preprocessing function
            transform: Optional albumentations transforms for augmentation
        """
        self.dataloader = dataloader
        self._indexes = np.arange(len(dataloader))
        if seed:
            self._rng = np.random.default_rng(seed)
            self._rng.shuffle(self._indexes)
        else:
            self._rng = np.random.default_rng()
        
        self.batch_size = batch_size
        self.target_size = target_size
        self.num_classes = num_classes
        
        self._transform = transform
        self._do_augment = True if transform else False
        
        self.preprocess = preprocess
    
    @abstractmethod
    def __len__(self):
        raise NotImplementedError
        
    def __iter__(self):
        raise NotImplementedError
    
    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError

    @property
    def transform(self):
        return self._transform
    
    @transform.setter
    def transform(self, value):
        self._transform = value
        self._do_augment = True if value else False
        
    @property
    def do_augment(self):
        return self._do_augment
