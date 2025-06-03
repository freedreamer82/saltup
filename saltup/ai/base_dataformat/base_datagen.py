from abc import ABC, abstractmethod
from typing import Tuple, List
import albumentations as A
import numpy as np

from saltup.ai.base_dataformat.base_dataset import BaseDataloader

class BaseDatagenerator(ABC):
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

    @abstractmethod
    def split(self, ratio):
        raise NotImplementedError
    
    @abstractmethod
    def merge(dg1:'BaseDatagenerator', dg2:'BaseDatagenerator') -> 'BaseDatagenerator':
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
    
class kfoldGenerator():
    def __init__(self, list_of_datagenerator: List[BaseDatagenerator], idx: int):
        """
        Initialize the k-fold generator.

        Args:
            list_of_datagenerator (List[BaseDatagenerator]): List of datagenerators for each fold.
            idx (int): Index of the fold to use for validation.
        """
        self.folds = list_of_datagenerator
        self.val_idx = idx
        self.train_generator = self._merge_folds(exclude_idx=idx)
        self.val_generator = self.folds[idx]

    def _merge_folds(self, exclude_idx: int) -> BaseDatagenerator:
        """
        Merge all folds except the one at exclude_idx using the provided static `merge` method.

        Args:
            exclude_idx (int): The index of the fold to exclude from the training set.

        Returns:
            BaseDatagenerator: A merged generator for training.
        """
        # Filter out the fold to be used for validation
        train_folds = [fold for i, fold in enumerate(self.folds) if i != exclude_idx]
        
        # Merge iteratively using the provided static method
        merged_data = train_folds[0]
        datagen_cls = merged_data.__class__
        for fold in train_folds[1:]:
            merged_data = datagen_cls.merge(merged_data, fold)

        return merged_data

    def get_fold_generators(self):
        """
        Return the training and validation generators for the current fold.

        Returns:
            Tuple[BaseDatagenerator, BaseDatagenerator]: (train_generator, val_generator)
        """
        return self.train_generator, self.val_generator