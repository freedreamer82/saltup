import numpy as np
import cv2

from typing import Tuple, Union,List
import numpy as np
import os
import random
from glob import glob
from tensorflow.keras.utils import to_categorical, Sequence

from saltup.ai.base_dataformat.base_dataset import BaseDataloader
from saltup.ai.base_dataformat.base_datagen import BaseDatagenerator
from saltup.utils.data.image.image_utils import Image
from saltup.utils.configure_logging import get_logger


class ClassificationDataloader(BaseDataloader):
    def __init__(self, source:Union[str,List[List]], classes_dict:dict={}, img_size:Tuple=(), extensions:str='jpg'):
        """
        Args:
            source (Union[str,List[List]]): Root directory containing subfolders per class or a list of image paths and labels.
            classes_dict (dict): Dictionary mapping class names to indices (optional)
            img_size (tuple): Image size (H, W, C) for preallocation (optional)
            extensions (tuple): Allowed image extensions
        """
        
        if isinstance(source, str):
           self.root_dir = source
        
        self.img_size = img_size
        self.extensions = extensions
        
        if isinstance(source, list):
            # If source is a list, assume it contains image paths and labels
            self.image_paths = source[0]
            self.labels = source[1]
            self.root_dir = None
        else:
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
        if len(self.image_paths) == 0:
            # Load images and labels from directories
            for class_name in classes:
                class_dir = os.path.join(self.root_dir, class_name)
                if not os.path.isdir(class_dir):
                    continue
                files = glob(os.path.join(class_dir, f'*.{self.extensions}'))
                self.image_paths.extend(files)
                self.labels.extend([self.class_to_idx[class_name]] * len(files))
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image(img_path)
        img = image.get_data()
        
        return img, label
    def get_num_samples_per_class(self):
        """
        Get the number of samples per class in the dataset.

        Returns:
            dict: Dictionary mapping class names to number of samples
        """
        num_samples = {class_name: 0 for class_name in self.class_to_idx.keys()}
        for label in self.labels:
            class_name = self.idx_to_class[label]
            num_samples[class_name] += 1
        return num_samples
    
    def get_image_paths(self):
        return self.image_paths
    
    def get_labels(self):
        return self.labels

    def split(self, ratios:List[float]=[0.2, 0.8]) -> List['ClassificationDataloader']:
        """
        Split the current dataset into two ClassificationDataloader instances.

        Args:
            split_ratio (float): Ratio of the first split dataset size (e.g., 0.8 means 80% train, 20% test)

        Returns:
            (ClassificationDataloader, ClassificationDataloader): Tuple of two new dataloaders
        """
        
        if not np.isclose(sum(ratios), 1.0):
            raise ValueError("Ratios must sum to 1.0")
        
        list_output = []
        list_image_paths = self.get_image_paths()
        list_labels = self.get_labels()
        # Shuffle image paths and labels together
        combined = list(zip(list_image_paths, list_labels))
        random.shuffle(combined)
        list_image_paths, list_labels = zip(*combined)
        list_image_paths = list(list_image_paths)
        list_labels = list(list_labels)
        
        total_samples = len(list_image_paths)
        start = 0
        for ratio in ratios:
            end = start + int(ratio * total_samples)
            current_list_image_paths = list_image_paths[start:end]
            current_list_labels = list_labels[start:end]
            current_dataloader = ClassificationDataloader(
                source=[current_list_image_paths, current_list_labels],
                classes_dict=self.get_classes(),
                img_size=self.get_img_size(),
                extensions=self.get_extensions()
            )
            start = end
            list_output.append(current_dataloader)
            
        return list_output
    @staticmethod
    def merge(dl1, dl2) -> 'ClassificationDataloader':
        """
        Merge two ClassificationDataloader instances into one combined instance.

        Args:
            dl1 (ClassificationDataloader): first dataloader
            dl2 (ClassificationDataloader): second dataloader

        Returns:
            ClassificationDataloader: merged dataloader
        """
        # Check that classes match
        if dl1.get_classes() != dl2.get_classes():
            raise ValueError("Class dictionaries do not match and cannot be merged.")
        
        list_images_paths = dl1.get_image_paths() + dl2.get_image_paths()
        list_labels = dl1.get_labels() + dl2.get_labels()
        
        merged = ClassificationDataloader(
            source=[list_images_paths, list_labels],
            classes_dict=dl1.get_classes(),
            img_size=dl1.get_img_size(),
            extensions=dl1.get_extensions()
        )

        return merged
    
    def get_extensions(self):
        return self.extensions
    
    def get_img_size(self):
        return self.img_size
        
    def get_num_classes(self):
        return len(set(self.labels))
    
    def get_classes(self):
        return self.class_to_idx
        
    def get_idx_to_class(self):
        return self.idx_to_class
        
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

class keras_ClassificationDataGenerator(BaseDatagenerator, Sequence):
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
    
    def split(self, ratios:List[float]=[0.2, 0.8]) -> List['keras_ClassificationDataGenerator']:
        list_output = []
        list_dataloaders = self.dataloader.split(ratios)
        for dl in list_dataloaders:
            current_datagen = keras_ClassificationDataGenerator(
                dataloader=dl,
                target_size=self.target_size,
                num_classes=self.num_classes,
                batch_size=self.batch_size,
                preprocess=self.preprocess,
                transform=self.transform
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
            preprocess=dg1.preprocess,
            transform=dg1.transform
        )
        return current_datagen
    
    
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
    
class pytorch_ClassificationDataGenerator(BaseDatagenerator, Dataset):
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
    
    def split(self, ratios:List[float]=[0.2, 0.8]):
        list_output = []
        list_dataloaders = self.dataloader.split(ratios)
        for dl in list_dataloaders:
            current_datagen = pytorch_ClassificationDataGenerator(
                dataloader=dl,
                target_size=self.target_size,
                num_classes=self.num_classes,
                batch_size=self.batch_size,
                preprocess=self.preprocess,
                transform=self.transform
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
            preprocess=dg1.preprocess,
            transform=dg1.transform
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