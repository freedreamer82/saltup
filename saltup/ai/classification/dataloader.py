from saltup.ai.base_dataformat.base_dataloader import BaseDataloader
from saltup.utils.data.image.image_utils import Image


from typing import List, Tuple, Union
from glob import glob
import os
import random
import numpy as np

class ClassificationDataloader(BaseDataloader):
    def __init__(self, source:Union[str,List[List]], classes_dict:dict={}, img_size:Tuple=(), extensions:str='jpg'):
        """
        Args:
            source (Union[str,List[List]]): Root directory containing subfolders per class or a list of image paths and labels.
            classes_dict (dict): Dictionary mapping class names to indices (optional)
            img_size (tuple): Image size (H, W, C) for preallocation (optional)
            extensions (tuple): Allowed image extensions
        """
        super().__init__()
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
