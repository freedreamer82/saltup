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
        Split the current dataset into multiple ClassificationDataloader instances.

        Args:
            ratios (List[float]): Ratios for each split (must sum to 1.0)

        Returns:
            List[ClassificationDataloader]: List of dataloaders for each split
        """
        
        if not np.isclose(sum(ratios), 1.0):
            raise ValueError("Ratios must sum to 1.0")
        
        list_image_paths = self.get_image_paths()
        list_labels = self.get_labels()

        # if stratified:
        # Stratified split - maintains class distribution
        return self._stratified_split(list_image_paths, list_labels, ratios)
        # else:
        #     # Random split - may result in unbalanced class distribution
        #     return self._random_split(list_image_paths, list_labels, ratios)
    
    def _stratified_split(self, image_paths, labels, ratios):
        """
        Perform stratified split to maintain class distribution across splits.
        """
        # Group samples by class
        class_indices = {}
        for idx, label in enumerate(labels):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
        
        # Shuffle indices within each class
        for label in class_indices:
            random.shuffle(class_indices[label])
        
        # Initialize split containers
        num_splits = len(ratios)
        split_image_paths = [[] for _ in range(num_splits)]
        split_labels = [[] for _ in range(num_splits)]
        
        # For each class, distribute samples according to ratios
        for label, indices in class_indices.items():
            class_size = len(indices)
            start = 0
            
            for split_idx, ratio in enumerate(ratios):
                if split_idx == len(ratios) - 1:
                    # Last split gets all remaining samples to handle rounding
                    end = class_size
                else:
                    end = start + int(ratio * class_size)
                
                # Add samples from this class to current split
                for idx in indices[start:end]:
                    split_image_paths[split_idx].append(image_paths[idx])
                    split_labels[split_idx].append(labels[idx])
                
                start = end
        
        # Create dataloaders for each split
        list_output = []
        for split_idx in range(num_splits):
            # Shuffle the combined samples within each split
            combined = list(zip(split_image_paths[split_idx], split_labels[split_idx]))
            random.shuffle(combined)
            current_image_paths, current_labels = zip(*combined) if combined else ([], [])
            
            current_dataloader = ClassificationDataloader(
                source=[list(current_image_paths), list(current_labels)],
                classes_dict=self.get_classes(),
                img_size=self.get_img_size(),
                extensions=self.get_extensions()
            )
            list_output.append(current_dataloader)
        
        return list_output
    
    # def _random_split(self, image_paths, labels, ratios):
    #     """
    #     Perform random split (original behavior).
    #     """
    #     # Shuffle image paths and labels together
    #     combined = list(zip(image_paths, labels))
    #     random.shuffle(combined)
    #     image_paths, labels = zip(*combined)
    #     image_paths = list(image_paths)
    #     labels = list(labels)
        
    #     total_samples = len(image_paths)
    #     start = 0
    #     list_output = []
        
    #     for ratio in ratios:
    #         end = start + int(ratio * total_samples)
    #         current_image_paths = image_paths[start:end]
    #         current_labels = labels[start:end]
            
    #         current_dataloader = ClassificationDataloader(
    #             source=[current_image_paths, current_labels],
    #             classes_dict=self.get_classes(),
    #             img_size=self.get_img_size(),
    #             extensions=self.get_extensions()
    #         )
    #         start = end
    #         list_output.append(current_dataloader)
            
    #     return list_output
    
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
    
    def get_class_distribution(self) -> dict:
        """
        Get the class distribution in the dataset.
        
        Returns:
            dict: Dictionary with class names as keys and tuples (count, percentage) as values
        """
        total_samples = len(self.labels)
        if total_samples == 0:
            return {}
        
        class_counts = {}
        for label in self.labels:
            class_name = self.idx_to_class[label]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        class_distribution = {}
        for class_name, count in class_counts.items():
            percentage = (count / total_samples) * 100
            class_distribution[class_name] = (count, percentage)
        
        return class_distribution
    
    def print_class_distribution(self, title="Dataset"):
        """
        Print the class distribution in a readable format.
        
        Args:
            title (str): Title for the distribution report
        """
        distribution = self.get_class_distribution()
        print(f"\n{title} Class Distribution:")
        print(f"Total samples: {len(self.labels)}")
        print("-" * 40)
        
        for class_name, (count, percentage) in distribution.items():
            print(f"{class_name:20}: {count:6} samples ({percentage:5.1f}%)")
    
    @staticmethod
    def compare_distributions(dataloaders: List['ClassificationDataloader'], titles: List[str] = None):
        """
        Compare class distributions across multiple dataloaders.
        
        Args:
            dataloaders (List[ClassificationDataloader]): List of dataloaders to compare
            titles (List[str]): Optional titles for each dataloader
        """
        if titles is None:
            titles = [f"Split {i+1}" for i in range(len(dataloaders))]
        
        print("\n" + "="*60)
        print("CLASS DISTRIBUTION COMPARISON")
        print("="*60)
        
        for i, (dataloader, title) in enumerate(zip(dataloaders, titles)):
            dataloader.print_class_distribution(title)
            if i < len(dataloaders) - 1:
                print()

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
