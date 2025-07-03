from pathlib import Path

from saltup.ai.object_detection.dataset.yolo_darknet import (
    is_yolo_darknet_dataset,
    get_dataset_paths as get_yolo_paths,
    YoloDarknetLoader,
)
from saltup.ai.object_detection.dataset.coco import (
    is_coco_dataset,
    get_dataset_paths as get_coco_paths,
    COCOLoader,
)
from saltup.ai.object_detection.dataset.pascal_voc import (
    is_pascal_voc_dataset,
    get_dataset_paths as get_voc_paths,
    PascalVOCLoader,
)


class DataLoaderFactory:
    """
    Factory class to automatically detect the dataset format (YOLO, COCO, VOC)
    and instantiate the appropriate dataloader(s) for train, val, and test splits.

    Usage:
        train_loader, val_loader, test_loader = DataLoaderFactory.create(root_dir)
    """

    @staticmethod
    def create(root_dir, *args, **kwargs):
        """
        Detects the dataset type in the given root directory and returns the corresponding
        dataloaders for train, val, and test splits.

        Args:
            root_dir (str or Path): Path to the dataset root directory.
            *args, **kwargs: Additional arguments passed to the dataloader constructors.

        Returns:
            tuple: (train_dataloader, val_dataloader, test_dataloader)

        Raises:
            ValueError: If the dataset type is not supported or the path is invalid.
        """
        if not isinstance(root_dir, Path):
            root_dir = Path(root_dir)
        if not root_dir.is_dir():
            raise ValueError(f"The provided path is not a directory: {root_dir}")

        train_dataloader = None
        val_dataloader = None
        test_dataloader = None

        print(f"Detecting dataset format in: {root_dir}")

        if is_yolo_darknet_dataset(root_dir):
            print("Detected YOLO Darknet dataset format.")
            (
            train_images_dir, 
            train_labels_dir, 
            val_images_dir, 
            val_labels_dir, 
            test_images_dir, 
            test_labels_dir
            ) = get_yolo_paths(root_dir)

            if all(x is None for x in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir, test_images_dir, test_labels_dir]):
                print("Warning: All YOLO split directories are None. The dataset may not follow the expected structure. Please create dataloaders manually.")
                return None, None, None
            
            if train_images_dir and train_labels_dir:
                print(f"Creating YOLO train dataloader: {train_images_dir}, {train_labels_dir}")
                train_dataloader = YoloDarknetLoader(train_images_dir, train_labels_dir, *args, **kwargs)
                train_dataloader.set_name("Train YOLO Dataloader")

            if val_images_dir and val_labels_dir:
                print(f"Creating YOLO val dataloader: {val_images_dir}, {val_labels_dir}")
                val_dataloader = YoloDarknetLoader(val_images_dir, val_labels_dir, *args, **kwargs)
                val_dataloader.set_name("Validation YOLO Dataloader")

            if test_images_dir and test_labels_dir:
                print(f"Creating YOLO test dataloader: {test_images_dir}, {test_labels_dir}")
                test_dataloader = YoloDarknetLoader(test_images_dir, test_labels_dir, *args, **kwargs)
                test_dataloader.set_name("Test YOLO Dataloader")

        elif is_coco_dataset(root_dir):
            print("Detected COCO dataset format.")
            (
            train_images_dir, 
            train_labels_file, 
            val_images_dir, 
            val_labels_file, 
            test_images_dir, 
            test_labels_file
            ) = get_coco_paths(root_dir)

            if all(x is None for x in [train_images_dir, train_labels_file, val_images_dir, val_labels_file, test_images_dir, test_labels_file]):
                print("Warning: All COCO split directories/files are None. The dataset may not follow the expected structure. Please create dataloaders manually.")
                return None, None, None
            
            if train_images_dir and train_labels_file:
                print(f"Creating COCO train dataloader: {train_images_dir}, {train_labels_file}")
                train_dataloader = COCOLoader(train_images_dir, train_labels_file, *args, **kwargs)
                train_dataloader.set_name("Train COCO Dataloader")

            if val_images_dir and val_labels_file:
                print(f"Creating COCO val dataloader: {val_images_dir}, {val_labels_file}")
                val_dataloader = COCOLoader(val_images_dir, val_labels_file, *args, **kwargs)
                val_dataloader.set_name("Validation COCO Dataloader")

            if test_images_dir and test_labels_file:
                print(f"Creating COCO test dataloader: {test_images_dir}, {test_labels_file}")
                test_dataloader = COCOLoader(test_images_dir, test_labels_file, *args, **kwargs)
                test_dataloader.set_name("Test COCO Dataloader")

        elif is_pascal_voc_dataset(root_dir):
            print("Detected Pascal VOC dataset format.")
            (
            train_images_dir, 
            train_labels_dir, 
            val_images_dir, 
            val_labels_dir, 
            test_images_dir, 
            test_labels_dir
            ) = get_voc_paths(root_dir)

            if all(x is None for x in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir, test_images_dir, test_labels_dir]):
                print("Warning: All VOC split directories are None. The dataset may not follow the expected structure. Please create dataloaders manually.")
                return None, None, None
            
            if train_images_dir and train_labels_dir:
                print(f"Creating VOC train dataloader: {train_images_dir}, {train_labels_dir}")
                train_dataloader = PascalVOCLoader(train_images_dir, train_labels_dir, *args, **kwargs)
                train_dataloader.set_name("Train VOC Dataloader")

            if val_images_dir and val_labels_dir:
                print(f"Creating VOC val dataloader: {val_images_dir}, {val_labels_dir}")
                val_dataloader = PascalVOCLoader(val_images_dir, val_labels_dir, *args, **kwargs)
                val_dataloader.set_name("Validation VOC Dataloader")

            if test_images_dir and test_labels_dir:
                print(f"Creating VOC test dataloader: {test_images_dir}, {test_labels_dir}")
                test_dataloader = PascalVOCLoader(test_images_dir, test_labels_dir, *args, **kwargs)
                test_dataloader.set_name("Test VOC Dataloader")

        else:
            raise ValueError("Unsupported or unknown dataset type in directory: {}".format(root_dir))

        return train_dataloader, val_dataloader, test_dataloader
