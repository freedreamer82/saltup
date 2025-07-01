from pathlib import Path
from enum import IntEnum

class DatasetFormat(IntEnum):
    # Aligned with common notations in BBoxFormat
    YOLO = 2
    COCO = 6
    VOC = 4

    @classmethod
    def from_string(cls, value: str) -> 'DatasetFormat':
        """
        Convert a string representation of the YoloType to its corresponding enum value.

        Args:
            value: The string representation of the YoloType (case-insensitive).

        Returns:
            The corresponding YoloType enum value.

        Raises:
            ValueError: If the string does not match any YoloType.
        """
        # Case-insensitive mapping
        value_upper = value.upper()
        for enum_value in cls:
            if value_upper == enum_value.name.upper():
                return enum_value
        raise ValueError(f"Invalid YoloType string: {value}. Valid options are: {', '.join([e.name for e in cls])}")


class DataLoaderFactory:
        
    @staticmethod
    def create(dir, *args, **kwargs):
        
        if not isinstance(dir, Path):
            dir = Path(dir)
        if not dir.is_dir():
            raise ValueError(f"The provided path is not a directory: {dir}")
        
        exts = ('*.jpg', '*.jpeg', '*.png')
        imgs_files: list[Path] = []
        for ext in exts:
            imgs_files.extend(dir.rglob(ext))
        if not imgs_files:
            raise ValueError(f"No images found in directory: {dir}")
    
        print(f"Found {len(imgs_files)} images in directory: {dir}")
        
        exts2type = {
            '*.txt': DatasetFormat.YOLO,
            '*.json': DatasetFormat.COCO,
            '*.xml': DatasetFormat.VOC
        }
        
        for ext in exts2type.keys():
            lbl_files = list(dir.rglob(ext))
            if lbl_files:
                dataset_type = exts2type[ext]
                print(f"Found {len(lbl_files)} label files of type {dataset_type.name} in directory: {dir}")
                break
        
        if dataset_type == DatasetFormat.YOLO or dataset_type == DatasetFormat.VOC:
            if len(lbl_files) != len(imgs_files):
                raise ValueError(f"Mismatch between number of images ({len(imgs_files)}) and labels ({len(lbl_files)}) in directory: {dir}")
        elif dataset_type == DatasetFormat.COCO:
            if len(lbl_files) != 1:
                raise ValueError(f"Expected exactly one COCO JSON file, found {len(lbl_files)} in directory: {dir}")
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
        
        image_dirs = set()
        for img_file in imgs_files:
            image_dirs.add(img_file.parent.absolute())
            
        lbls_dirs = set()
        for lbl_file in lbl_files:
            lbls_dirs.add(lbl_file.parent.absolute())
        
        if dataset_type != DatasetFormat.COCO:
            if len(image_dirs) != len(lbls_dirs):
                raise ValueError("Mismatch between number of image directories and label directories.")
        
            image_dirs = sorted(image_dirs)
            lbls_dirs = sorted(lbls_dirs)
            
            for img_dir, lbl_dir in zip(image_dirs, lbls_dirs):
                print(img_dir)
                print(lbl_dir)
                pass

        
        
        
        if dataset_type == DatasetFormat.YOLO:
            from saltup.ai.object_detection.dataset.yolo_darknet import YoloDarknetLoader
            pass
        elif dataset_type == DatasetFormat.COCO:
            from saltup.ai.object_detection.dataset.coco import COCOLoader
            pass
        elif dataset_type == DatasetFormat.VOC:
            from saltup.ai.object_detection.dataset.pascal_voc import PascalVOCLoader
            pass
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")