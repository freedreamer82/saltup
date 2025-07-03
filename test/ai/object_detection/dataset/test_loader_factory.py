import json
import pytest
from pathlib import Path

from saltup.ai.object_detection.dataset.loader_factory import DataLoaderFactory
from saltup.ai.object_detection.dataset.coco import COCOLoader, create_dataset_structure as create_coco_structure
from saltup.ai.object_detection.dataset.pascal_voc import PascalVOCLoader, create_dataset_structure as create_pascal_voc_structure
from saltup.ai.object_detection.dataset.yolo_darknet import YoloDarknetLoader, create_dataset_structure as create_yolo_darknet_structure


class TestDataLoaderFactory:

    @pytest.fixture
    def sample_coco_data(self, tmp_path):
        """Fixture to create a sample COCO dataset."""
        root_dir = tmp_path / "coco_dataset"
        root_dir.mkdir()

        # Create dataset structure using module function
        directories = create_coco_structure(str(root_dir))

        # Ensure annotations directory exists
        Path(directories['annotations']).mkdir(parents=True, exist_ok=True)

        # Ensure train images directory exists
        Path(directories['images']['train']).mkdir(parents=True, exist_ok=True)

        # Create sample COCO annotations
        annotation_file = Path(directories['annotations']) / "instances_train.json"
        coco_annotations = {
            "images": [
                {"id": 1, "file_name": "img1.jpg", "width": 640, "height": 480},
                {"id": 2, "file_name": "img2.jpg", "width": 640, "height": 480}
            ],
            "annotations": [
                {"image_id": 1, "bbox": [100, 100, 50, 50], "category_id": 1},
                {"image_id": 2, "bbox": [200, 200, 60, 60], "category_id": 2}
            ],
            "categories": [
                {"id": 1, "name": "category1"},
                {"id": 2, "name": "category2"}
            ]
        }
        with open(annotation_file, "w") as f:
            json.dump(coco_annotations, f)

        # Create corresponding image files
        train_images_dir = Path(directories['images']['train'])
        for img in coco_annotations["images"]:
            (train_images_dir / img["file_name"]).touch()

        return str(root_dir), str(Path(directories['images']['train'])), str(annotation_file)

    def test_loader_factory_coco(self, sample_coco_data):
        """Test DataLoaderFactory for COCO dataset."""
        root_dir, train_images_dir, annotation_file = sample_coco_data

        train_loader, val_loader, test_loader = DataLoaderFactory.create(root_dir)
        assert isinstance(train_loader, COCOLoader)
        assert val_loader is None
        assert test_loader is None

    def test_loader_factory_pascal_voc(self, tmp_path):
        """Test DataLoaderFactory for Pascal VOC dataset."""
        root_dir = tmp_path / "pascal_voc_dataset"
        root_dir.mkdir()

        # Create dataset structure using module function
        directories = create_pascal_voc_structure(str(root_dir))

        # Create sample Pascal VOC annotations
        annotation_file = Path(directories['annotations']['train']) / "img1.xml"
        with open(annotation_file, "w") as f:
            f.write("""<annotation><object><name>category1</name></object></annotation>""")

        # Create corresponding image files
        train_images_dir = Path(directories['images']['train'])
        (train_images_dir / "img1.jpg").touch()

        train_loader, val_loader, test_loader = DataLoaderFactory.create(root_dir)
        assert isinstance(train_loader, PascalVOCLoader)
        assert val_loader is None
        assert test_loader is None

    def test_loader_factory_yolo_darknet(self, tmp_path):
        """Test DataLoaderFactory for YOLO Darknet dataset."""
        root_dir = tmp_path / "yolo_darknet_dataset"
        root_dir.mkdir()

        # Create dataset structure using module function
        directories = create_yolo_darknet_structure(str(root_dir))

        # Create sample YOLO Darknet annotations
        annotation_file = Path(directories['labels']['train']) / "img1.txt"
        with open(annotation_file, "w") as f:
            f.write("""0 0.5 0.5 0.2 0.2""")

        # Create corresponding image files
        train_images_dir = Path(directories['images']['train'])
        (train_images_dir / "img1.jpg").touch()

        train_loader, val_loader, test_loader = DataLoaderFactory.create(root_dir)
        assert isinstance(train_loader, YoloDarknetLoader)
        assert val_loader is None
        assert test_loader is None

    def test_loader_factory_invalid_type(self):
        """Test DataLoaderFactory with invalid dataset type."""
        with pytest.raises(ValueError):
            DataLoaderFactory.create("InvalidType")

    def test_loader_factory_dataloader_names(self, sample_coco_data, tmp_path):
        """Test that dataloader names are correctly set by the factory."""
        root_dir, train_images_dir, annotation_file = sample_coco_data
        train_loader, val_loader, test_loader = DataLoaderFactory.create(root_dir)
        assert train_loader.get_name() == "Train COCO Dataloader"
        assert val_loader is None
        assert test_loader is None
