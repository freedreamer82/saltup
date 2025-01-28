import pytest
import os
import json
import numpy as np
from PIL import Image
from collections import defaultdict

from saltup.utils.data.image.image_utils import Image as SaltupImage
from saltup.ai.object_detection.utils.bbox import BBox, BBoxClassId
from saltup.ai.object_detection.dataset.coco import (
    create_dataset_structure, validate_dataset_structure,
    get_dataset_paths, read_annotations, write_annotations,
    replace_annotations_class, shift_class_ids,
    analyze_dataset, convert_coco_to_yolo_labels,
    split_dataset, split_and_organize_dataset,
    count_annotations, COCOLoader, ColorMode
)

class TestCOCODataset:
    @pytest.fixture
    def sample_coco_data(self):
        """Create sample COCO format data."""
        return {
            "images": [
                {"id": 1, "file_name": "img1.jpg", "width": 640, "height": 480},
                {"id": 2, "file_name": "img2.jpg", "width": 640, "height": 480},
                {"id": 3, "file_name": "img3.jpg", "width": 640, "height": 480}
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [100, 100, 50, 50], "area": 2500, "iscrowd": 0},
                {"id": 2, "image_id": 1, "category_id": 2, "bbox": [200, 200, 40, 60], "area": 2400, "iscrowd": 0},
                {"id": 3, "image_id": 2, "category_id": 1, "bbox": [150, 150, 45, 45], "area": 2025, "iscrowd": 0},
                {"id": 4, "image_id": 3, "category_id": 2, "bbox": [300, 300, 50, 50], "area": 2500, "iscrowd": 0}
            ],
            "categories": [
                {"id": 1, "name": "person", "supercategory": "none"},
                {"id": 2, "name": "car", "supercategory": "none"}
            ]
        }

    @pytest.fixture
    def dataset_dir(self, tmp_path):
        """Create a temporary dataset directory with COCO structure."""
        dataset_dir = tmp_path / "coco_dataset"
        dirs = create_dataset_structure(str(dataset_dir))
        return dataset_dir, dirs

    def test_create_dataset_structure(self, tmp_path):
        """Test creation of COCO dataset directory structure."""
        root_dir = tmp_path / "dataset"
        directories = create_dataset_structure(str(root_dir))
        
        # Verify all required directories exist
        assert os.path.exists(directories['images']['train'])
        assert os.path.exists(directories['images']['val'])
        assert os.path.exists(directories['images']['test'])
        assert os.path.exists(directories['annotations'])
        
        # Verify directory permissions
        assert os.access(directories['annotations'], os.W_OK)
        for split_dir in directories['images'].values():
            assert os.access(split_dir, os.W_OK)

    def test_validate_dataset_structure(self, dataset_dir, sample_coco_data):
        """Test validation of dataset structure with sample data."""
        root_dir, _ = dataset_dir
        
        # Create annotations file
        ann_file = root_dir / "annotations" / "instances_train.json"
        with open(ann_file, 'w') as f:
            json.dump(sample_coco_data, f)
        
        # Create corresponding images
        train_img_dir = root_dir / "images" / "train"
        for img in sample_coco_data['images']:
            (train_img_dir / img['file_name']).touch()
        
        stats = validate_dataset_structure(str(root_dir))
        
        assert stats['train']['images'] == 3
        assert stats['train']['annotations'] == 4
        assert stats['val']['images'] == 0
        assert stats['test']['images'] == 0
        
    def test_get_dataset_paths(self, dataset_dir):
        """Test get_dataset_paths() function."""
        root_dir, _ = dataset_dir

        # Create dataset structure
        create_dataset_structure(str(root_dir))

        # Create sample COCO annotations
        train_annotations = {
            "images": [{"id": 1, "file_name": "img1.jpg"}],
            "annotations": [{"id": 1, "image_id": 1}],
            "categories": [{"id": 1, "name": "test"}]
        }
        val_annotations = {
            "images": [{"id": 1, "file_name": "img1.jpg"}],
            "annotations": [{"id": 1, "image_id": 1}],
            "categories": [{"id": 1, "name": "test"}]
        }

        # Create annotation files
        train_ann_file = root_dir / "annotations" / "instances_train.json"
        val_ann_file = root_dir / "annotations" / "instances_val.json"
        with open(train_ann_file, 'w') as f:
            json.dump(train_annotations, f)
        with open(val_ann_file, 'w') as f:
            json.dump(val_annotations, f)

        # Create image directories and sample images
        train_img_dir = root_dir / "images" / "train"
        val_img_dir = root_dir / "images" / "val"
        train_img_dir.mkdir(parents=True, exist_ok=True)
        val_img_dir.mkdir(parents=True, exist_ok=True)

        # Create sample images
        (train_img_dir / "img1.jpg").touch()
        (val_img_dir / "img1.jpg").touch()

        # Get paths using get_dataset_paths()
        train_images_dir, train_annotations_file, val_images_dir, val_annotations_file = get_dataset_paths(str(root_dir))

        # Verify paths are correct when correlation exists
        assert train_images_dir == str(train_img_dir)
        assert train_annotations_file == str(train_ann_file)
        assert val_images_dir == str(val_img_dir)
        assert val_annotations_file == str(val_ann_file)

        # Test case with missing images
        (train_img_dir / "img1.jpg").unlink()
        train_images_dir, train_annotations_file, val_images_dir, val_annotations_file = get_dataset_paths(str(root_dir))
        
        # Verify train paths are None when correlation is broken
        assert train_images_dir is None
        assert train_annotations_file is None
        # Val paths should still be valid
        assert val_images_dir == str(val_img_dir)
        assert val_annotations_file == str(val_ann_file)

        # Test case with missing annotations file
        train_ann_file.unlink()
        train_images_dir, train_annotations_file, val_images_dir, val_annotations_file = get_dataset_paths(str(root_dir))
        
        # Verify both train paths are None
        assert train_images_dir is None
        assert train_annotations_file is None
        # Val paths should still be valid
        assert val_images_dir == str(val_img_dir)
        assert val_annotations_file == str(val_ann_file)
    
    def test_read_write_annotations(self, tmp_path, sample_coco_data):
        """Test reading and writing COCO annotations."""
        json_path = tmp_path / "annotations.json"
        
        # Test writing
        write_annotations(sample_coco_data, str(json_path))
        assert json_path.exists()
        
        # Test reading
        loaded_data = read_annotations(str(json_path))
        assert loaded_data == sample_coco_data
        
        # Verify structure
        assert "images" in loaded_data
        assert "annotations" in loaded_data
        assert "categories" in loaded_data

    def test_replace_label_class(self, tmp_path, sample_coco_data):
        """Test replacing category IDs in annotations."""
        json_path = tmp_path / "annotations.json"
        write_annotations(sample_coco_data, str(json_path))
        
        old_class_id = 1
        new_class_id = 3
        modified_count, modified_data = replace_annotations_class(
            old_class_id, new_class_id, str(json_path)
        )
        
        # Verify modifications
        assert modified_count > 0
        assert not any(ann['category_id'] == old_class_id 
                      for ann in modified_data['annotations'])
        assert any(ann['category_id'] == new_class_id 
                  for ann in modified_data['annotations'])
        assert any(cat['id'] == new_class_id 
                  for cat in modified_data['categories'])

    def test_shift_class_ids(self, tmp_path, sample_coco_data):
        """Test shifting all category IDs by a constant value."""
        json_path = tmp_path / "annotations.json"
        write_annotations(sample_coco_data, str(json_path))
        
        shift_value = 10
        shifted_data = shift_class_ids(str(json_path), shift_value)
        
        # Verify shifts
        original_ids = {cat['id'] for cat in sample_coco_data['categories']}
        shifted_ids = {cat['id'] for cat in shifted_data['categories']}
        assert all(sid == oid + shift_value 
                  for sid, oid in zip(sorted(shifted_ids), sorted(original_ids)))
        
        # Verify annotations updated
        assert all(ann['category_id'] >= shift_value 
                  for ann in shifted_data['annotations'])

    def test_split_dataset(self, sample_coco_data):
        """Test splitting dataset into train/val/test sets."""
        train, val, test = split_dataset(
            sample_coco_data,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2
        )
        
        total_images = len(sample_coco_data['images'])
        assert len(train['images']) == int(total_images * 0.6)
        assert len(val['images']) == int(total_images * 0.2)
        assert len(test['images']) == total_images - len(train['images']) - len(val['images'])
        
        # Verify no overlap between splits
        train_ids = {img['id'] for img in train['images']}
        val_ids = {img['id'] for img in val['images']}
        test_ids = {img['id'] for img in test['images']}
        assert not (train_ids & val_ids)
        assert not (train_ids & test_ids)
        assert not (val_ids & test_ids)
        
        # Verify categories preserved
        assert train['categories'] == sample_coco_data['categories']
        assert val['categories'] == sample_coco_data['categories']
        assert test['categories'] == sample_coco_data['categories']

    def test_split_dataset_with_max_images(self, sample_coco_data):
        """Test dataset splitting with maximum images per class limit."""
        max_images = 1
        train, val, test = split_dataset(
            sample_coco_data,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            max_images_per_class=max_images
        )
        
        # Count images per class in training set
        class_counts = defaultdict(int)
        for ann in train['annotations']:
            class_counts[ann['category_id']] += 1
            
        assert all(count <= max_images for count in class_counts.values())

    def test_convert_to_yolo(self, tmp_path, sample_coco_data):
        """Test conversion from COCO to YOLO format."""
        # Setup directory
        output_dir = tmp_path / "labels"
        
        # Create COCO json file
        coco_json = tmp_path / "annotations.json"
        with open(coco_json, 'w') as f:
            json.dump(sample_coco_data, f, indent=3)

        # Convert annotations
        yolo_annotations = convert_coco_to_yolo_labels(
            str(coco_json),
            str(output_dir)
        )
        
        # Verify conversion
        assert len(yolo_annotations) > 0
        
        # Verify YOLO format
        for filename, annotations in yolo_annotations.items():
            for ann in annotations:
                # Check format [class_id, x, y, w, h]
                assert len(ann) == 5
                # Verify normalized coordinates
                assert all(0 <= coord <= 1 for coord in ann[1:])
                # Verify class_id is integer
                assert isinstance(ann[0], int)
        
        # Verify output files if directory was provided
        assert output_dir.exists()
        assert (output_dir / "classes.names").exists()
        # Check label files
        for img in sample_coco_data['images']:
            base_name = os.path.splitext(img['file_name'])[0]
            assert (output_dir / f"{base_name}.txt").exists()

    def test_count_annotations(self, tmp_path, sample_coco_data):
        """Test counting annotations and annotated images."""
        json_path = tmp_path / "annotations.json"
        write_annotations(sample_coco_data, str(json_path))
        
        # Test without class names
        counts, total_images = count_annotations(str(json_path))
        assert total_images == len(sample_coco_data['images'])
        assert sum(counts.values()) == len(sample_coco_data['annotations'])
        
        # Test with class names
        class_names = [cat['name'] for cat in sample_coco_data['categories']]
        counts, total_images = count_annotations(str(json_path), class_names)
        assert all(isinstance(name, str) for name in counts.keys())
        assert sum(counts.values()) == len(sample_coco_data['annotations'])

    def test_split_and_organize(self, dataset_dir, sample_coco_data):
        """Test splitting and organizing dataset into directory structure."""
        root_dir, _ = dataset_dir
        
        # Create initial dataset
        for img in sample_coco_data['images']:
            (root_dir / "images" / img['file_name']).touch()
            
        ann_file = root_dir / "annotations" / "annotations.json"
        write_annotations(sample_coco_data, str(ann_file))
        
        split_and_organize_dataset(
            str(root_dir),
            str(ann_file),
            max_images_per_class=None,
            train_ratio=0.7,
            val_ratio=0.2,
            test_ratio=0.1
        )
        
        # Verify split files exist
        for split in ['train', 'val', 'test']:
            split_ann = root_dir / "annotations" / f"instances_{split}.json"
            assert split_ann.exists()
            
            # Verify split contents
            with open(split_ann) as f:
                split_data = json.load(f)
                assert all(key in split_data for key in ['images', 'annotations', 'categories'])
                

class TestCOCOLoader:
    @pytest.fixture
    def sample_coco_data(self):
        """Create sample COCO format data."""
        return {
            "images": [
                {"id": 1, "file_name": "img1.jpg", "width": 640, "height": 480},
                {"id": 2, "file_name": "img2.jpg", "width": 640, "height": 480},
                {"id": 3, "file_name": "img3.jpg", "width": 640, "height": 480}
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [100, 100, 50, 50], "area": 2500, "iscrowd": 0},
                {"id": 2, "image_id": 1, "category_id": 2, "bbox": [200, 200, 40, 60], "area": 2400, "iscrowd": 0},
                {"id": 3, "image_id": 2, "category_id": 1, "bbox": [150, 150, 45, 45], "area": 2025, "iscrowd": 0},
                {"id": 4, "image_id": 3, "category_id": 2, "bbox": [300, 300, 50, 50], "area": 2500, "iscrowd": 0}
            ],
            "categories": [
                {"id": 1, "name": "person", "supercategory": "none"},
                {"id": 2, "name": "car", "supercategory": "none"}
            ]
        }

    @pytest.fixture
    def dataset_dir(self, tmp_path):
        """Create a temporary dataset directory."""
        dataset_dir = tmp_path / "coco_dataset"
        image_dir = dataset_dir / "images"
        image_dir.mkdir(parents=True)
        return dataset_dir

    @pytest.fixture
    def sample_images(self, dataset_dir):
        """Create sample images for testing."""
        image_dir = dataset_dir / "images"

        # Create sample images (10x10 black images)
        for img in ["img1.jpg", "img2.jpg", "img3.jpg"]:
            image = Image.new('RGB', (10, 10), color='black')
            image.save(image_dir / img)
        
        return image_dir

    def test_coco_dataloader(self, dataset_dir, sample_coco_data, sample_images):
        """Test basic functionality of COCODatasetLoader."""
        # Create annotations file
        ann_file = dataset_dir / "annotations.json"
        with open(ann_file, 'w') as f:
            json.dump(sample_coco_data, f)

        # Initialize loader
        loader = COCOLoader(
            image_dir=str(sample_images),
            annotations_file=str(ann_file)
        )

        # Test length
        assert len(loader) == 3

        # Test iteration
        for image, annotations in loader:
            assert isinstance(image, SaltupImage)
            assert isinstance(image.get_data(), np.ndarray)
            assert len(annotations) > 0
            for ann in annotations:
                if isinstance(ann, dict):
                    assert "bbox" in ann
                    assert "category_id" in ann
                elif isinstance(ann, BBoxClassId):
                    assert isinstance(ann, BBoxClassId)
                else:
                    raise ValueError(f"Annotation type '{type(ann)}' not recognized.")

    def test_missing_images_directory(self, dataset_dir, sample_coco_data):
        """Test loader with non-existent images directory."""
        ann_file = dataset_dir / "annotations.json"
        with open(ann_file, 'w') as f:
            json.dump(sample_coco_data, f)

        with pytest.raises(FileNotFoundError):
            COCOLoader(
                image_dir=str(dataset_dir / "nonexistent"),
                annotations_file=str(ann_file)
            )

    def test_missing_annotations_file(self, dataset_dir, sample_images):
        """Test loader with non-existent annotations file."""
        with pytest.raises(FileNotFoundError):
            COCOLoader(
                image_dir=str(sample_images),
                annotations_file=str(dataset_dir / "nonexistent.json")
            )

    def test_invalid_annotations_format(self, dataset_dir, sample_images):
        """Test loader with invalid annotations format."""
        ann_file = dataset_dir / "invalid.json"
        with open(ann_file, 'w') as f:
            json.dump({"invalid": "data"}, f)

        with pytest.raises(ValueError):
            COCOLoader(
                image_dir=str(sample_images),
                annotations_file=str(ann_file)
            )

    def test_missing_images_files(self, dataset_dir, sample_coco_data, sample_images):
        """Test loader when some image files are missing."""
        # Create annotations file
        ann_file = dataset_dir / "annotations.json"
        with open(ann_file, 'w') as f:
            json.dump(sample_coco_data, f)

        # Remove one image file
        (sample_images / "img1.jpg").unlink()

        # Initialize loader
        loader = COCOLoader(
            image_dir=str(sample_images),
            annotations_file=str(ann_file)
        )

        # Should have 2 valid pairs instead of 3
        assert len(loader) == 2

    def test_color_modes(self, dataset_dir, sample_coco_data, sample_images):
        """Test loader with different color modes."""
        ann_file = dataset_dir / "annotations.json"
        with open(ann_file, 'w') as f:
            json.dump(sample_coco_data, f)

        # Test RGB mode
        loader_rgb = COCOLoader(
            image_dir=str(sample_images),
            annotations_file=str(ann_file),
            color_mode=ColorMode.RGB
        )
        image, _ = next(loader_rgb)
        assert image.get_data().shape[-1] == 3

        # Test BGR mode
        loader_bgr = COCOLoader(
            image_dir=str(sample_images),
            annotations_file=str(ann_file),
            color_mode=ColorMode.BGR
        )
        image, _ = next(loader_bgr)
        assert image.get_data().shape[-1] == 3

        # Test GRAY mode
        loader_gray = COCOLoader(
            image_dir=str(sample_images),
            annotations_file=str(ann_file),
            color_mode=ColorMode.GRAY
        )
        image, _ = next(loader_gray)
        assert len(image.get_data().shape) == 2 or image.get_data().shape[-1] == 1

    def test_iterator_reset(self, dataset_dir, sample_coco_data, sample_images):
        """Test that iterator properly resets after completion."""
        ann_file = dataset_dir / "annotations.json"
        with open(ann_file, 'w') as f:
            json.dump(sample_coco_data, f)

        loader = COCOLoader(
            image_dir=str(sample_images),
            annotations_file=str(ann_file)
        )

        # First iteration
        first_images = [img for img, _ in loader]
        assert len(first_images) == 3

        # Second iteration
        second_images = [img for img, _ in loader]
        assert len(second_images) == 3

        # Compare iterations
        for img1, img2 in zip(first_images, second_images):
            assert isinstance(img1, SaltupImage)
            assert isinstance(img2, SaltupImage)
            assert np.array_equal(img1.get_data(), img2.get_data())

if __name__ == '__main__':
    pytest.main(['-v', __file__])
