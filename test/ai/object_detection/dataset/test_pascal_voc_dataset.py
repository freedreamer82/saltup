import pytest
import os
import numpy as np
from PIL import Image

from saltup.utils.data.image.image_utils import Image as SaltupImage
from saltup.ai.object_detection.utils.bbox import BBoxClassId, BBoxFormat
from saltup.ai.object_detection.dataset.pascal_voc import (
    PascalVOCLoader, ColorMode,
    create_dataset_structure,
    validate_dataset_structure,
    read_annotation,
    write_annotation,
    get_dataset_paths,
)

class TestPascalVOCDataset:
    @pytest.fixture
    def sample_pascal_voc_data(self):
        """Create sample Pascal VOC format data."""
        return {
            "images": [
                {"id": 1, "file_name": "img1.jpg", "width": 640, "height": 480},
                {"id": 2, "file_name": "img2.jpg", "width": 640, "height": 480},
                {"id": 3, "file_name": "img3.jpg", "width": 640, "height": 480},
            ],
            "annotations": [
                {
                    "class_name": "person",
                    "bbox": (100, 100, 250, 250),
                },
                {
                    "class_name": "car",
                    "bbox": (200, 200, 400, 400),
                },
            ],
        }

    @pytest.fixture
    def dataset_dir(self, tmp_path):
        """Create a temporary dataset directory with Pascal VOC structure."""
        dataset_dir = tmp_path / "pascal_voc_dataset"
        dirs = create_dataset_structure(str(dataset_dir))
        return dataset_dir, dirs

    def test_create_dataset_structure(self, tmp_path):
        """Test creation of Pascal VOC dataset directory structure."""
        root_dir = tmp_path / "dataset"
        directories = create_dataset_structure(str(root_dir))

        # Verify all required directories exist
        assert os.path.exists(directories['images']['train'])
        assert os.path.exists(directories['images']['val'])
        assert os.path.exists(directories['images']['test'])
        assert os.path.exists(directories['annotations']['train'])
        assert os.path.exists(directories['annotations']['val'])
        assert os.path.exists(directories['annotations']['test'])

        # Verify directory permissions
        for split_dir in directories['images'].values():
            assert os.access(split_dir, os.W_OK)
        for split_dir in directories['annotations'].values():
            assert os.access(split_dir, os.W_OK)

    def test_validate_dataset_structure(self, dataset_dir, sample_pascal_voc_data):
        """Test validation of dataset structure with sample data."""
        root_dir, _ = dataset_dir

        # Create annotations file
        ann_file = root_dir / "annotations" / "train" / "example.xml"
        with open(ann_file, 'w') as f:
            write_annotation(str(ann_file), sample_pascal_voc_data["annotations"], {
                "filename": "example.jpg",
                "width": 640,
                "height": 480,
            })

        # Create corresponding images
        train_img_dir = root_dir / "images" / "train"
        (train_img_dir / "example.jpg").touch()

        stats = validate_dataset_structure(str(root_dir))

        assert stats['train']['images'] == 1
        assert stats['train']['annotations'] == 1
        assert stats['val']['images'] == 0
        assert stats['test']['images'] == 0

    def test_read_write_annotation(self, tmp_path, sample_pascal_voc_data):
        """Test reading and writing Pascal VOC annotations."""
        annotation_file = tmp_path / "example.xml"

        # Test writing
        write_annotation(
            str(annotation_file),
            sample_pascal_voc_data["annotations"],
            {
                "filename": "example.jpg",
                "width": 640,
                "height": 480,
            }
        )
        assert annotation_file.exists()

        # Test reading
        annotations = read_annotation(str(annotation_file))
        assert len(annotations) == len(sample_pascal_voc_data["annotations"])
        for ann, expected in zip(annotations, sample_pascal_voc_data["annotations"]):
            if isinstance(ann, dict):
                assert ann["class_name"] == expected["class_name"]
                assert ann["bbox"] == expected["bbox"]
            elif isinstance(ann, BBoxClassId):
                assert ann.class_name == expected["class_name"]
                assert ann.get_coordinates(fmt=BBoxFormat.PASCALVOC) == pytest.approx(expected["bbox"])
            else:
                raise ValueError(f"Annotation type '{type(ann)}' not recognized.")

    def test_get_dataset_paths(self, dataset_dir):
        """Test getting directory paths for Pascal VOC dataset."""
        root_dir, _ = dataset_dir
        train_images_dir, train_annotations_dir, val_images_dir, val_annotations_dir = get_dataset_paths(str(root_dir))

        assert train_images_dir == str(root_dir / "images" / "train")
        assert train_annotations_dir == str(root_dir / "annotations" / "train")
        assert val_images_dir == str(root_dir / "images" / "val")
        assert val_annotations_dir == str(root_dir / "annotations" / "val")
        

class TestPascalVOCLoader:
    @pytest.fixture
    def sample_pascal_voc_data(self):
        """Create sample Pascal VOC format data."""
        return {
            "annotations": [
                {
                    "class_name": "person",
                    "bbox": (100, 100, 250, 250),
                },
                {
                    "class_name": "car",
                    "bbox": (200, 200, 400, 400),
                },
            ]
        }

    @pytest.fixture
    def sample_dataset(self, tmp_path, sample_pascal_voc_data):
        """Create a temporary dataset with sample images and annotations."""
        dataset_dir = tmp_path / "dataset"
        image_dir = dataset_dir / "images"
        annotation_dir = dataset_dir / "annotations"
        
        # Create directories
        image_dir.mkdir(parents=True)
        annotation_dir.mkdir(parents=True)
        
        # Create sample image
        image = Image.new('RGB', (10, 10), color='black')
        image_paths = []
        annotation_paths = []
        
        # Create multiple samples
        for i in range(3):
            # Save image
            image_path = image_dir / f"img{i}.jpg"
            image.save(str(image_path))
            image_paths.append(image_path)
            
            # Write annotation
            annotation_path = annotation_dir / f"img{i}.xml"
            write_annotation(
                str(annotation_path),
                sample_pascal_voc_data["annotations"],
                {
                    "filename": f"img{i}.jpg",
                    "width": 10,
                    "height": 10,
                }
            )
            annotation_paths.append(annotation_path)
        
        return dataset_dir, {"image_dir": image_dir, "annotation_dir": annotation_dir}

    def test_pascal_voc_loader(self, sample_dataset):
        """Test basic functionality of PascalVOCLoader."""
        _, dirs = sample_dataset
        
        loader = PascalVOCLoader(
            images_dir=str(dirs["image_dir"]),
            annotations_dir=str(dirs["annotation_dir"])
        )
        
        # Test length
        assert len(loader) == 3
        
        # Test iteration
        for image, annotations in loader:
            assert isinstance(image, SaltupImage)
            assert isinstance(image.get_data(), np.ndarray)
            assert len(annotations) == 2  # Two objects per annotation
            for ann in annotations:
                if isinstance(ann, dict):
                    assert "class_name" in ann
                    assert "bbox" in ann
                    assert len(ann["bbox"]) == 4
                elif isinstance(ann, BBoxClassId):
                    assert isinstance(ann, BBoxClassId)
                    assert len(ann.get_coordinates(fmt=BBoxFormat.CORNERS_NORMALIZED)) == 4
                else:
                    raise ValueError(f"Annotation type '{type(ann)}' not recognized.")

    def test_pascal_voc_loader_missing_directories(self, sample_dataset):
        """Test PascalVOCLoader with missing directories."""
        _, dirs = sample_dataset
        
        # Test with non-existent image directory
        with pytest.raises(FileNotFoundError):
            PascalVOCLoader(
                images_dir="/nonexistent/path",
                annotations_dir=str(dirs["annotation_dir"])
            )
        
        # Test with non-existent annotations directory
        with pytest.raises(FileNotFoundError):
            PascalVOCLoader(
                images_dir=str(dirs["image_dir"]),
                annotations_dir="/nonexistent/path"
            )

    def test_pascal_voc_loader_missing_files(self, sample_dataset):
        """Test PascalVOCLoader with missing annotation files."""
        root_dir, dirs = sample_dataset
        
        # Create directory with missing annotation files
        incomplete_annotations = root_dir / "incomplete_annotations"
        incomplete_annotations.mkdir()
        
        # Should not raise error but should skip missing pairs
        loader = PascalVOCLoader(
            images_dir=str(dirs["image_dir"]),
            annotations_dir=str(incomplete_annotations)
        )
        
        assert len(loader) == 0  # No valid image-annotation pairs

    def test_pascal_voc_loader_iterator_reset(self, sample_dataset):
        """Test that iterator properly resets after completion."""
        _, dirs = sample_dataset
        
        loader = PascalVOCLoader(
            images_dir=str(dirs["image_dir"]),
            annotations_dir=str(dirs["annotation_dir"])
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

    def test_pascal_voc_loader_color_modes(self, sample_dataset):
        """Test different color modes in PascalVOCLoader."""
        _, dirs = sample_dataset
        
        # Test RGB mode
        loader_rgb = PascalVOCLoader(
            images_dir=str(dirs["image_dir"]),
            annotations_dir=str(dirs["annotation_dir"]),
            color_mode=ColorMode.RGB
        )
        
        # Test BGR mode
        loader_bgr = PascalVOCLoader(
            images_dir=str(dirs["image_dir"]),
            annotations_dir=str(dirs["annotation_dir"]),
            color_mode=ColorMode.BGR
        )
        
        # Test GRAY mode
        loader_gray = PascalVOCLoader(
            images_dir=str(dirs["image_dir"]),
            annotations_dir=str(dirs["annotation_dir"]),
            color_mode=ColorMode.GRAY
        )
        
        # Check first image of each loader
        for loader in [loader_rgb, loader_bgr, loader_gray]:
            image, _ = next(iter(loader))
            assert isinstance(image, SaltupImage)
            image_array = image.get_data()
            assert isinstance(image_array, np.ndarray)
            if loader.color_mode == ColorMode.GRAY:
                assert len(image_array.shape) == 2 or image_array.shape[-1] == 1
            else:
                assert image_array.shape[-1] == 3


if __name__ == '__main__':
    pytest.main(['-v', __file__])
