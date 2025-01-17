import pytest
import os
import numpy as np
from PIL import Image

from saltup.ai.object_detection.dataset.pascal_voc import (
    PascalVOCLoader,
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
            assert ann["class_name"] == expected["class_name"]
            assert ann["bbox"] == expected["bbox"]

    def test_pascal_voc_loader(self, dataset_dir, sample_pascal_voc_data):
        """Test PascalVOCLoader with sample data."""
        root_dir, dirs = dataset_dir

        # Create sample image and annotation
        train_img_dir = root_dir / "images" / "train"
        train_ann_dir = root_dir / "annotations" / "train"

        # Crea un'immagine valida (10x10 pixel, nera)
        image = Image.new('RGB', (10, 10), color='black')
        image_path = train_img_dir / "example.jpg"
        image.save(str(image_path))

        # Scrivi l'annotazione
        write_annotation(
            str(train_ann_dir / "example.xml"),
            sample_pascal_voc_data["annotations"],
            {
                "filename": "example.jpg",
                "width": 10,  # Larghezza dell'immagine
                "height": 10,  # Altezza dell'immagine
            }
        )

        # Initialize loader
        loader = PascalVOCLoader(root_dir=str(root_dir))

        # Test length
        assert len(loader) == 1

        # Test iteration
        for image, annotations in loader:
            assert isinstance(image, np.ndarray)
            assert len(annotations) == len(sample_pascal_voc_data["annotations"])
            for ann, expected in zip(annotations, sample_pascal_voc_data["annotations"]):
                assert ann["class_name"] == expected["class_name"]
                assert ann["bbox"] == expected["bbox"]

    def test_pascal_voc_loader_invalid_init(self, dataset_dir):
        """Test invalid initializations of PascalVOCLoader."""
        root_dir, _ = dataset_dir

        # Test with no arguments
        with pytest.raises(ValueError):
            PascalVOCLoader()

        # Test with incomplete arguments
        with pytest.raises(ValueError):
            PascalVOCLoader(image_dir=str(root_dir / "images" / "train"))

        # Test with mixed arguments
        with pytest.raises(ValueError):
            PascalVOCLoader(
                root_dir=str(root_dir),
                image_dir=str(root_dir / "images" / "train")
            )

    def test_pascal_voc_loader_missing_files(self, dataset_dir):
        """Test PascalVOCLoader with missing files."""
        root_dir, _ = dataset_dir

        # Create directory with missing annotation files
        incomplete_annotations = root_dir / "annotations" / "incomplete"
        incomplete_annotations.mkdir()

        # Should not raise error but should skip missing pairs
        loader = PascalVOCLoader(
            image_dir=str(root_dir / "images" / "train"),
            annotations_dir=str(incomplete_annotations)
        )

        assert len(loader) == 0  # No valid image-annotation pairs

    def test_get_dataset_paths(self, dataset_dir):
        """Test getting directory paths for Pascal VOC dataset."""
        root_dir, _ = dataset_dir
        train_images_dir, train_annotations_dir, val_images_dir, val_annotations_dir = get_dataset_paths(str(root_dir))

        assert train_images_dir == str(root_dir / "images" / "train")
        assert train_annotations_dir == str(root_dir / "annotations" / "train")
        assert val_images_dir == str(root_dir / "images" / "val")
        assert val_annotations_dir == str(root_dir / "annotations" / "val")

if __name__ == '__main__':
    pytest.main(['-v', __file__])