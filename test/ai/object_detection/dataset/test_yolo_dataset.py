import pytest
import os
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

from saltup.utils.data.image.image_utils import Image as SaltupImage
from saltup.ai.object_detection.utils.bbox import BBoxClassId, BBoxFormat
from saltup.ai.object_detection.dataset.yolo_darknet import (
    read_label, write_label, create_dataset_structure,
    validate_dataset_structure, analyze_dataset,
    create_symlinks_by_class, replace_label_class,
    shift_class_ids, split_dataset,
    split_and_organize_dataset, count_objects,
    _extract_unique_classes, _create_labels_map,
    _image_per_class_id, YoloDarknetLoader, ColorMode, YoloDataset
)

class TestYOLODarknet:
    @pytest.fixture
    def sample_dataset(self, tmp_path):
        """Create a sample YOLO dataset with images and labels."""
        dataset_root = tmp_path / "dataset"
        dirs = create_dataset_structure(str(dataset_root))
        
        # Sample data with multiple classes and overlapping instances
        sample_data = [
            ("img1.jpg", "0 0.5 0.5 0.2 0.3\n1 0.3 0.4 0.1 0.2"),  # Two objects
            ("img2.jpg", "1 0.4 0.6 0.3 0.2\n2 0.7 0.3 0.2 0.4"),  # Two objects
            ("img3.jpg", "0 0.6 0.5 0.25 0.35\n1 0.5 0.5 0.2 0.2"), # Two objects
            ("img4.jpg", "2 0.3 0.3 0.15 0.25")  # Single object
        ]
        
        # Create dummy image data (10x10 black image)
        dummy_image = np.zeros((10, 10, 3), dtype=np.uint8)
        
        # Create the sample files
        for img_name, label_content in sample_data:
            # Create and save image file
            img_path = Path(dirs['images']['train']) / img_name
            cv2.imwrite(str(img_path), dummy_image)
            
            # Create corresponding label file
            label_path = Path(dirs['labels']['train']) / f"{img_name.replace('.jpg', '.txt')}"
            with open(label_path, 'w') as f:
                f.write(label_content)
        
        return dataset_root, dirs

    @pytest.fixture
    def class_names(self):
        """Sample class names for testing."""
        return ['person', 'car', 'bike']

    def test_create_dataset_structure(self, tmp_path):
        """Test creation of YOLO dataset directory structure."""
        root_dir = tmp_path / "yolo_dataset"
        dirs = create_dataset_structure(str(root_dir))
        
        # Verify all required directories exist
        for split in ['train', 'val', 'test']:
            assert os.path.exists(dirs['images'][split])
            assert os.path.exists(dirs['labels'][split])
            
            # Check directory permissions
            assert os.access(dirs['images'][split], os.W_OK)
            assert os.access(dirs['labels'][split], os.W_OK)

    def test_validate_dataset_structure(self, sample_dataset):
        """Test validation of dataset structure and image-label pairs."""
        root_dir, _ = sample_dataset
        stats = validate_dataset_structure(str(root_dir))
        
        # Check training set statistics
        assert stats['train']['images'] == 4
        assert stats['train']['labels'] == 4
        assert stats['train']['matched'] == 4
        assert not stats['train']['unmatched_images']
        assert not stats['train']['unmatched_labels']
        
        # Check validation and test sets are empty
        assert stats['val']['images'] == 0
        # assert stats['test']['images'] == 0

    def test_read_write_label(self, tmp_path):
        """Test reading and writing YOLO format labels."""
        label_file = tmp_path / "test.txt"
        
        # Test data with multiple classes and objects
        test_labels = [
            [0, 0.5, 0.5, 0.2, 0.3],  # class 0
            [1, 0.3, 0.4, 0.1, 0.2],  # class 1
            [2, 0.7, 0.7, 0.15, 0.15]  # class 2
        ]
        
        # Test write operation
        write_label(str(label_file), test_labels)
        assert label_file.exists()
        
        # Test read operation and compare
        read_labels = read_label(str(label_file))
        assert len(read_labels) == len(test_labels)
        for read, expected in zip(read_labels, test_labels):
            np.testing.assert_array_almost_equal(read, expected)
        
        # Test invalid coordinates
        with pytest.raises(ValueError):
            write_label(str(label_file), [[0, 1.5, 0.5, 0.2, 0.3]])  # x > 1
            
        with pytest.raises(ValueError):
            write_label(str(label_file), [[0, 0.5, 0.5, -0.2, 0.3]])  # negative width

    def test_count_objects(self, sample_dataset, class_names):
        """Test counting objects per class and total images."""
        root_dir, dirs = sample_dataset
        counts, total_images = count_objects(
            str(dirs['labels']['train']), 
            class_names
        )
        
        # Verify total images
        assert total_images == 4
        
        # Verify object counts per class
        if isinstance(counts, dict):
            assert counts['person'] == 2  # class 0
            assert counts['car'] == 3     # class 1
            assert counts['bike'] == 2    # class 2
        else:
            assert counts[0] == 2  # class 0
            assert counts[1] == 3  # class 1
            assert counts[2] == 2  # class 2

    def test_replace_label_class(self, sample_dataset):
        """Test replacing class IDs in label files."""
        root_dir, dirs = sample_dataset
        train_labels = dirs['labels']['train']
        
        # Replace class 1 with class 3
        modified_count, modified_files = replace_label_class(1, 3, str(train_labels))
        
        assert modified_count > 0
        assert len(modified_files) > 0
        
        # Verify modifications
        for file_path in modified_files:
            labels = read_label(file_path)
            # Verify no labels have class_id 1 and some have class_id 3
            assert not any(label[0] == 1 for label in labels)
            assert any(label[0] == 3 for label in labels)
            # Verify other values weren't modified
            for label in labels:
                assert 0 <= label[1] <= 1  # x center
                assert 0 <= label[2] <= 1  # y center
                assert 0 <= label[3] <= 1  # width
                assert 0 <= label[4] <= 1  # height

    def test_shift_class_ids(self, sample_dataset):
        """Test shifting all class IDs by a constant value."""
        root_dir, dirs = sample_dataset
        train_labels = dirs['labels']['train']
        output_dir = root_dir / 'shifted_labels'
        
        shift_value = 10
        shift_class_ids(str(train_labels), shift_value, str(output_dir))
        
        assert output_dir.exists()
        
        # Verify shifted labels
        for label_file in output_dir.glob('*.txt'):
            with open(label_file) as f:
                lines = f.readlines()
                for line in lines:
                    class_id = int(line.split()[0])
                    assert class_id >= shift_value

    def test_split_dataset(self, sample_dataset):
        """Test splitting dataset while maintaining class distribution."""
        root_dir, dirs = sample_dataset
        
        # Get class distribution
        class_to_images = _image_per_class_id(
            str(dirs['labels']['train']),
            str(dirs['images']['train'])
        )
        
        # Test splitting
        train, val, test = split_dataset(
            class_to_images,
            split_ratio=0.8,
            split_val_ratio=0.5
        )
        
        # Verify splits
        total_images = sum(len(imgs) for imgs in class_to_images.values())
        total_split = len(train) + len(val) + len(test)
        assert total_split <= total_images
        
        # Check no overlap between splits
        train_set = set(str(x) for x in train)
        val_set = set(str(x) for x in val)
        test_set = set(str(x) for x in test)
        assert not (train_set & val_set)
        assert not (train_set & test_set)
        assert not (val_set & test_set)

    def test_split_with_max_images(self, sample_dataset):
        """Test dataset splitting with maximum images per class limit."""
        root_dir, dirs = sample_dataset
        class_to_images = _image_per_class_id(
            str(dirs['labels']['train']),
            str(dirs['images']['train'])
        )
        
        max_images = 1
        train, val, test = split_dataset(
            class_to_images,
            split_ratio=0.6,
            split_val_ratio=0.2,
            max_images_per_class=max_images
        )
        
        # Count images per class in training set
        class_counts = defaultdict(int)
        for img_path, _ in train:
            label_path = Path(str(img_path).replace('/images/', '/labels/').replace('.jpg', '.txt'))
            with open(label_path) as f:
                for line in f:
                    class_id = int(line.split()[0])
                    class_counts[class_id] += 1
                    
        # Verify limits
        assert all(count <= max_images for count in class_counts.values())

    def test_create_symlinks(self, sample_dataset, class_names):
        """Test creation of symbolic links organized by class."""
        root_dir, dirs = sample_dataset
        dest_dir = root_dir / 'by_class'
        
        create_symlinks_by_class(
            str(dirs['images']['train']),
            str(dirs['labels']['train']),
            str(dest_dir),
            class_names
        )
        
        # Verify structure
        assert (dest_dir / 'classes.names').exists()
        
        # Check class directories and symlinks
        for class_name in class_names:
            class_dir = dest_dir / class_name
            assert class_dir.exists()
            assert any(class_dir.iterdir())

    def test_analyze_dataset(self, sample_dataset, class_names):
        """Test dataset analysis functionality."""
        root_dir, _ = sample_dataset
        
        # This should not raise any errors
        analyze_dataset(str(root_dir), class_names)
        
        # Test without class names
        analyze_dataset(str(root_dir))
        

class TestYOLODarknetLoader:
    @pytest.fixture
    def sample_dataset(self, tmp_path):
        """Create a sample YOLO dataset with images and labels."""
        dataset_root = tmp_path / "dataset"
        image_dir = dataset_root / "images"
        labels_dir = dataset_root / "labels"
        image_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)
        
        # Sample data with multiple classes and overlapping instances
        sample_data = [
            ("img1.jpg", "0 0.5 0.5 0.2 0.3\n1 0.3 0.4 0.1 0.2"),  # Two objects
            ("img2.jpg", "1 0.4 0.6 0.3 0.2\n2 0.7 0.3 0.2 0.4"),  # Two objects
            ("img3.jpg", "0 0.6 0.5 0.25 0.35\n1 0.5 0.5 0.2 0.2"), # Two objects
            ("img4.jpg", "2 0.3 0.3 0.15 0.25")  # Single object
        ]
        
        # Create dummy image data (10x10 black image)
        dummy_image = np.zeros((10, 10, 3), dtype=np.uint8)
        
        # Create the sample files
        for img_name, label_content in sample_data:
            # Create and save image file
            img_path = image_dir / img_name
            cv2.imwrite(str(img_path), dummy_image)
            
            # Create corresponding label file
            label_path = labels_dir / f"{img_name.replace('.jpg', '.txt')}"
            with open(label_path, 'w') as f:
                f.write(label_content)
        
        return dataset_root, {"image_dir": image_dir, "labels_dir": labels_dir}

    def test_yolo_darknet_loader(self, sample_dataset):
        """Test YoloDarknetLoader basic functionality."""
        _, dirs = sample_dataset
        
        loader = YoloDarknetLoader(
            images_dir=str(dirs["image_dir"]),
            labels_dir=str(dirs["labels_dir"])
        )
        
        # Test length
        assert len(loader) == 4  # Based on sample_dataset fixture
        
        # Test iteration
        for image, labels in loader:
            assert isinstance(image, SaltupImage)
            assert isinstance(image.get_data(), np.ndarray)
            assert len(labels) > 0  # Each image has at least one label
            
            # Check labels format
            for label in labels:
                if isinstance(label, np.ndarray):
                    assert len(label) == 5  # class_id, x, y, w, h
                    assert 0 <= label[1] <= 1  # x normalized
                    assert 0 <= label[2] <= 1  # y normalized
                    assert 0 < label[3] <= 1   # width normalized
                    assert 0 < label[4] <= 1   # height normalized
                elif isinstance(label, BBoxClassId):
                    assert isinstance(label, BBoxClassId)
                    coordinates = label.get_coordinates(fmt=BBoxFormat.YOLO)
                    assert len(coordinates) == 4
                    xc, yc, w, h = coordinates
                    assert 0 <= xc <= 1
                    assert 0 <= yc <= 1
                    assert 0 < w <= 1  
                    assert 0 < h <= 1
                else:
                    raise ValueError(f"Label type '{type(label)}' not recognized.")

    def test_yolo_darknet_loader_color_modes(self, sample_dataset):
        """Test different color modes in YoloDarknetLoader."""
        _, dirs = sample_dataset
        
        # Test RGB mode
        loader_rgb = YoloDarknetLoader(
            images_dir=str(dirs["image_dir"]),
            labels_dir=str(dirs["labels_dir"]),
            color_mode=ColorMode.RGB
        )
        
        # Test BGR mode
        loader_bgr = YoloDarknetLoader(
            images_dir=str(dirs["image_dir"]),
            labels_dir=str(dirs["labels_dir"]),
            color_mode=ColorMode.BGR
        )
        
        # Test GRAY mode
        loader_gray = YoloDarknetLoader(
            images_dir=str(dirs["image_dir"]),
            labels_dir=str(dirs["labels_dir"]),
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

    def test_yolo_darknet_loader_missing_directories(self, sample_dataset):
        """Test YoloDarknetLoader with missing directories."""
        _, dirs = sample_dataset
        
        # Test with non-existent image directory
        with pytest.raises(FileNotFoundError):
            YoloDarknetLoader(
                images_dir="/nonexistent/path",
                labels_dir=str(dirs["labels_dir"])
            )
        
        # Test with non-existent labels directory
        with pytest.raises(FileNotFoundError):
            YoloDarknetLoader(
                images_dir=str(dirs["image_dir"]),
                labels_dir="/nonexistent/path"
            )

    def test_yolo_darknet_loader_missing_files(self, sample_dataset):
        """Test YoloDarknetLoader with missing files."""
        root_dir, dirs = sample_dataset
        
        # Create directory with missing label files
        incomplete_labels = root_dir / 'incomplete_labels'
        incomplete_labels.mkdir()
        
        # Should not raise error but should skip missing pairs
        loader = YoloDarknetLoader(
            images_dir=str(dirs["image_dir"]),
            labels_dir=str(incomplete_labels)
        )
        
        assert len(loader) == 0  # No valid image-label pairs

    def test_yolo_darknet_loader_iterator_reset(self, sample_dataset):
        """Test that iterator properly resets after completion."""
        _, dirs = sample_dataset
        
        loader = YoloDarknetLoader(
            images_dir=str(dirs["image_dir"]),
            labels_dir=str(dirs["labels_dir"])
        )
        
        # First iteration
        first_images = [img for img, _ in loader]
        assert len(first_images) == 4
        
        # Second iteration
        second_images = [img for img, _ in loader]
        assert len(second_images) == 4
        
        # Compare iterations
        for img1, img2 in zip(first_images, second_images):
            assert isinstance(img1, SaltupImage)
            assert isinstance(img2, SaltupImage)
            assert np.array_equal(img1.get_data(), img2.get_data())


class TestYoloDataset:
    @pytest.fixture
    def sample_dataset(self, tmpdir):
        """Create a sample YOLO dataset with images and labels."""
        images_dir = tmpdir.mkdir("images")
        labels_dir = tmpdir.mkdir("labels")
        
        # Sample data con diverse classi e overlapping
        sample_data = [
            ("img1.jpg", "0 0.5 0.5 0.2 0.3\n1 0.3 0.4 0.1 0.2"),
            ("img2.jpg", "1 0.4 0.6 0.3 0.2\n2 0.7 0.3 0.2 0.4"),
            ("img3.jpg", "0 0.6 0.5 0.25 0.35"),
        ]
        
        # Crea immagine dummy (10x10 nera)
        dummy_image = np.zeros((10, 10, 3), dtype=np.uint8)
        
        # Crea i file di test
        for img_name, label_content in sample_data:
            # Crea e salva l'immagine
            img_path = images_dir / img_name
            cv2.imwrite(str(img_path), dummy_image)
            
            # Crea il file delle label
            label_path = labels_dir / f"{img_name.replace('.jpg', '.txt')}"
            with open(label_path, 'w') as f:
                f.write(label_content)
        
        return str(images_dir), str(labels_dir)

    def test_initialization(self, sample_dataset):
        """Test inizializzazione del dataset."""
        images_dir, labels_dir = sample_dataset
        dataset = YoloDataset(images_dir, labels_dir)
        assert len(dataset.list_images_ids()) == 3

    def test_get_annotations(self, sample_dataset):
        """Test caricamento annotazioni."""
        images_dir, labels_dir = sample_dataset
        dataset = YoloDataset(images_dir, labels_dir)
        
        annotations = dataset.get_annotations("img1")
        assert len(annotations) == 2
        assert isinstance(annotations[0], BBoxClassId)
        assert annotations[0].class_id == 0
        assert annotations[1].class_id == 1

    def test_save_and_remove_image(self, sample_dataset):
        """Test salvataggio e rimozione immagini."""
        images_dir, labels_dir = sample_dataset
        dataset = YoloDataset(images_dir, labels_dir)
        
        # Salva nuova immagine
        new_image = np.zeros((10, 10, 3), dtype=np.uint8)
        dataset.save_image(new_image, "new_image")
        assert "new_image" in dataset.list_images_ids()
        
        # Rimuovi immagine
        assert dataset.remove_image("new_image")
        assert "new_image" not in dataset.list_images_ids()

    def test_check_integrity(self, sample_dataset):
        """Test controllo integrità."""
        images_dir, labels_dir = sample_dataset
        dataset = YoloDataset(images_dir, labels_dir)
        
        assert dataset.check_integrity()
        
        # Introduce un errore rimuovendo un file di label
        os.remove(Path(labels_dir) / "img1.txt")
        assert not dataset.check_integrity()
        
    def test_save_annotations(self, temp_dataset_dir):
        """Test saving annotations."""
        images_dir, labels_dir = temp_dataset_dir
        dataset = YoloDataset(images_dir, labels_dir)

        # Save new annotations
        new_annotations = [(1, 0.4, 0.4, 0.1, 0.1)]  # YOLO format: class_id, x_center, y_center, width, height
        dataset.save_annotations(new_annotations, "image1", overwrite=True)

        # Check if annotations were saved
        annotations = dataset.get_annotations("image1")
        assert len(annotations) == 1
        assert annotations[0].class_id == 1

    def test_refresh_mechanism(self, sample_dataset):
        """Test refresh mechanism."""
        images_dir, labels_dir = sample_dataset
        dataset = YoloDataset(images_dir, labels_dir, refresh_each=2)
        
        # Test adding image
        new_image = np.zeros((10, 10, 3), dtype=np.uint8)
        dataset.save_image(new_image, "new_image")
        
        # Verify refresh after two operations
        initial_ids = dataset.list_images_ids()
        dataset.save_image(new_image, "another_image")  # First operation
        mid_ids = dataset.list_images_ids()
        dataset.save_image(new_image, "third_image")    # Second operation
        final_ids = dataset.list_images_ids()
        
        assert len(final_ids) > len(initial_ids)
        
    def test_list_images_ids_max_entries(self, sample_dataset):
        """Test list_images_ids with max_entries."""
        images_dir, labels_dir = sample_dataset
        dataset = YoloDataset(images_dir, labels_dir)

        # Add multiple images
        for i in range(5):
            dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
            dataset.save_image(dummy_image, f"image{i}")

        # Check if max_entries works
        assert len(dataset.list_images_ids(max_entries=3)) == 3

    def test_save_annotations(self, sample_dataset):
        """Test salvataggio annotazioni."""
        images_dir, labels_dir = sample_dataset
        dataset = YoloDataset(images_dir, labels_dir)
        
        annotations = [(1, 0.4, 0.4, 0.1, 0.1)]
        dataset.save_annotations(annotations, "img1", overwrite=True)
        
        loaded_annotations = dataset.get_annotations("img1")
        assert len(loaded_annotations) == 1
        assert loaded_annotations[0].class_id == 1
        
    def test_save_image_annotations(self, sample_dataset):
        """Test combined saving of image and annotations."""
        images_dir, labels_dir = sample_dataset
        dataset = YoloDataset(images_dir, labels_dir)
        
        # Prepare test data
        test_image = np.zeros((10, 10, 3), dtype=np.uint8)
        test_annotations = [
            (0, 0.5, 0.5, 0.2, 0.3),  # class 0 in center
            (1, 0.3, 0.7, 0.1, 0.1)   # class 1 top right
        ]
        
        # Save both image and annotations
        dataset.save_image_annotations(test_image, "test_combined", test_annotations)
        
        # Verify image was saved
        assert "test_combined" in dataset.list_images_ids()
        
        # Load and verify image
        loaded_image = dataset.get_image("test_combined")
        assert isinstance(loaded_image, SaltupImage)
        assert loaded_image.get_shape() == (10, 10, 3)
        
        # Load and verify annotations
        loaded_annotations = dataset.get_annotations("test_combined")
        assert len(loaded_annotations) == 2
        assert loaded_annotations[0].class_id == 0
        assert loaded_annotations[1].class_id == 1
        
        # Verify annotation coordinates
        coords0 = loaded_annotations[0].get_coordinates(fmt=BBoxFormat.YOLO)
        coords1 = loaded_annotations[1].get_coordinates(fmt=BBoxFormat.YOLO)
        
        assert coords0[0] == pytest.approx(0.5)  # x center
        assert coords0[1] == pytest.approx(0.5)  # y center
        assert coords1[0] == pytest.approx(0.3)  # x center
        assert coords1[1] == pytest.approx(0.7)  # y center
        
    def test_invalid_image_or_annotation(self, sample_dataset):
        """Test handling of invalid images or annotations."""
        images_dir, labels_dir = sample_dataset
        dataset = YoloDataset(images_dir, labels_dir)

        # Try to load a non-existent image
        with pytest.raises(FileNotFoundError):
            dataset.get_image("nonexistent_image")

        # Try to save invalid annotations
        with pytest.raises(TypeError):
            dataset.save_annotations("invalid_annotations", "image1")

    def test_invalid_operations(self, sample_dataset):
        """Test gestione operazioni invalide."""
        images_dir, labels_dir = sample_dataset
        dataset = YoloDataset(images_dir, labels_dir)
        
        # Test immagine non esistente
        with pytest.raises(FileNotFoundError):
            dataset.get_image("nonexistent")
            
        # Test annotazioni invalide
        with pytest.raises(TypeError):
            dataset.save_annotations("invalid", "img1")


if __name__ == '__main__':
    pytest.main(['-v', __file__])
