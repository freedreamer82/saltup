import pytest
import numpy as np
import os
import pickle
from pathlib import Path
import shutil

from saltup.utils.data.image.image_utils import Image, ColorMode
from saltup.ai.object_detection.dataset.base_dataset_loader import BaseDatasetLoader


class MockDatasetLoader(BaseDatasetLoader):
    """Mock dataset loader for testing BaseDatasetLoader functionality."""
    
    def __init__(self, num_samples=3, image_size=(10, 10)):
        self.num_samples = num_samples
        self.image_size = image_size
        self._current_index = 0
        
        # Create dummy data
        self.data = []
        for i in range(num_samples):
            # Create dummy image array
            image_array = np.zeros((*image_size, 3), dtype=np.uint8)
            image = Image(image_array, ColorMode.RGB)
            
            # Create dummy labels (similar to YOLO format)
            labels = [(0, 0.5, 0.5, 0.2, 0.3), (1, 0.3, 0.4, 0.1, 0.2)]
            self.data.append((image, labels))

    def __iter__(self):
        self._current_index = 0
        return self

    def __next__(self):
        if self._current_index >= len(self.data):
            raise StopIteration
        item = self.data[self._current_index]
        self._current_index += 1
        return item

    def __len__(self):
        return self.num_samples


class TestBaseDatasetLoader:
    @pytest.fixture
    def tmp_dataset_dir(self, tmp_path):
        """Create a temporary directory for dataset files."""
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        return dataset_dir

    @pytest.fixture
    def mock_loader(self):
        """Create a mock dataset loader instance."""
        return MockDatasetLoader()

    def test_save_dataset_basic(self, mock_loader, tmp_dataset_dir):
        """Test basic save functionality with default parameters."""
        output_path = mock_loader.save_dataset(str(tmp_dataset_dir / "test.pkl"))
        
        assert output_path.exists()
        assert output_path.suffix == '.pkl'
        
        # Verify saved data structure
        with open(output_path, 'rb') as f:
            saved_data = pickle.load(f)
        
        assert isinstance(saved_data, list)
        assert len(saved_data) == len(mock_loader)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in saved_data)

    def test_save_dataset_custom_process(self, mock_loader, tmp_dataset_dir):
        """Test save with custom processing function."""
        def custom_process(image):
            # Example processing: set a specific value
            return np.full_like(image, 100, dtype=np.uint8)

        output_path = mock_loader.save_dataset(
            str(tmp_dataset_dir / "processed.pkl"),
            process_fn=custom_process
        )
        
        # Verify processed data
        with open(output_path, 'rb') as f:
            saved_data = pickle.load(f)
        
        # Check if processing was applied
        first_image = saved_data[0][0]
        assert np.all(first_image == 100)  # All values should be 100

    def test_save_dataset_auto_filename(self, mock_loader, tmp_dataset_dir):
        """Test save functionality with auto-generated filename."""
        output_path = mock_loader.save_dataset()
        
        assert output_path.exists()
        assert output_path.stem.startswith('dataset_')
        assert output_path.suffix == '.pkl'

    def test_save_dataset_existing_dir(self, mock_loader, tmp_dataset_dir):
        """Test save to existing directory structure."""
        nested_dir = tmp_dataset_dir / "nested" / "dir"
        output_path = mock_loader.save_dataset(str(nested_dir / "test.pkl"))
        
        assert output_path.exists()
        assert output_path.parent == nested_dir

    def test_load_dataset_basic(self, mock_loader, tmp_dataset_dir):
        """Test basic load functionality."""
        # First save some data
        save_path = mock_loader.save_dataset(str(tmp_dataset_dir / "test.pkl"))
        
        # Load the data
        loaded_data = list(BaseDatasetLoader.load_dataset(str(save_path)))
        
        assert len(loaded_data) == len(mock_loader)
        
        # Verify data structure
        for item in loaded_data:
            assert isinstance(item, tuple)
            assert len(item) == 2
            image, labels = item
            assert isinstance(image, np.ndarray)
            assert isinstance(labels, list)
            assert all(len(label) == 5 for label in labels)  # YOLO format

    def test_load_dataset_missing_file(self):
        """Test load with non-existent file."""
        with pytest.raises(FileNotFoundError):
            list(BaseDatasetLoader.load_dataset("nonexistent.pkl"))

    def test_load_dataset_wrong_extension(self, tmp_dataset_dir):
        """Test load with wrong file extension."""
        wrong_file = tmp_dataset_dir / "wrong.txt"
        wrong_file.touch()
        
        with pytest.raises(ValueError, match="Invalid file format"):
            list(BaseDatasetLoader.load_dataset(str(wrong_file)))

    def test_load_dataset_corrupted_file(self, tmp_dataset_dir):
        """Test load with corrupted pickle file."""
        corrupted_file = tmp_dataset_dir / "corrupted.pkl"
        with open(corrupted_file, 'wb') as f:
            f.write(b'corrupted data')
        
        with pytest.raises(pickle.UnpicklingError):
            list(BaseDatasetLoader.load_dataset(str(corrupted_file)))

    def test_load_dataset_invalid_format(self, tmp_dataset_dir):
        """Test load with invalid data format."""
        invalid_file = tmp_dataset_dir / "invalid.pkl"
        
        # Save invalid data structure
        with open(invalid_file, 'wb') as f:
            pickle.dump({"invalid": "format"}, f)  # Not a list of tuples
        
        with pytest.raises(ValueError, match="Invalid dataset format"):
            list(BaseDatasetLoader.load_dataset(str(invalid_file)))

    def test_save_load_cycle(self, mock_loader, tmp_dataset_dir):
        """Test full save-load cycle preserves data integrity."""
        # Save the dataset
        save_path = mock_loader.save_dataset(str(tmp_dataset_dir / "cycle_test.pkl"))
        
        # Load the dataset
        loaded_data = list(BaseDatasetLoader.load_dataset(str(save_path)))
        
        # Compare original and loaded data
        original_data = [(img.get_data(), label) for img, label in mock_loader]
        
        assert len(loaded_data) == len(original_data)
        
        for (loaded_img, loaded_labels), (orig_img, orig_labels) in zip(loaded_data, original_data):
            # Compare images
            np.testing.assert_array_equal(loaded_img, orig_img)
            
            # Compare labels
            assert len(loaded_labels) == len(orig_labels)
            for loaded_label, orig_label in zip(loaded_labels, orig_labels):
                assert len(loaded_label) == len(orig_label)
                assert all(x == y for x, y in zip(loaded_label, orig_label))
