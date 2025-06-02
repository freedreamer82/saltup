import pytest
import numpy as np
import os
import pickle
from pathlib import Path
import shutil

from saltup.utils.data.image.image_utils import Image, ColorMode
from saltup.ai.base_dataformat.base_dataset import BaseDataloader, StorageFormat


class MockDatasetLoader(BaseDataloader):
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
    def split(self, ratio):
        return super().split(ratio)

    @staticmethod
    def merge(dl1, dl2):
        """Merge two dataset loaders."""
        if not isinstance(dl1, MockDatasetLoader) or not isinstance(dl2, MockDatasetLoader):
            raise TypeError("Both datasets must be instances of MockDatasetLoader.")
        
        merged_loader = MockDatasetLoader(num_samples=len(dl1) + len(dl2))
        merged_loader.data = dl1.data + dl2.data
        return merged_loader
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
        
        os.remove(output_path)

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
        loaded_data = list(BaseDataloader.load_dataset(str(save_path)))
        
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
            list(BaseDataloader.load_dataset("nonexistent.pkl"))

    def test_load_dataset_wrong_extension(self, tmp_dataset_dir):
        """Test load with wrong file extension."""
        wrong_file = tmp_dataset_dir / "wrong.txt"
        wrong_file.touch()
        
        with pytest.raises(ValueError, match="Invalid file format"):
            list(BaseDataloader.load_dataset(str(wrong_file)))

    def test_load_dataset_corrupted_file(self, tmp_dataset_dir):
        """Test load with corrupted pickle file."""
        corrupted_file = tmp_dataset_dir / "corrupted.pkl"
        with open(corrupted_file, 'wb') as f:
            f.write(b'corrupted data')
        
        with pytest.raises(pickle.UnpicklingError):
            list(BaseDataloader.load_dataset(str(corrupted_file)))

    def test_load_dataset_invalid_format(self, tmp_dataset_dir):
        """Test load with invalid data format."""
        invalid_file = tmp_dataset_dir / "invalid.pkl"
        
        # Save invalid data structure
        with open(invalid_file, 'wb') as f:
            pickle.dump({"invalid": "format"}, f)  # Not a list of tuples
        
        with pytest.raises(ValueError, match="Invalid dataset format"):
            list(BaseDataloader.load_dataset(str(invalid_file)))

    def test_save_load_cycle(self, mock_loader, tmp_dataset_dir):
        """Test full save-load cycle preserves data integrity."""
        # Save the dataset
        save_path = mock_loader.save_dataset(str(tmp_dataset_dir / "cycle_test.pkl"))
        
        # Load the dataset
        loaded_data = list(BaseDataloader.load_dataset(str(save_path)))
        
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

    def test_save_dataset_parquet_basic(self, mock_loader, tmp_dataset_dir):
        """Test basic save functionality with Parquet format."""
        output_path = mock_loader.save_dataset(
            str(tmp_dataset_dir / "test.parquet"),
            format=StorageFormat.PARQUET
        )
        
        assert output_path.exists()
        assert output_path.suffix == '.parquet'
        
        # Load and verify data structure
        loaded_data = list(BaseDataloader.load_dataset(
            str(output_path),
            format=StorageFormat.PARQUET
        ))
        
        assert isinstance(loaded_data, list)
        assert len(loaded_data) == len(mock_loader)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in loaded_data)

    def test_save_dataset_parquet_custom_process(self, mock_loader, tmp_dataset_dir):
        """Test save with custom processing function in Parquet format."""
        def custom_process(image):
            return np.full_like(image, 100, dtype=np.uint8)

        output_path = mock_loader.save_dataset(
            str(tmp_dataset_dir / "processed.parquet"),
            process_fn=custom_process,
            format=StorageFormat.PARQUET
        )
        
        # Load and verify processed data
        loaded_data = list(BaseDataloader.load_dataset(
            str(output_path),
            format=StorageFormat.PARQUET
        ))
        
        # Check if processing was applied
        first_image = loaded_data[0][0]
        assert np.all(first_image == 100)  # All values should be 100

    def test_save_dataset_auto_filename_parquet(self, mock_loader):
        """Test save functionality with auto-generated filename in Parquet format."""
        output_path = mock_loader.save_dataset(format=StorageFormat.PARQUET)
        
        assert output_path.exists()
        assert output_path.stem.startswith('dataset_')
        assert output_path.suffix == '.parquet'
        
        os.remove(output_path)

    def test_load_dataset_parquet_wrong_extension(self, tmp_dataset_dir):
        """Test load Parquet with wrong file extension."""
        wrong_file = tmp_dataset_dir / "wrong.txt"
        wrong_file.touch()
        
        with pytest.raises(ValueError, match="Invalid file format"):
            list(BaseDataloader.load_dataset(
                str(wrong_file),
                format=StorageFormat.PARQUET
            ))

    def test_load_dataset_corrupted_parquet(self, tmp_dataset_dir):
        """Test load with corrupted Parquet file."""
        corrupted_file = tmp_dataset_dir / "corrupted.parquet"
        with open(corrupted_file, 'wb') as f:
            f.write(b'corrupted data')
        
        with pytest.raises(ValueError):
            list(BaseDataloader.load_dataset(
                str(corrupted_file),
                format=StorageFormat.PARQUET
            ))

    def test_save_load_cycle_parquet(self, mock_loader, tmp_dataset_dir):
        """Test full save-load cycle preserves data integrity with Parquet format."""
        # Save the dataset
        save_path = mock_loader.save_dataset(
            str(tmp_dataset_dir / "cycle_test.parquet"),
            format=StorageFormat.PARQUET
        )
        
        # Load the dataset
        loaded_data = list(BaseDataloader.load_dataset(
            str(save_path),
            format=StorageFormat.PARQUET
        ))
        
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

    def test_format_conversion(self, mock_loader, tmp_dataset_dir):
        """Test saving in one format and converting to another."""
        # Save in Pickle format
        pickle_path = mock_loader.save_dataset(
            str(tmp_dataset_dir / "data.pkl"),
            format=StorageFormat.PICKLE
        )
        
        # Load and save in Parquet format
        loaded_data = list(BaseDataloader.load_dataset(str(pickle_path)))
        mock_loader.data = [(Image(img, ColorMode.RGB), label) for img, label in loaded_data]
        parquet_path = mock_loader.save_dataset(
            str(tmp_dataset_dir / "data.parquet"),
            format=StorageFormat.PARQUET
        )
        
        # Load from Parquet and compare
        final_data = list(BaseDataloader.load_dataset(
            str(parquet_path),
            format=StorageFormat.PARQUET
        ))
        
        assert len(loaded_data) == len(final_data)
        for (pkl_img, pkl_labels), (parq_img, parq_labels) in zip(loaded_data, final_data):
            np.testing.assert_array_equal(pkl_img, parq_img)
            assert pkl_labels == parq_labels
