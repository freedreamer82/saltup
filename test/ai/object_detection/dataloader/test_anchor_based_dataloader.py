import pytest
import numpy as np
from pathlib import Path

from saltup.ai.object_detection.dataloader.anchors_based_dataloader import AnchorsBasedDataloader
from saltup.ai.object_detection.dataset.base_dataset_loader import BaseDatasetLoader, ColorMode


class MockDatasetLoader(BaseDatasetLoader):
    """Mock dataset loader for testing."""
    
    def __init__(self, num_samples=4):
        super().__init__()
        self.num_samples = num_samples
        self._current_index = 0
        
    def __iter__(self):
        self._current_index = 0
        return self
        
    def __next__(self):
        if self._current_index >= self.num_samples:
            raise StopIteration
            
        # Create dummy image and annotations
        image = np.zeros((100, 100, 1), dtype=np.uint8)
        annotations = np.array([
            [0.5, 0.5, 0.2, 0.2, 0],  # center_x, center_y, width, height, class_id
            [0.2, 0.2, 0.1, 0.1, 1]
        ])
        
        self._current_index += 1
        return image, annotations
        
    def __len__(self):
        return self.num_samples


class TestAnchorsBasedDataloader:
    @pytest.fixture
    def mock_loader(self):
        return MockDatasetLoader()
    
    @pytest.fixture
    def anchors(self):
        return np.array([[0.2, 0.2], [0.5, 0.5]])
        
    @pytest.fixture
    def dataloader(self, mock_loader, anchors):
        return AnchorsBasedDataloader(
            dataset_loader=mock_loader,
            anchors=anchors,
            target_size=(224, 224),
            grid_size=(7, 7),
            num_classes=2
        )
    
    def test_initialization(self, dataloader, mock_loader, anchors):
        """Test dataloader initialization."""
        assert dataloader.dataset_loader == mock_loader
        assert np.array_equal(dataloader.anchors, anchors)
        assert dataloader.target_size == (224, 224)
        assert dataloader.grid_size == (7, 7)
        assert dataloader.num_classes == 2
        assert dataloader.batch_size == 1
        assert not dataloader.augment
        
    def test_length(self, dataloader):
        """Test length calculation."""
        assert len(dataloader) == 4  # From mock loader
        
        # Test with different batch sizes
        dataloader.batch_size = 2
        assert len(dataloader) == 2
        
        dataloader.batch_size = 3
        assert len(dataloader) == 2  # Ceil(4/3)
        
    def test_batch_generation(self, dataloader):
        """Test batch generation."""
        images, labels = next(iter(dataloader))
        
        # Check shapes
        assert images.shape == (1, 224, 224, 1)
        assert labels.shape == (1, 7, 7, 2, 7)  # batch, grid_h, grid_w, anchors, 5+num_classes
        
        # Check value ranges
        assert np.all((images >= 0) & (images <= 1))
        assert np.all((labels >= 0) & (labels <= 1))
        
    def test_batch_size(self, dataloader):
        """Test different batch sizes."""
        dataloader.batch_size = 2
        images, labels = next(iter(dataloader))
        
        assert images.shape[0] == 2
        assert labels.shape[0] == 2
        
    def test_augmentation(self, dataloader):
        """Test with augmentations."""
        import albumentations as A
        
        dataloader.transform = A.Compose([
            A.RandomBrightnessContrast(p=1)
        ], bbox_params=A.BboxParams(format='yolo'))
        
        assert dataloader.augment
        images, labels = next(iter(dataloader))
        
        assert images.shape == (1, 224, 224, 1)
        assert labels.shape == (1, 7, 7, 2, 7)
        
    @pytest.mark.parametrize("show_grid,show_anchors", [
        (True, True),
        (True, False),
        (False, True),
        (False, False)
    ])
    def test_visualization(self, dataloader, show_grid, show_anchors):
        """Test visualization with different options."""
        try:
            dataloader.visualize_sample(0, show_grid=show_grid, show_anchors=show_anchors)
        except Exception as e:
            pytest.fail(f"Visualization failed: {e}")