import unittest
import pytest
import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf
from tensorflow import keras
import albumentations as A
from typing import Tuple, List

from saltup.ai.object_detection.datagenerator.anchors_based_datagen import (
    AnchorsBasedDatagen, BaseDataloader, PyTorchAnchorBasedDatagen, KerasAnchorBasedDatagen
)


# Mock dataset loader for testing
class MockDatasetLoader(BaseDataloader):
    def __init__(self, num_samples: int = 100):
        self.num_samples = num_samples
        self.current_idx = 0
        
    def __len__(self) -> int:
        return self.num_samples
        
    def __iter__(self):
        self.current_idx = 0
        return self
        
    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.current_idx >= self.num_samples:
            raise StopIteration
            
        # Generate mock image and annotations
        image = np.random.rand(416, 416, 1).astype(np.float32)
        num_objects = np.random.randint(1, 4)
        
        # Generate random boxes and labels
        boxes = []
        for _ in range(num_objects):
            x_center = np.random.uniform(0.2, 0.8)
            y_center = np.random.uniform(0.2, 0.8)
            width = np.random.uniform(0.1, 0.3)
            height = np.random.uniform(0.1, 0.3)
            label = np.random.randint(0, 3)  # 3 classes
            boxes.append([x_center, y_center, width, height, label])
            
        self.current_idx += 1
        return image, np.array(boxes)
        
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return next(self)
    
    def __len__(self):
        return self.num_samples

# Simple PyTorch model for testing
class SimplePyTorchModel(nn.Module):
    def __init__(self, num_classes: int, num_anchors: int):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        
        # Feature extraction with proper downsampling to 13x13
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 208x208
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 104x104
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 52x52
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 26x26
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 13x13
        )
        
        # Output head
        self.head = nn.Conv2d(256, num_anchors * (5 + num_classes), 1)
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.features(x)
        x = self.head(x)  # [batch, num_anchors * (5 + num_classes), 13, 13]
        
        # Reshape to match target format [batch, 13, 13, num_anchors, 5 + num_classes]
        return x.permute(0, 2, 3, 1).view(batch_size, 13, 13, self.num_anchors, 5 + self.num_classes)

# Simple Keras model for testing
def create_simple_keras_model(input_shape: Tuple[int, int, int], num_classes: int, num_anchors: int):
    inputs = keras.Input(shape=input_shape)
    
    # Calculate required stride to reach grid size of 13x13
    # For 416x416 input, we need a total stride of 32 to get 13x13 output
    x = keras.layers.Conv2D(16, 3, padding='same', activation='relu')(inputs)
    x = keras.layers.MaxPooling2D()(x)  # stride 2
    x = keras.layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = keras.layers.MaxPooling2D()(x)  # stride 4
    x = keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = keras.layers.MaxPooling2D()(x)  # stride 8
    x = keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = keras.layers.MaxPooling2D()(x)  # stride 16
    x = keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = keras.layers.MaxPooling2D()(x)  # stride 32 -> 13x13
    
    # Output convolution to match target shape [batch, 13, 13, num_anchors * (5 + num_classes)]
    x = keras.layers.Conv2D(num_anchors * (5 + num_classes), 1, padding='same')(x)
    
    # Reshape to match target shape [batch, 13, 13, num_anchors, (5 + num_classes)]
    outputs = keras.layers.Reshape((13, 13, num_anchors, 5 + num_classes))(x)
    
    return keras.Model(inputs=inputs, outputs=outputs)


class TestAnchorsBasedDataloader:
    @pytest.fixture
    def mock_loader(self):
        return MockDatasetLoader(num_samples=4)
    
    @pytest.fixture
    def anchors(self):
        return np.array([[0.2, 0.2], [0.5, 0.5]])
        
    @pytest.fixture
    def dataloader(self, mock_loader, anchors):
        return AnchorsBasedDatagen(
            dataloader=mock_loader,
            anchors=anchors,
            target_size=(224, 224),
            grid_size=(7, 7),
            num_classes=2
        )
    
    def test_initialization(self, dataloader, mock_loader, anchors):
        """Test dataloader initialization."""
        assert dataloader.dataloader == mock_loader
        assert np.array_equal(dataloader.anchors, anchors)
        assert dataloader.target_size == (224, 224)
        assert dataloader.grid_size == (7, 7)
        assert dataloader.num_classes == 2
        assert dataloader.batch_size == 1
        assert not dataloader.do_augment
        
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
        
        assert dataloader.do_augment
        images, labels = next(iter(dataloader))
        
        assert images.shape == (1, 224, 224, 1)
        assert labels.shape == (1, 7, 7, 2, 7)
        
    @pytest.mark.parametrize("show_grid,show_anchors", [
        (True, True),
        (True, False),
        (False, True),
        (False, False)
    ])
    def test_visualization(self, dataloader: AnchorsBasedDatagen, show_grid, show_anchors):
        """Test visualization with different options."""
        try:
            dataloader.visualize_sample(0, show_grid=show_grid, show_anchors=show_anchors)
        except Exception as e:
            pytest.fail(f"Visualization failed: {e}")


class TestFrameworksAnchorsBasedDataloader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Configure TensorFlow to use CPU only
        tf.config.set_visible_devices([], 'GPU')
        physical_devices = tf.config.list_physical_devices('CPU')
        try:
            # Configurazione per la CPU
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except:
            pass
            
    def setUp(self):
        # Common test parameters
        self.dataloader = MockDatasetLoader(num_samples=10)
        self.anchors = np.array([[0.12, 0.15], [0.25, 0.25], [0.35, 0.35]])
        self.target_size = (416, 416)
        self.grid_size = (13, 13)
        self.num_classes = 3
        self.batch_size = 2
        
        # Test augmentation
        self.transform = A.Compose([
            A.RandomBrightnessContrast(p=0.5),
            A.GaussNoise(p=0.5)
        ], bbox_params=A.BboxParams(format='yolo'))
        
    def test_pytorch_dataloader(self):
        print("\nTesting PyTorch Dataloader...")
        
        # Initialize dataset
        dataset = PyTorchAnchorBasedDatagen(
            dataloader=self.dataloader,
            anchors=self.anchors,
            target_size=self.target_size,
            grid_size=self.grid_size,
            num_classes=self.num_classes,
            transform=self.transform
        )
        
        # Create dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=PyTorchAnchorBasedDatagen.collate_fn
        )
        
        # Test model
        model = SimplePyTorchModel(
            num_classes=self.num_classes,
            num_anchors=len(self.anchors)
        )
        optimizer = torch.optim.Adam(model.parameters())
        
        # Test training loop
        model.train()
        for batch_idx, (images, targets) in enumerate(dataloader):
            # Verify shapes
            self.assertEqual(images.shape, (self.batch_size, 1, 416, 416))
            self.assertEqual(targets.shape, (self.batch_size, 13, 13, 3, 8))  # 8 = 5 + num_classes
            
            # Test forward pass
            optimizer.zero_grad()
            outputs = model(images)
            
            # Dummy loss
            loss = torch.mean((outputs - targets) ** 2)
            loss.backward()
            optimizer.step()
            
            print(f"PyTorch Batch {batch_idx}, Loss: {loss.item():.4f}")
            if batch_idx >= 2:  # Test a few batches
                break
                
    def test_keras_dataloader(self):
        print("\nTesting Keras Dataloader...")
        
        # Initialize dataset
        dataset = KerasAnchorBasedDatagen(
            dataloader=self.dataloader,
            anchors=self.anchors,
            target_size=self.target_size,
            grid_size=self.grid_size,
            num_classes=self.num_classes,
            batch_size=self.batch_size,
            transform=self.transform
        )
        
        # Verify dataset can generate batches correctly
        for _ in range(2):
            batch = next(iter(dataset))
            self.assertEqual(len(batch), 2)  # images and labels
            images, labels = batch
            self.assertEqual(images.shape, (self.batch_size, 416, 416, 1))
            self.assertEqual(labels.shape, (self.batch_size, 13, 13, 3, 8))

        # Create and compile model with mixed precision disabled
        tf.keras.mixed_precision.set_global_policy('float32')
        model = create_simple_keras_model(
            input_shape=(416, 416, 1),
            num_classes=self.num_classes,
            num_anchors=len(self.anchors)
        )
        
        model.compile(
            optimizer='adam',
            loss='mse',  # Dummy loss for testing
            run_eagerly=True  # For better error messages
        )
        
        # Test training with reduced steps
        try:
            history = model.fit(
                dataset,
                epochs=1,
                steps_per_epoch=2,
                verbose=1
            )
            
            self.assertTrue(len(history.history['loss']) == 1)
            print("Keras training completed successfully")
        except Exception as e:
            self.fail(f"Training failed with error: {str(e)}")
            
# TODO: Add test with DataLoader integrated with Image and BBox classes OR integrate that classes in MockDatasetLoader

if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2)