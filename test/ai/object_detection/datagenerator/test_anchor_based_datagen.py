import pytest
import os
import yaml
from dotmap import DotMap
import math
import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf
from tensorflow import keras
import albumentations as A
import matplotlib.pyplot as plt
from typing import Tuple, List

from saltup.utils.data.image.image_utils import Image
from saltup.ai.object_detection.utils.bbox import BBoxClassId
from saltup.ai.object_detection.datagenerator.anchors_based_datagen import (
    AnchorsBasedDatagen, BaseDataloader, PyTorchAnchorBasedDatagen, KerasAnchorBasedDatagen
)


# Mock dataset loader for testing
class MockDatasetLoader(BaseDataloader):
    def __init__(
        self, 
        num_samples: int = 100, 
        img_height=416, 
        img_width=416, 
        num_channels=1, 
        use_embedded_classes: bool = False
    ):
        self.num_samples = num_samples
        self.current_idx = 0
        self.img_height = img_height
        self.img_width = img_width
        self.num_channels = num_channels
        self.use_embedded_classes = use_embedded_classes
        
    def __len__(self) -> int:
        return self.num_samples
        
    def __iter__(self):
        self.current_idx = 0
        return self
        
    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.current_idx >= self.num_samples:
            raise StopIteration
            
        # Generate mock image and annotations
        image = np.random.rand(self.img_height, self.img_width, self.num_channels).astype(np.float32)
        if self.use_embedded_classes:
            image = Image(image)
        num_objects = np.random.randint(1, 4)
        
        # Generate random boxes and labels
        boxes = []
        for _ in range(num_objects):
            x_center = np.random.uniform(0.2, 0.8)
            y_center = np.random.uniform(0.2, 0.8)
            width = np.random.uniform(0.1, 0.3)
            height = np.random.uniform(0.1, 0.3)
            label = np.random.randint(0, 3)  # 3 classes
            if self.use_embedded_classes:
                bbox = BBoxClassId(
                    coordinates=[x_center, y_center, width, height],
                    class_id=label,
                    img_height=self.img_height,
                    img_width=self.img_width
                )
            else:
                bbox = [x_center, y_center, width, height, label]
            boxes.append(bbox)
            
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
    def configs(self, root_dir):
        with open(os.path.join(str(root_dir), 'tests_data/configs/anchor_based_datagen.yaml')) as stream:
            configs = yaml.safe_load(stream)
        return DotMap(configs["TestAnchorsBasedDataloader"])
    
    @pytest.fixture
    def mock_loader(self, configs):
        return MockDatasetLoader(**configs.MockDatasetLoader)
           
    @pytest.fixture
    def dataloader(self, mock_loader, configs):
        datagen_configs = configs.AnchorsBasedDatagen
        return AnchorsBasedDatagen(
            dataloader=mock_loader,
            anchors=datagen_configs.anchors,
            target_size=tuple(datagen_configs.target_size),
            grid_size=tuple(datagen_configs.grid_size),
            num_classes=2,
            batch_size=datagen_configs.batch_size
        )
    
    def test_initialization(self, dataloader, mock_loader, configs):
        """Test dataloader initialization."""
        datagen_configs = configs.AnchorsBasedDatagen
        assert dataloader.dataloader == mock_loader
        assert np.array_equal(dataloader.anchors, datagen_configs.anchors)
        assert dataloader.target_size == tuple(datagen_configs.target_size)
        assert dataloader.grid_size == tuple(datagen_configs.grid_size)
        assert dataloader.num_classes == datagen_configs.num_classes
        assert dataloader.batch_size == datagen_configs.batch_size
        assert not dataloader.do_augment
        
    def test_length(self, dataloader, configs):
        """Test length calculation."""
        num_samples = configs.MockDatasetLoader.num_samples
        batch_size = configs.AnchorsBasedDatagen.batch_size
        assert len(dataloader) == math.ceil(num_samples / batch_size)
        
        # Test with different batch sizes
        dataloader.batch_size = 2
        assert len(dataloader) == math.ceil(num_samples / 2)
        
        dataloader.batch_size = 3
        assert len(dataloader) == math.ceil(num_samples / 3)
        
    def test_batch_generation(self, dataloader, configs):
        """Test batch generation with Anchor Based models specific grid format encoding.
        
        In Anchor Based models (like YOLO v2), each grid cell predicts B bounding boxes using anchor boxes.
        For each box we predict 5 + num_classes values:
        - tx, ty: Center coordinates offset relative to grid cell (0 to 1)
        - tw, th: Width and height as log-scale transformations relative to anchors (unbounded)
        - confidence: Objectness score (0 to 1)
        - class probabilities: One score per class (0 to 1)
        """
        datagen_configs = configs.AnchorsBasedDatagen
        images, labels = next(iter(dataloader))
        
        # Check shapes
        expected_image_shape = (datagen_configs.batch_size, *datagen_configs.target_size, 
                            configs.MockDatasetLoader.num_channels)
        expected_label_shape = (datagen_configs.batch_size, *datagen_configs.grid_size,
                            len(datagen_configs.anchors), 5 + datagen_configs.num_classes)
        
        assert images.shape == expected_image_shape
        assert labels.shape == expected_label_shape
        
        # Images should be normalized to [0,1]
        assert np.all((images >= 0) & (images <= 1))
        
        # tx, ty: Box center coordinates relative to grid cell
        # These are sigmoid activated in the network, so should be in [0,1]
        assert np.all((labels[..., 0:2] >= 0) & (labels[..., 0:2] <= 1))
        
        # tw, th: Width and height relative to anchors
        # These use a log-space transform: t = log(truth/anchor)
        # Can be negative (box smaller than anchor) or positive (box larger than anchor)
        # No range assertions needed as they can be any real number
        
        # Objectness score (confidence) should be in [0,1] 
        # This is sigmoid activated in the network
        assert np.all((labels[..., 4] >= 0) & (labels[..., 4] <= 1))
        
        # Class probabilities should be in [0,1]
        # These are softmax activated in the network
        assert np.all((labels[..., 5:] >= 0) & (labels[..., 5:] <= 1))
        
    def test_batch_size(self, dataloader, configs):
        """Test different batch sizes."""
        datagen_configs = configs.AnchorsBasedDatagen
        target_size = datagen_configs.target_size
        grid_size = datagen_configs.grid_size
        num_anchors = len(datagen_configs.anchors)
        num_classes = datagen_configs.num_classes
        num_channels = configs.MockDatasetLoader.num_channels
        
        test_batch_size = 2
        dataloader.batch_size = test_batch_size
        images, labels = next(iter(dataloader))
        
        assert images.shape == (test_batch_size, *target_size, num_channels)
        assert labels.shape == (test_batch_size, *grid_size, num_anchors, 5 + num_classes)
        
    def test_augmentation(self, dataloader, configs):
        """Test with augmentations."""
        datagen_configs = configs.AnchorsBasedDatagen
        target_size = datagen_configs.target_size
        grid_size = datagen_configs.grid_size
        num_anchors = len(datagen_configs.anchors)
        num_classes = datagen_configs.num_classes
        batch_size = datagen_configs.batch_size
        num_channels = configs.MockDatasetLoader.num_channels
        
        dataloader.transform = A.Compose([
            A.RandomBrightnessContrast(p=1)
        ], bbox_params=A.BboxParams(format='yolo'))
        
        assert dataloader.do_augment
        images, labels = next(iter(dataloader))
        
        assert images.shape == (batch_size, *target_size, num_channels)
        assert labels.shape == (batch_size, *grid_size, num_anchors, 5 + num_classes)
        
    @pytest.mark.parametrize("show_grid,show_anchors", [
        (True, True),
        (True, False),
        (False, True),
        (False, False)
    ])
    def test_visualization(self, dataloader: AnchorsBasedDatagen, show_grid, show_anchors, monkeypatch):
        """Test visualization with different options."""
        
        # Mock plt.show() per i test
        shown = False
        def mock_show():
            nonlocal shown
            shown = True
        monkeypatch.setattr(plt, 'show', mock_show)
        
        try:
            dataloader.visualize_sample(0, show_grid=show_grid, show_anchors=show_anchors)
            # Verifica che plt.show() sia stato chiamato
            assert shown, "Plot was not displayed"
        except Exception as e:
            pytest.fail(f"Visualization failed: {e}")


class TestFrameworksAnchorsBasedDataloader:
    
    @pytest.fixture
    def configs(self, root_dir):
        with open(os.path.join(str(root_dir), 'tests_data/configs/anchor_based_datagen.yaml')) as stream:
            configs = yaml.safe_load(stream)
        return DotMap(configs["TestFrameworksAnchorsBasedDataloader"])
    
    @pytest.fixture
    def mock_loader(self, configs):
        return MockDatasetLoader(**configs.MockDatasetLoader)

    @pytest.fixture
    def tensorflow_config(self):
        """Configure TensorFlow to use CPU only."""
        tf.config.set_visible_devices([], 'GPU')
        physical_devices = tf.config.list_physical_devices('CPU')
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except:
            pass
            
    @pytest.fixture
    def transform(self):
        """Test augmentation transform."""
        return A.Compose([
            A.RandomBrightnessContrast(p=0.5),
            A.GaussNoise(p=0.5)
        ], bbox_params=A.BboxParams(format='yolo'))
    
    def test_pytorch_dataloader(self, tensorflow_config, mock_loader, configs, transform):
        """Test PyTorch dataloader implementation."""
        print("\nTesting PyTorch Dataloader...")
        
        datagen_configs = configs.AnchorsBasedDatagen
        # Initialize dataset
        dataset = PyTorchAnchorBasedDatagen(
            dataloader=mock_loader,
            anchors=np.array(datagen_configs.anchors),
            target_size=tuple(datagen_configs.target_size),
            grid_size=tuple(datagen_configs.grid_size),
            num_classes=datagen_configs.num_classes,
            transform=transform
        )
        
        # Create dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=datagen_configs.batch_size,
            shuffle=True,
            collate_fn=PyTorchAnchorBasedDatagen.collate_fn
        )
        
        # Test model
        model = SimplePyTorchModel(
            num_classes=datagen_configs.num_classes,
            num_anchors=len(datagen_configs.anchors)
        )
        optimizer = torch.optim.Adam(model.parameters())
        
        # Test training loop
        model.train()
        for batch_idx, (images, targets) in enumerate(dataloader):
            # Verify shapes
            assert images.shape == (datagen_configs.batch_size, 1, *datagen_configs.target_size)
            assert targets.shape == (datagen_configs.batch_size, *datagen_configs.grid_size, 
                                  len(datagen_configs.anchors), 5 + datagen_configs.num_classes)
            
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
                
    def test_keras_dataloader(self, tensorflow_config, mock_loader, configs, transform):
        """Test Keras dataloader implementation."""
        print("\nTesting Keras Dataloader...")
        
        datagen_configs = configs.AnchorsBasedDatagen
        # Initialize dataset
        dataset = KerasAnchorBasedDatagen(
            dataloader=mock_loader,
            anchors=np.array(datagen_configs.anchors),
            target_size=tuple(datagen_configs.target_size),
            grid_size=tuple(datagen_configs.grid_size),
            num_classes=datagen_configs.num_classes,
            batch_size=datagen_configs.batch_size,
            transform=transform
        )
        
        # Verify dataset can generate batches correctly
        for _ in range(2):
            batch = next(iter(dataset))
            assert len(batch) == 2  # images and labels
            images, labels = batch
            assert images.shape == (datagen_configs.batch_size, *datagen_configs.target_size, 1)
            assert labels.shape == (datagen_configs.batch_size, *datagen_configs.grid_size, 
                                 len(datagen_configs.anchors), 5 + datagen_configs.num_classes)

        # Create and compile model with mixed precision disabled
        tf.keras.mixed_precision.set_global_policy('float32')
        model = create_simple_keras_model(
            input_shape=(*datagen_configs.target_size, 1),
            num_classes=datagen_configs.num_classes,
            num_anchors=len(datagen_configs.anchors)
        )
        
        model.compile(
            optimizer='adam',
            loss='mse',  # Dummy loss for testing
            run_eagerly=True  # For better error messages
        )
        
        # Test training with reduced steps
        history = model.fit(
            dataset,
            epochs=1,
            steps_per_epoch=2,
            verbose=1
        )
        
        assert len(history.history['loss']) == 1
        print("Keras training completed successfully")            

# TODO: Add test with DataLoader integrated with Image and BBox classes OR integrate that classes in MockDatasetLoader