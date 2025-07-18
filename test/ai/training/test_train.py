import pytest
import os
import numpy as np
from PIL import Image
from unittest.mock import Mock, patch, MagicMock, PropertyMock

import tensorflow as tf
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from saltup.ai.classification.datagenerator import (
    ClassificationDataloader,
    keras_ClassificationDataGenerator,
    pytorch_ClassificationDataGenerator,
)
from saltup.ai.training.train import _train_model, training
from saltup.ai.training.callbacks import CallbackContext
from saltup.saltup_env import SaltupEnv


@pytest.fixture
def mock_test_data_dir(tmp_path):
    """Create a mock test data directory with class subfolders and temporary jpg images."""
    class_names = ["class_0", "class_1"]
    for class_name in class_names:
        class_dir = tmp_path / class_name
        class_dir.mkdir()
        for i in range(2):  # Create 2 images per class
            img_path = class_dir / f"image_{i}.jpg"
            # Generate a random image matrix and save it as an image
            random_image = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
            image = Image.fromarray(random_image)
            image.save(img_path)
    return str(tmp_path)


@pytest.fixture
def mock_keras_model():
    """Create a mock Keras model."""
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(32, 32, 3)),
        tf.keras.layers.Dense(2, activation="softmax")
    ])
    return model


class PyTorchModel(nn.Module):
    def __init__(self, num_classes=2):
        super(PyTorchModel, self).__init__()
        self.fc = nn.Linear(32 * 32 * 3, num_classes)  # Fully connected layer

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = F.softmax(self.fc(x), dim=1)  # Apply softmax activation
        return x


@pytest.fixture
def mock_pytorch_model():
    """Create a mock PyTorch model."""
    return PyTorchModel(num_classes=2)


@pytest.fixture
def mock_keras_data_generator(mock_test_data_dir):
    """Create a mock Keras data generator."""
    class_dict = {"class_0": 0, "class_1": 1}
    dataloader = ClassificationDataloader(
        source=mock_test_data_dir, 
        classes_dict=class_dict, 
        img_size=(32, 32, 3)
    )
    return keras_ClassificationDataGenerator(
        dataloader=dataloader,
        target_size=(32, 32),
        num_classes=2,
        batch_size=4
    )


@pytest.fixture
def mock_pytorch_data_generator(mock_test_data_dir):
    """Create a mock PyTorch data generator."""
    class_dict = {"class_0": 0, "class_1": 1}
    dataloader = ClassificationDataloader(
        source=mock_test_data_dir, 
        classes_dict=class_dict, 
        img_size=(32, 32, 3)
    )
    return pytorch_ClassificationDataGenerator(
        dataloader=dataloader,
        target_size=(32, 32),
        num_classes=2,
        batch_size=4
    )


class TestTrainModel:
    """Test the _train_model function."""
    
    def test_train_model_keras(self, mock_keras_model, mock_keras_data_generator, tmp_path):
        """Test training a Keras model."""
        output_dir = str(tmp_path / "output")
        os.makedirs(output_dir, exist_ok=True)
        
        with patch.object(type(SaltupEnv), 'SALTUP_TRAINING_KERAS_COMPILE_ARGS', new_callable=PropertyMock) as mock_compile, \
             patch.object(type(SaltupEnv), 'SALTUP_TRAINING_KERAS_FIT_ARGS', new_callable=PropertyMock) as mock_fit, \
             patch.object(type(SaltupEnv), 'SALTUP_KERAS_TRAIN_SHUFFLE', new_callable=PropertyMock) as mock_shuffle, \
             patch.object(type(SaltupEnv), 'SALTUP_KERAS_TRAIN_VERBOSE', new_callable=PropertyMock) as mock_verbose:
            
            mock_compile.return_value = {}
            mock_fit.return_value = {}
            mock_shuffle.return_value = True
            mock_verbose.return_value = 0
            
            # Define loss function and optimizer
            loss_function = tf.keras.losses.CategoricalCrossentropy()
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            
            # Train the model
            trained_model_path = _train_model(
                model=mock_keras_model,
                train_gen=mock_keras_data_generator,
                val_gen=mock_keras_data_generator,
                output_dir=output_dir,
                epochs=1,
                loss_function=loss_function,
                optimizer=optimizer,
                scheduler=None,
                model_output_name="test_model"
            )
            
            # Assertions
            assert os.path.exists(trained_model_path)
            assert trained_model_path.endswith(".keras")
            assert os.path.exists(os.path.join(output_dir, "saved_models"))
        
    def test_train_model_pytorch(self, mock_pytorch_model, mock_pytorch_data_generator, tmp_path):
        """Test training a PyTorch model."""
        output_dir = str(tmp_path / "output")
        os.makedirs(output_dir, exist_ok=True)
        
        with patch.object(type(SaltupEnv), 'SALTUP_TRAINING_PYTORCH_ARGS', new_callable=PropertyMock) as mock_pytorch_args, \
             patch.object(type(SaltupEnv), 'SALTUP_PYTORCH_DEVICE', new_callable=PropertyMock) as mock_device, \
             patch.object(type(SaltupEnv), 'SALTUP_KERAS_TRAIN_VERBOSE', new_callable=PropertyMock) as mock_verbose:
            
            mock_pytorch_args.return_value = {
                'early_stopping_patience': 0,
                'use_scheduler_per_epoch': False
            }
            mock_device.return_value = 'cpu'
            mock_verbose.return_value = 0
            
            # The new implementation expects PyTorch DataLoader directly
            # Convert our data generator to DataLoader format
            train_loader = DataLoader(mock_pytorch_data_generator, batch_size=4, shuffle=True)
            val_loader = DataLoader(mock_pytorch_data_generator, batch_size=4, shuffle=False)
            
            # Define loss function and optimizer
            loss_function = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(mock_pytorch_model.parameters(), lr=0.001)
            
            # Train the model
            trained_model_path = _train_model(
                model=mock_pytorch_model,
                train_gen=train_loader,
                val_gen=val_loader,
                output_dir=output_dir,
                epochs=1,
                loss_function=loss_function,
                optimizer=optimizer,
                scheduler=None,
                model_output_name="test_model"
            )
            
            # Assertions
            assert os.path.exists(trained_model_path)
            assert trained_model_path.endswith(".pt")
            assert os.path.exists(os.path.join(output_dir, "saved_models"))
        
    def test_train_model_keras_missing_optimizer(self, mock_keras_model, mock_keras_data_generator, tmp_path):
        """Test that training fails when optimizer is missing for Keras model."""
        output_dir = str(tmp_path / "output")
        os.makedirs(output_dir, exist_ok=True)
        
        loss_function = tf.keras.losses.CategoricalCrossentropy()
        
        with pytest.raises(ValueError, match="both `optimizer` and `loss_function` must be provided"):
            _train_model(
                model=mock_keras_model,
                train_gen=mock_keras_data_generator,
                val_gen=mock_keras_data_generator,
                output_dir=output_dir,
                epochs=1,
                loss_function=loss_function,
                optimizer=None,
                scheduler=None
            )


class TestTrainingFunction:
    """Test the main training function."""
    
    def test_training_without_kfold_keras(self, mock_keras_model, mock_keras_data_generator, tmp_path):
        """Test training without k-fold cross validation for Keras model."""
        output_dir = str(tmp_path / "output")
        os.makedirs(output_dir, exist_ok=True)
        
        with patch.object(type(SaltupEnv), 'SALTUP_TRAINING_KERAS_COMPILE_ARGS', new_callable=PropertyMock) as mock_compile, \
             patch.object(type(SaltupEnv), 'SALTUP_TRAINING_KERAS_FIT_ARGS', new_callable=PropertyMock) as mock_fit, \
             patch.object(type(SaltupEnv), 'SALTUP_KERAS_TRAIN_SHUFFLE', new_callable=PropertyMock) as mock_shuffle, \
             patch.object(type(SaltupEnv), 'SALTUP_KERAS_TRAIN_VERBOSE', new_callable=PropertyMock) as mock_verbose, \
             patch.object(type(SaltupEnv), 'SALTUP_ONNX_OPSET', new_callable=PropertyMock) as mock_opset, \
             patch('saltup.ai.training.train.convert_keras_to_onnx') as mock_onnx_conv, \
             patch('saltup.ai.training.train.tflite_conversion') as mock_tflite_conv:
            
            mock_compile.return_value = {}
            mock_fit.return_value = {}
            mock_shuffle.return_value = True
            mock_verbose.return_value = 0
            mock_opset.return_value = 16
            
            # Mock the conversion functions
            mock_onnx_conv.return_value = (Mock(), Mock())
            mock_tflite_conv.return_value = os.path.join(output_dir, "saved_models", "test_model_best.tflite")
            
            # Define loss function and optimizer
            loss_function = tf.keras.losses.CategoricalCrossentropy()
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            
            # Train the model
            result = training(
                train_DataGenerator=mock_keras_data_generator,
                model=mock_keras_model,
                loss_function=loss_function,
                optimizer=optimizer,
                epochs=1,
                output_dir=output_dir,
                validation=[0.8, 0.2],
                kfold_param={'enable': False},
                model_output_name="test_model"
            )
            
            # Assertions
            assert result['kfolds'] is False
            assert len(result['models_paths']) >= 1
            assert os.path.exists(os.path.join(output_dir, "options.txt"))
        
    def test_training_without_kfold_pytorch(self, mock_pytorch_model, mock_pytorch_data_generator, tmp_path):
        """Test training without k-fold cross validation for PyTorch model."""
        output_dir = str(tmp_path / "output")
        os.makedirs(output_dir, exist_ok=True)
        
        with patch.object(type(SaltupEnv), 'SALTUP_TRAINING_PYTORCH_ARGS', new_callable=PropertyMock) as mock_pytorch_args, \
             patch.object(type(SaltupEnv), 'SALTUP_PYTORCH_DEVICE', new_callable=PropertyMock) as mock_device, \
             patch.object(type(SaltupEnv), 'SALTUP_KERAS_TRAIN_VERBOSE', new_callable=PropertyMock) as mock_verbose, \
             patch('saltup.ai.training.train.convert_torch_to_onnx') as mock_torch_onnx_conv:
            
            mock_pytorch_args.return_value = {
                'early_stopping_patience': 0,
                'use_scheduler_per_epoch': False
            }
            mock_device.return_value = 'cpu'
            mock_verbose.return_value = 0
            
            # Define loss function and optimizer
            loss_function = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(mock_pytorch_model.parameters(), lr=0.001)
            
            # Train the model
            result = training(
                train_DataGenerator=mock_pytorch_data_generator,
                model=mock_pytorch_model,
                loss_function=loss_function,
                optimizer=optimizer,
                epochs=1,
                output_dir=output_dir,
                validation=[0.8, 0.2],
                kfold_param={'enable': False},
                model_output_name="test_model"
            )
            
            # Assertions
            assert result['kfolds'] is False
            assert len(result['models_paths']) >= 1
            assert os.path.exists(os.path.join(output_dir, "options.txt"))
        
    def test_training_with_kfold_keras(self, mock_keras_model, mock_keras_data_generator, tmp_path):
        """Test training with k-fold cross validation for Keras model."""
        output_dir = str(tmp_path / "output")
        os.makedirs(output_dir, exist_ok=True)
        
        with patch.object(type(SaltupEnv), 'SALTUP_TRAINING_KERAS_COMPILE_ARGS', new_callable=PropertyMock) as mock_compile, \
             patch.object(type(SaltupEnv), 'SALTUP_TRAINING_KERAS_FIT_ARGS', new_callable=PropertyMock) as mock_fit, \
             patch.object(type(SaltupEnv), 'SALTUP_KERAS_TRAIN_SHUFFLE', new_callable=PropertyMock) as mock_shuffle, \
             patch.object(type(SaltupEnv), 'SALTUP_KERAS_TRAIN_VERBOSE', new_callable=PropertyMock) as mock_verbose, \
             patch.object(type(SaltupEnv), 'SALTUP_ONNX_OPSET', new_callable=PropertyMock) as mock_opset, \
             patch('saltup.ai.training.train.convert_keras_to_onnx') as mock_onnx_conv, \
             patch('saltup.ai.training.train.tflite_conversion') as mock_tflite_conv:
            
            mock_compile.return_value = {}
            mock_fit.return_value = {}
            mock_shuffle.return_value = True
            mock_verbose.return_value = 0
            mock_opset.return_value = 16
            
            # Mock the conversion functions
            mock_onnx_conv.return_value = (Mock(), Mock())
            mock_tflite_conv.return_value = os.path.join(output_dir, "golden_model_folder", "test_model.tflite")
            
            # Define loss function and optimizer
            loss_function = tf.keras.losses.CategoricalCrossentropy()
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            
            # Mock the split method to return proper generators
            mock_split_generators = [mock_keras_data_generator, mock_keras_data_generator]
            
            with patch.object(mock_keras_data_generator, 'split', return_value=mock_split_generators):
                # Train the model
                result = training(
                    train_DataGenerator=mock_keras_data_generator,
                    model=mock_keras_model,
                    loss_function=loss_function,
                    optimizer=optimizer,
                    epochs=1,
                    output_dir=output_dir,
                    kfold_param={'enable': True, 'split': [0.8, 0.2]},
                    model_output_name="test_model"
                )
                
                # Assertions
                assert result['kfolds'] is True
                assert len(result['models_paths']) >= 1
                assert os.path.exists(os.path.join(output_dir, "options.txt"))
        
    def test_training_with_validation_generator(self, mock_keras_model, mock_keras_data_generator, tmp_path):
        """Test training with validation generator instead of split ratio."""
        output_dir = str(tmp_path / "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Define loss function and optimizer
        loss_function = tf.keras.losses.CategoricalCrossentropy()
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        with patch.object(type(SaltupEnv), 'SALTUP_TRAINING_KERAS_COMPILE_ARGS', new_callable=PropertyMock) as mock_compile, \
             patch.object(type(SaltupEnv), 'SALTUP_TRAINING_KERAS_FIT_ARGS', new_callable=PropertyMock) as mock_fit, \
             patch.object(type(SaltupEnv), 'SALTUP_KERAS_TRAIN_SHUFFLE', new_callable=PropertyMock) as mock_shuffle, \
             patch.object(type(SaltupEnv), 'SALTUP_KERAS_TRAIN_VERBOSE', new_callable=PropertyMock) as mock_verbose, \
             patch('saltup.ai.training.train.convert_keras_to_onnx') as mock_onnx_conv, \
             patch('saltup.ai.training.train.tflite_conversion') as mock_tflite_conv:
            
            mock_compile.return_value = {}
            mock_fit.return_value = {}
            mock_shuffle.return_value = True
            mock_verbose.return_value = 0
            
            mock_onnx_conv.return_value = (Mock(), Mock())
            mock_tflite_conv.return_value = os.path.join(output_dir, "saved_models", "test_model_best.tflite")
            
            # Train the model
            result = training(
                train_DataGenerator=mock_keras_data_generator,
                model=mock_keras_model,
                loss_function=loss_function,
                optimizer=optimizer,
                epochs=1,
                output_dir=output_dir,
                validation=mock_keras_data_generator,  # Use generator instead of split
                kfold_param={'enable': False},
                model_output_name="test_model"
            )
            
            # Assertions
            assert result['kfolds'] is False
            assert len(result['models_paths']) >= 1
            
    def test_training_pytorch_missing_optimizer(self, mock_pytorch_model, mock_pytorch_data_generator, tmp_path):
        """Test that training fails when optimizer is missing for PyTorch model."""
        output_dir = str(tmp_path / "output")
        os.makedirs(output_dir, exist_ok=True)
        
        loss_function = nn.CrossEntropyLoss()
        
        with pytest.raises(ValueError, match="both `loss_function` and `optimizer` must be provided"):
            training(
                train_DataGenerator=mock_pytorch_data_generator,
                model=mock_pytorch_model,
                loss_function=loss_function,
                optimizer=None,
                epochs=1,
                output_dir=output_dir,
                kfold_param={'enable': False}
            )
            
    def test_training_class_weight_parameter(self, mock_keras_model, mock_keras_data_generator, tmp_path):
        """Test training with class weight parameter."""
        output_dir = str(tmp_path / "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Define loss function and optimizer
        loss_function = tf.keras.losses.CategoricalCrossentropy()
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        with patch.object(type(SaltupEnv), 'SALTUP_TRAINING_KERAS_COMPILE_ARGS', new_callable=PropertyMock) as mock_compile, \
             patch.object(type(SaltupEnv), 'SALTUP_TRAINING_KERAS_FIT_ARGS', new_callable=PropertyMock) as mock_fit, \
             patch.object(type(SaltupEnv), 'SALTUP_KERAS_TRAIN_SHUFFLE', new_callable=PropertyMock) as mock_shuffle, \
             patch.object(type(SaltupEnv), 'SALTUP_KERAS_TRAIN_VERBOSE', new_callable=PropertyMock) as mock_verbose, \
             patch('saltup.ai.training.train.convert_keras_to_onnx') as mock_onnx_conv, \
             patch('saltup.ai.training.train.tflite_conversion') as mock_tflite_conv:
            
            mock_compile.return_value = {}
            mock_fit.return_value = {}
            mock_shuffle.return_value = True
            mock_verbose.return_value = 0
            
            mock_onnx_conv.return_value = (Mock(), Mock())
            mock_tflite_conv.return_value = os.path.join(output_dir, "saved_models", "test_model_best.tflite")
            
            # Train the model with class weights
            result = training(
                train_DataGenerator=mock_keras_data_generator,
                model=mock_keras_model,
                loss_function=loss_function,
                optimizer=optimizer,
                epochs=1,
                output_dir=output_dir,
                kfold_param={'enable': False},
                model_output_name="test_model",
                classification_class_weight={0: 1.0, 1: 2.0}
            )
            
            # Assertions
            assert result['kfolds'] is False
            assert len(result['models_paths']) >= 1


class TestCallbackIntegration:
    """Test callback integration in training."""
    
    def test_pytorch_training_with_callbacks(self, mock_pytorch_model, mock_pytorch_data_generator, tmp_path):
        """Test PyTorch training with callbacks."""
        output_dir = str(tmp_path / "output")
        os.makedirs(output_dir, exist_ok=True)
        
        with patch.object(type(SaltupEnv), 'SALTUP_TRAINING_PYTORCH_ARGS', new_callable=PropertyMock) as mock_pytorch_args, \
             patch.object(type(SaltupEnv), 'SALTUP_PYTORCH_DEVICE', new_callable=PropertyMock) as mock_device, \
             patch.object(type(SaltupEnv), 'SALTUP_KERAS_TRAIN_VERBOSE', new_callable=PropertyMock) as mock_verbose:
            
            mock_pytorch_args.return_value = {
                'early_stopping_patience': 0,
                'use_scheduler_per_epoch': False
            }
            mock_device.return_value = 'cpu'
            mock_verbose.return_value = 0
            
            # Create a mock callback
            mock_callback = Mock()
            mock_callback.on_train_begin = Mock()
            mock_callback.on_epoch_end = Mock()
            mock_callback.on_train_end = Mock()
            
            # Convert to DataLoader
            train_loader = DataLoader(mock_pytorch_data_generator, batch_size=4, shuffle=True)
            val_loader = DataLoader(mock_pytorch_data_generator, batch_size=4, shuffle=False)
            
            # Define loss function and optimizer
            loss_function = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(mock_pytorch_model.parameters(), lr=0.001)
            
            # Train the model with callbacks
            trained_model_path = _train_model(
                model=mock_pytorch_model,
                train_gen=train_loader,
                val_gen=val_loader,
                output_dir=output_dir,
                epochs=1,
                loss_function=loss_function,
                optimizer=optimizer,
                scheduler=None,
                model_output_name="test_model",
                app_callbacks=[mock_callback]
            )
            
            # Assertions
            assert os.path.exists(trained_model_path)
            mock_callback.on_train_begin.assert_called_once()
            mock_callback.on_epoch_end.assert_called_once()
            mock_callback.on_train_end.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])