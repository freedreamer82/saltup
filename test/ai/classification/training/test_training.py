import pytest
import numpy as np
import os
import torch
import tensorflow as tf
from unittest.mock import MagicMock
from saltup.ai.classification.training.training import evaluate_model
from saltup.ai.object_detection.dataset.base_dataset import BaseDataloader
from saltup.ai.object_detection.datagenerator.base_datagen import BasedDatagenerator
from saltup.ai.classification.datagenerator.classification_datagen import ClassificationDataloader, keras_ClassificationDataGenerator, pytorch_ClassificationDataGenerator
from PIL import Image

@pytest.fixture
def mock_test_data_dir(tmp_path):
    # Create a mock test data directory with class subfolders and temporary jpg images
    class_names = ["class_0", "class_1"]
    for class_name in class_names:
        class_dir = tmp_path / class_name
        class_dir.mkdir()
        for i in range(2):  # Create 2 images per class
            img_path = class_dir / f"image_{i}.jpg"
            # Generate a random image matrix and save it as an image
            random_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            image = Image.fromarray(random_image)
            image.save(img_path)
    return str(tmp_path)

@pytest.fixture
def mock_keras_model(tmp_path):
    # Create a mock Keras model and save it
    model = tf.keras.Sequential([tf.keras.layers.Dense(2, activation="softmax")])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model_path = str(tmp_path / "mock_model.keras")
    model.save(model_path)
    return model_path

@pytest.fixture
def mock_pytorch_model(tmp_path):
    # Create a mock PyTorch model and save it
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(10, 2)

        def forward(self, x):
            return self.fc(x)

    model = MockModel()
    model_path = str(tmp_path / "mock_model.pt")
    torch.save(model, model_path)
    return model_path

@pytest.fixture
def mock_tflite_model(tmp_path):
    # Create a mock TFLite model and save it
    model = tf.keras.Sequential([tf.keras.layers.Dense(2, activation="softmax")])
    model._set_save_spec(None)  # Explicitly set the save spec to avoid AttributeError
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    model_path = str(tmp_path / "mock_model.tflite")
    with open(model_path, "wb") as f:
        f.write(tflite_model)
    return model_path

@pytest.fixture
def mock_test_gen(mock_test_data_dir):
    # Create a mock ClassificationDataloader and DataGenerator
    class_dict = {"class_0": 0, "class_1": 1}
    dataloader = ClassificationDataloader(root_dir=mock_test_data_dir, classes_dict=class_dict, img_size=(224, 224, 3))

    keras_gen = keras_ClassificationDataGenerator(
        dataloader=dataloader,
        target_size=(224, 224),
        num_classes=2,
        batch_size=4
    )
    return keras_gen

def test_evaluate_model_keras(mock_keras_model, mock_test_gen):
    accuracy = evaluate_model(mock_keras_model, mock_test_gen)
    assert isinstance(accuracy, float)

def test_evaluate_model_pytorch(mock_pytorch_model, mock_test_gen):
    def mock_loss_function(outputs, labels):
        return torch.tensor(0.5)
    class_dict = {"class_0": 0, "class_1": 1}
    dataloader = ClassificationDataloader(root_dir=mock_test_data_dir, classes_dict=class_dict, img_size=(224, 224, 3))
    pytorch_gen = pytorch_ClassificationDataGenerator(
        dataloader=dataloader,
        target_size=(224, 224),
        num_classes=2,
        batch_size=4
    )
    accuracy = evaluate_model(mock_pytorch_model, pytorch_gen, loss_function=mock_loss_function)
    assert isinstance(accuracy, float)

def test_evaluate_model_tflite(mock_tflite_model, mock_test_gen):
    accuracy = evaluate_model(mock_tflite_model, mock_test_gen)
    assert isinstance(accuracy, float)

def test_evaluate_model_invalid_model_type(mock_test_gen):
    with pytest.raises(ValueError, match="Unsupported model type"):
        evaluate_model("invalid_model.xyz", mock_test_gen)

def test_evaluate_model_missing_loss_function(mock_pytorch_model, mock_test_gen):
    pytorch_gen = pytorch_ClassificationDataGenerator(
        dataloader=mock_test_gen.dataloader,
        target_size=(224, 224),
        num_classes=2,
        batch_size=4
    )
    with pytest.raises(ValueError, match="please provide a loss_function"):
        evaluate_model(mock_pytorch_model, pytorch_gen)
