import pytest
import numpy as np
import os
import torch
from torch import nn
import tensorflow as tf
from torch.utils.data import DataLoader
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

class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
        self.fc = torch.nn.Linear(224 * 224, 2)

    def forward(self, x):
        x = x.view(x.size(0), 224 * 224)  # Flatten the batch of inputs
        return self.fc(x)

@pytest.fixture
def mock_pytorch_model(tmp_path):
    # Create a mock PyTorch model and save it
    model = MockModel()
    model_path = str(tmp_path / "mock_model.pt")
    scripted = torch.jit.script(model.cpu())
    scripted.save(model_path)
    return model_path

@pytest.fixture
def mock_tflite_model(tmp_path):
    # Create a mock TFLite model and save it
    model = tf.keras.Sequential([tf.keras.layers.Dense(2, activation="softmax")])
    model.build(input_shape=(224, 224, 3))  # Build the model with an input shape

    # Convert the Keras model to TFLite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the TFLite model to a temporary path
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


@pytest.fixture
def mock_test_pytorch_gen(mock_test_data_dir):
    # Create a mock ClassificationDataloader and DataGenerator
    class_dict = {"class_0": 0, "class_1": 1}
    dataloader = ClassificationDataloader(root_dir=mock_test_data_dir, classes_dict=class_dict, img_size=(224, 224, 3))

    pytorch_gen = pytorch_ClassificationDataGenerator(
        dataloader=dataloader,
        target_size=(224, 224),
        num_classes=2,
        batch_size=4
    )
    pytorch_gen = DataLoader(pytorch_gen, batch_size=4)
    return pytorch_gen

def mock_loss_function(outputs, labels):
    # Mock loss value using outputs and labels
    return torch.mean((outputs - labels.float()) ** 2)

def test_evaluate_model_pytorch_with_output_dir(mock_pytorch_model, mock_test_pytorch_gen, tmp_path):
    output_dir = str(tmp_path / "output")
    # Example outputs and labels for testing
    outputs = torch.tensor([[0.8, 0.2], [0.4, 0.6]])
    labels = torch.tensor([[1, 0], [0, 1]])
    loss_function = lambda outputs, labels: mock_loss_function(outputs, labels)
    #loss_function = nn.CrossEntropyLoss()
    accuracy = evaluate_model(mock_pytorch_model, mock_test_pytorch_gen, output_dir=output_dir, loss_function=loss_function)

    assert isinstance(accuracy, float)
    assert os.path.exists(output_dir)
    assert any(fname.endswith("_pt_confusion_matrix.png") for fname in os.listdir(output_dir))

def test_evaluate_model_tflite_with_output_dir(mock_tflite_model, mock_test_gen, tmp_path):
    output_dir = str(tmp_path / "output")
    accuracy = evaluate_model(mock_tflite_model, mock_test_gen, output_dir=output_dir)

    assert isinstance(accuracy, float)
    assert os.path.exists(output_dir)
    assert any(fname.endswith("_tflite_confusion_matrix.png") for fname in os.listdir(output_dir))

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
