import pytest
from unittest import mock
import numpy as np
import torch
import os
import tensorflow as tf

from saltup.ai.classification.datagenerator.classification_datagen import keras_ClassificationDataGenerator, pytorch_ClassificationDataGenerator
from saltup.ai.classification.training.training import evaluate_model, train_model

@pytest.fixture
def mock_test_data_dir(tmp_path):
    # Create mock class folders
    class_names = ['class0', 'class1']
    for cname in class_names:
        (tmp_path / cname).mkdir()
    return str(tmp_path)

@pytest.fixture
def dummy_images_labels():
    images = np.random.rand(2, 224, 224, 3).astype(np.float32)
    labels = np.array([[1, 0], [0, 1]])  # One-hot
    return images, labels

def test_evaluate_model_keras(monkeypatch, dummy_images_labels, mock_test_data_dir):
    mock_model = mock.Mock()
    mock_model.predict.return_value = np.array([[0.9, 0.1], [0.2, 0.8]])

    monkeypatch.setattr(tf.keras.models, "load_model", lambda _: mock_model)
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: None)  # suppress prints

    class DummyGen:
        def __len__(self): return 1
        def __iter__(self): return iter([dummy_images_labels])
    
    acc = evaluate_model("model.keras", DummyGen(), mock_test_data_dir)
    assert 0.0 <= acc <= 1.0

def test_evaluate_model_pytorch(monkeypatch, dummy_images_labels, mock_test_data_dir):
    mock_model = mock.Mock()
    outputs = torch.tensor([[2.0, 1.0], [1.0, 2.0]])
    mock_model.return_value = outputs
    mock_model.eval = lambda: None

    def dummy_load(path, weights_only=False):
        return mock_model

    monkeypatch.setattr(torch, "load", dummy_load)
    monkeypatch.setattr("torch.no_grad", lambda: mock.MagicMock())
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: None)

    class DummyGen(torch.utils.data.Dataset):
        def __getitem__(self, index):
            # Simulate image + one-hot encoded label (for 2 classes)
            image = torch.rand(3, 224, 224)
            label = torch.tensor([[0.0, 1.0]])  # one-hot encoded
            return image, label

        def __len__(self):
            return 2

        batch_size = 1

    criterion = torch.nn.CrossEntropyLoss()
    acc = evaluate_model("model.pt", DummyGen(), mock_test_data_dir, criterion)
    assert 0.0 <= acc <= 1.0

def test_evaluate_model_invalid(monkeypatch, dummy_images_labels, mock_test_data_dir):
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: None)

    with pytest.raises(ValueError):
        evaluate_model("model.unknown", dummy_images_labels, mock_test_data_dir)

def test_evaluate_model_pytorch_missing_criterion(dummy_images_labels, mock_test_data_dir):
    with pytest.raises(ValueError):
        evaluate_model("model.pt", dummy_images_labels, mock_test_data_dir, criterion=None)


@pytest.fixture
def tmp_output_dir(tmp_path):
    return str(tmp_path)

@pytest.fixture
def keras_dummy_data():
    class DummyGen:
        def __len__(self): return 1
        def __iter__(self): return iter([(
            np.random.rand(2, 224, 224, 3),
            np.array([[1, 0], [0, 1]])
        )])
    return DummyGen()

def test_train_model_keras(monkeypatch, tmp_output_dir, keras_dummy_data):
    model = mock.MagicMock(spec=tf.keras.Model)
    fit_history = mock.Mock()
    fit_history.history = {
        'loss': [0.5, 0.3],
        'val_loss': [0.6, 0.4],
        'accuracy': [0.8, 0.9],
        'val_accuracy': [0.75, 0.85],
    }
    model.fit.return_value = fit_history
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: None)  # Suppress output

    result_path = train_model(
        model=model,
        train_gen=keras_dummy_data,
        val_gen=keras_dummy_data,
        output_dir=tmp_output_dir,
        epochs=2,
        optimizer=None,
        criterion=None,
        scheduler=None
    )

    assert result_path.endswith('.keras')
    assert os.path.exists(result_path)

@pytest.fixture
def pytorch_dummy_data():
    class DummyDataset(torch.utils.data.Dataset):
        def __getitem__(self, index):
            return torch.rand(3, 224, 224), torch.tensor([1, 0])
        def __len__(self): return 4
    train_gen = torch.utils.data.DataLoader(DummyDataset(), batch_size=2)
    val_gen = torch.utils.data.DataLoader(DummyDataset(), batch_size=2)
    return train_gen, val_gen

def test_train_model_pytorch(monkeypatch, tmp_output_dir, pytorch_dummy_data):
    model = mock.MagicMock(spec=torch.nn.Module)
    model.forward = mock.Mock(return_value=torch.rand(2, 2))
    model.eval = lambda: None
    model.train = lambda: None
    model.to = lambda device: model

    optimizer = torch.optim.SGD(params=model.parameters() if hasattr(model, "parameters") else [], lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = mock.Mock()
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: None)

    path = train_model(
        model=model,
        train_gen=pytorch_dummy_data[0],
        val_gen=pytorch_dummy_data[1],
        output_dir=tmp_output_dir,
        epochs=1,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler
    )
    assert path.endswith('.pt')
    assert os.path.exists(path)