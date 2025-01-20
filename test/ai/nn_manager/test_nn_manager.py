import pytest
import os
import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort
import tensorflow as tf
import tf2onnx
import onnx
from onnx import helper, TensorProto
from saltup.utils.misc import suppress_stdout

from saltup.ai.nn_manager import NeuralNetworkManager
from saltup.ai.keras_utils.keras_to_onnx import convert_keras_to_onnx
from onnx import helper, TensorProto

# Fixture to create sample models and return their paths
@pytest.fixture(scope="session")
def sample_models():
    # Directory to store sample models
    model_dir = "sample_models"
    os.makedirs(model_dir, exist_ok=True)

    # Define paths for sample models
    model_paths = {
        "pt": os.path.join(model_dir, "sample_model.pt"),
        "keras": os.path.join(model_dir, "sample_model.keras"),
        "onnx": os.path.join(model_dir, "sample_model.onnx"),
        "tflite": os.path.join(model_dir, "sample_model.tflite"),
    }

    # Create a simple TensorFlow/Keras model
    def create_simple_model():
        input_layer = tf.keras.layers.Input(shape=(28, 28, 1), name="input")

        # Add layers
        x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Flatten()(x)
        output_layer = tf.keras.layers.Dense(10, activation='softmax')(x)

        # Create the model
        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer, name="functional_model")
        return model

    # Create and save a PyTorch model
    if not os.path.exists(model_paths["pt"]):
        def create_pytorch_model():
            class SimpleModel(nn.Module):
                def __init__(self, input_shape, output_shape):
                    super(SimpleModel, self).__init__()
                    self.input_shape = input_shape  # (28, 28, 1)
                    self.output_shape = output_shape  # (100, 28)
                    
                    # Define a convolutional neural network
                    self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
                    self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
                    self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
                    self.fc1 = nn.Linear(64 * 14 * 14, 128)  # After pooling, the spatial dimensions are halved
                    self.fc2 = nn.Linear(128, output_shape[0] * output_shape[1])  # Output shape: (100, 28)
                    self.relu = nn.ReLU()

                def forward(self, x):
                    # Input shape: (batch_size, 1, 28, 28)
                    x = self.relu(self.conv1(x))
                    x = self.pool(x)  # Output shape: (batch_size, 32, 14, 14)
                    x = self.relu(self.conv2(x))
                    x = self.pool(x)  # Output shape: (batch_size, 64, 7, 7)
                    x = x.view(x.size(0), -1)  # Flatten the tensor
                    x = self.relu(self.fc1(x))
                    x = self.fc2(x)
                    x = x.view(x.size(0), self.output_shape[0], self.output_shape[1])  # Reshape to (100, 28)
                    return x

            # Define input and output shapes
            input_shape = (28, 28, 1)  # (height, width, channels)
            output_shape = (100, 28)   # (num_features, sequence_length)

            # Instantiate the model
            model = SimpleModel(input_shape, output_shape)
            torch.save(model.state_dict(), model_paths["pt"])
        #create_pytorch_model()

    # Create and save a TensorFlow/Keras model (.keras)
    if not os.path.exists(model_paths["keras"]):
        model = create_simple_model()
        model.save(model_paths["keras"])

    # Create and save an ONNX model
    if not os.path.exists(model_paths["onnx"]):
        input_shape = [1, 28, 28, 1]
        output_shape = [128, 10]

        input1 = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)

        # Define a dummy graph (identity operation for simplicity)
        identity_node = helper.make_node('Identity', inputs=['input'], outputs=['output'])

        # Create the graph
        graph = helper.make_graph(
            [identity_node],
            'simple_yolo_graph',
            [input1],
            [output]
        )

        # Create the model with opset version 21
        model = helper.make_model(graph, producer_name='onnx-yolo-example', opset_imports=[helper.make_opsetid("", 16)])

        # Save the model to the output path
        onnx.save(model, model_paths["onnx"])

    # Create and save a TensorFlow Lite model
    if not os.path.exists(model_paths["tflite"]):
        model = create_simple_model()
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        with open(model_paths["tflite"], "wb") as f:
            f.write(tflite_model)

    return model_paths

# Fixture to create an instance of NeuralNetworkManager
@pytest.fixture
def nn_manager():
    return NeuralNetworkManager()

# Test loading models of different formats
@pytest.mark.parametrize("model_format", ["pt", "keras", "onnx", "tflite"])
def test_load_model(nn_manager, sample_models, model_format):
    model_path = sample_models[model_format]
    model, input_shape, output_shape = nn_manager.load_model(model_path)

    assert model is not None
    assert isinstance(input_shape, tuple)
    assert isinstance(output_shape, tuple)

# Test inference for each model format
@pytest.mark.parametrize("model_format", ["pt", "keras", "onnx", "tflite"])
def test_model_inference(nn_manager, sample_models, model_format):
    model_path = sample_models[model_format]

    # Load the model
    model, input_shape, _ = nn_manager.load_model(model_path)
    
    print(input_shape)
    # Generate random input data based on the model's input shape
    if model_format == "pt":
        input_data = torch.randn(1, *input_shape)
    else:
        input_data = np.random.uniform(1, 12, (1, 28, 28, 1)).astype(np.float32)
        
    # Perform inference
    output = nn_manager.model_inference(input_data)

    # Check that the output is not None
    assert output is not None

# Test unsupported model format
def test_unsupported_format(nn_manager):
    unsupported_model_path = "unsupported_model.xyz"
    with pytest.raises(ValueError):
        nn_manager.load_model(unsupported_model_path)

# Test inference without loading a model
def test_inference_without_loading_model(nn_manager):
    input_data = np.random.randn(1, 28, 28, 1).astype(np.float32)
    with pytest.raises(RuntimeError):
        nn_manager.model_inference(input_data)

# Test get_inference_time_ms before and after inference
def test_inference_time(nn_manager, sample_models):
    model_path = sample_models["keras"]

    # Load the model
    model, input_shape, _ = nn_manager.load_model(model_path)

    # Generate random input data
    input_data = np.random.uniform(1, 12, (1, 28, 28 ,1)).astype(np.float32)

    # Perform inference
    output = nn_manager.model_inference(input_data)

    # Check that the inference time is recorded
    inference_time = nn_manager.get_inference_time_ms()
    print(inference_time)
    assert inference_time is not None
    assert isinstance(inference_time, float)
    assert inference_time > 0