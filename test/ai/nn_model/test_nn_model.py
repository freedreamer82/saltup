import pytest
import os
import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort
import tensorflow as tf
import keras 
import tf2onnx
import onnx
from onnx import helper, TensorProto
from saltup.utils.misc import suppress_stdout

from saltup.ai.nn_model import NeuralNetworkModel, ModelType
from saltup.ai.utils.keras.to_onnx import convert_keras_to_onnx
import torch.nn.functional as F
# Move the SimpleModel class definition outside the fixture
class SimpleModel(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(SimpleModel, self).__init__()
        self.input_shape = input_shape  # (1, 28, 28)
        #self.output_shape = output_shape  # (10,)
        self.fc = nn.Linear(1 * 28 * 28, output_shape[0])

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input
        x = self.fc(x)
        return x
    

# Fixture to create sample models and return their paths
@pytest.fixture(scope="session")
def sample_models(tmp_path_factory, request):
    # Directory to store sample models
    model_dir = tmp_path_factory.mktemp("sample_models")

    # Define paths for sample models
    model_paths = {
        "pt": str(model_dir / "sample_model.pt"),
        "keras": str(model_dir / "sample_model.keras"),
        "h5": str(model_dir / "sample_model.h5"),
        "onnx": str(model_dir / "sample_model.onnx"),
        "tflite": str(model_dir / "sample_model.tflite"),
    }

    # Create a simple TensorFlow/Keras model
    def create_simple_model():
        input_layer = keras.layers.Input(shape=(28, 28, 1), name="input")

        # Add layers
        x = keras.layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
        x = keras.layers.MaxPooling2D((2, 2))(x)
        x = keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)
        x = keras.layers.Flatten()(x)
        output_layer = keras.layers.Dense(10, activation='softmax')(x)

        # Create the model
        model = keras.models.Model(inputs=input_layer, outputs=output_layer, name="functional_model")
        return model

    # Create and save a PyTorch model
    if not os.path.exists(model_paths["pt"]):
        def create_pytorch_model():
            input_shape = (1, 28, 28)  # channels first for PyTorch
            output_shape = (10,)   # 10 classes
            model = SimpleModel(input_shape, output_shape)
            return model
        
        model = create_pytorch_model()
        scripted = torch.jit.script(model.cpu())
        scripted.save(model_paths["pt"])

    # Create and save TensorFlow/Keras models
    if not os.path.exists(model_paths["keras"]):
        model = create_simple_model()
        model.save(model_paths["keras"])
    
    if not os.path.exists(model_paths["h5"]):
        model = create_simple_model()
        model.save(model_paths["h5"])

    # Create and save an ONNX model
    if not os.path.exists(model_paths["onnx"]):
        input_shape = [1, 1, 28, 28]  # NCHW format
        output_shape = [1, 10]

        input1 = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)

        # Define a dummy graph (identity operation for simplicity)
        identity_node = helper.make_node('Identity', inputs=['input'], outputs=['output'])

        # Create the graph
        graph = helper.make_graph(
            [identity_node],
            'simple_model_graph',
            [input1],
            [output]
        )

        # Create the model with opset version 16
        model = helper.make_model(graph, producer_name='onnx-example', opset_imports=[helper.make_opsetid("", 16)])

        # Save the model
        onnx.save(model, model_paths["onnx"])

    # Create and save a TensorFlow Lite model
    if not os.path.exists(model_paths["tflite"]):
        model = create_simple_model()
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        with open(model_paths["tflite"], "wb") as f:
            f.write(tflite_model)

    return model_paths


class TestNeuralNetworkModel:
    """Test class for NeuralNetworkModel functionality."""

    # Test loading models of different formats
    @pytest.mark.parametrize("model_format", ["pt", "keras", "h5", "onnx", "tflite"])
    def test_load_model(self, sample_models, model_format):
        """Test loading models of different formats."""
        model_path = sample_models[model_format]
        nn_model = NeuralNetworkModel(model_path)
        
        assert nn_model.is_loaded
        assert nn_model.get_model() is not None
        assert isinstance(nn_model.get_input_shape(), tuple)
        assert isinstance(nn_model.get_output_shape(), tuple)

    # Test model type detection
    @pytest.mark.parametrize("model_format,expected_type", [
        ("pt", ModelType.TORCH),
        ("keras", ModelType.KERAS),
        ("h5", ModelType.KERAS),
        ("onnx", ModelType.ONNX),
        ("tflite", ModelType.TFLITE),
    ])
    def test_model_type_detection(self, sample_models, model_format, expected_type):
        """Test that model types are correctly detected."""
        model_path = sample_models[model_format]
        nn_model = NeuralNetworkModel(model_path)
        
        assert nn_model.get_model_type() == expected_type

    # Test inference for each model format
    @pytest.mark.parametrize("model_format", ["pt", "keras", "h5", "onnx", "tflite"])
    def test_model_inference(self, sample_models, model_format):
        """Test inference for each model format."""
        model_path = sample_models[model_format]
        nn_model = NeuralNetworkModel(model_path)
        
        # Generate appropriate input data based on model format
        if model_format == "pt":
            # PyTorch expects NCHW format
            input_data = torch.randn(1, 1, 28, 28)
        elif model_format == "onnx":
            # ONNX expects NCHW format as numpy array
            input_data = np.random.uniform(0, 1, (1, 1, 28, 28)).astype(np.float32)
        else:
            # Keras and TFLite expect NHWC format
            input_data = np.random.uniform(0, 1, (1, 28, 28, 1)).astype(np.float32)
            
        # Perform inference
        output = nn_model.model_inference(input_data)

        # Check that the output is not None and has reasonable shape
        assert output is not None
        assert isinstance(output, np.ndarray)
        assert len(output.shape) >= 1

    def test_model_properties(self, sample_models):
        """Test model property methods."""
        model_path = sample_models["keras"]
        nn_model = NeuralNetworkModel(model_path)

        # Test basic properties
        assert nn_model.is_loaded
        assert nn_model.get_model_size_bytes() > 0
        assert nn_model.get_num_parameters() > 0
        
        # Test supported formats
        formats = nn_model.get_supported_formats()
        expected_formats = [".pt", ".keras", ".h5", ".onnx", ".tflite"]
        assert formats == expected_formats

    def test_inference_time_measurement(self, sample_models):
        """Test that inference time is properly measured."""
        model_path = sample_models["keras"]
        nn_model = NeuralNetworkModel(model_path)

        # Before inference, time should be None
        assert nn_model.get_inference_time_ms() is None

        # Generate input data
        input_data = np.random.uniform(0, 1, (1, 28, 28, 1)).astype(np.float32)

        # Perform inference
        output = nn_model.model_inference(input_data)

        # Check that inference time is recorded
        inference_time = nn_model.get_inference_time_ms()
        assert inference_time is not None
        assert isinstance(inference_time, float)
        assert inference_time > 0

    def test_model_instance_loading(self):
        """Test loading model instances directly (not from file paths)."""
        # Test with Keras model instance
        input_layer = keras.layers.Input(shape=(10,))
        output_layer = keras.layers.Dense(5)(input_layer)
        keras_model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        
        nn_model = NeuralNetworkModel(keras_model)
        assert nn_model.is_loaded
        assert nn_model.get_model_type() == ModelType.KERAS

        # Test inference with model instance
        input_data = np.random.uniform(0, 1, (1, 10)).astype(np.float32)
        output = nn_model.model_inference(input_data)
        assert output is not None

    def test_pytorch_model_instance(self):
        """Test loading PyTorch model instance."""
        # Create a simple PyTorch model
        input_shape = (1, 28, 28)
        output_shape = (10,)
        torch_model = SimpleModel(input_shape, output_shape)
        
        nn_model = NeuralNetworkModel(torch_model)
        assert nn_model.is_loaded
        assert nn_model.get_model_type() == ModelType.TORCH

        # Test inference
        input_data = torch.randn(1, 1, 28, 28)
        output = nn_model.model_inference(input_data)
        assert output is not None
        assert isinstance(output, np.ndarray)

    def test_unsupported_format(self):
        """Test handling of unsupported model format."""
        with pytest.raises(ValueError, match="Model file not found"):
            NeuralNetworkModel("nonexistent_model.xyz")

    def test_model_not_loaded_errors(self):
        """Test that appropriate errors are raised when model is not loaded."""
        # Create an uninitialized model object
        nn_model = NeuralNetworkModel.__new__(NeuralNetworkModel)
        nn_model._is_loaded = False
        nn_model.model = None

        with pytest.raises(RuntimeError, match="Model is not loaded"):
            nn_model.get_model()

        with pytest.raises(RuntimeError, match="Model is not loaded"):
            nn_model.get_input_shape()

        with pytest.raises(RuntimeError, match="Model is not loaded"):
            nn_model.get_output_shape()

        with pytest.raises(RuntimeError, match="Model is not loaded"):
            nn_model.model_inference(np.array([1, 2, 3]))

    def test_multiple_inferences(self, sample_models):
        """Test that multiple inferences work correctly and update timing."""
        model_path = sample_models["keras"]
        nn_model = NeuralNetworkModel(model_path)

        input_data = np.random.uniform(0, 1, (1, 28, 28, 1)).astype(np.float32)

        # Perform multiple inferences
        output1 = nn_model.model_inference(input_data)
        time1 = nn_model.get_inference_time_ms()

        output2 = nn_model.model_inference(input_data)
        time2 = nn_model.get_inference_time_ms()

        # Check that both inferences worked
        assert output1 is not None
        assert output2 is not None
        assert time1 > 0
        assert time2 > 0
        
        # Times should be different (though could be very close)
        assert isinstance(time1, float)
        assert isinstance(time2, float)


class TestModelType:
    """Test ModelType enum functionality."""
    
    def test_model_type_to_string(self):
        """Test ModelType enum to_string method."""
        assert ModelType.KERAS.to_string() == "keras_model"
        assert ModelType.TORCH.to_string() == "torch_model"
        assert ModelType.ONNX.to_string() == "onnx_model"
        assert ModelType.TFLITE.to_string() == "tflite_model"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])