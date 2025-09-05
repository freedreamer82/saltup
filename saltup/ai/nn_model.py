from typing import List, Tuple, Any
import torch
import os
import onnxruntime as ort
import onnx
import numpy as np

import tensorflow as tf
#from tf_keras.saving import load_model
import keras
import time

from saltup.utils.misc import suppress_stdout
from saltup.saltup_env import SaltupEnv
from enum import IntEnum

class ModelType(IntEnum):
    KERAS = 1
    TORCH = 2
    ONNX = 3
    TFLITE = 4
    
    def to_string(self):
        """Convert the ModelType enum to a human-readable string."""
        if self == ModelType.KERAS:
            return "keras_model"
        elif self == ModelType.TORCH:
            return "torch_model"
        elif self == ModelType.ONNX:
            return "onnx_model"
        elif self == ModelType.TFLITE:
            return "tflite_model"
        else:
            raise ValueError(f"Unknown ModelType: {self}")

class NeuralNetworkModel:
    """Class to manage loading and inference for different neural network model formats or instances."""

    def __init__(self, model_or_path: Any):
        self.model = None
        self.model_path = None
        self.supported_formats = [".pt", ".pth", ".keras", ".h5", ".onnx", ".tflite"]
        self.inference_time_ms = None
        self._is_loaded = False
        self.input_shape = None
        self.output_shape = None
        self.model_size_bytes = None
        self.num_parameters = None
        self._model_type = self._set_type(model_or_path)
        
        self.model, self.input_shape, self.output_shape = self._load(model_or_path)
    
    def _set_type(self, model_or_path):
        if isinstance(model_or_path, str):
            if not os.path.exists(model_or_path):
                raise ValueError(f"Model file not found: {model_or_path}")
            if model_or_path.endswith(".pt") or model_or_path.endswith(".pth"):
                return ModelType.TORCH
            elif model_or_path.endswith(".keras") or model_or_path.endswith(".h5"):
                return ModelType.KERAS
            elif model_or_path.endswith(".onnx"):
                return ModelType.ONNX
            elif model_or_path.endswith(".tflite"):
                return ModelType.TFLITE
        elif isinstance(model_or_path, torch.nn.Module):
            return ModelType.TORCH
        elif isinstance(model_or_path, keras.Model):
            return ModelType.KERAS
        elif isinstance(model_or_path, ort.InferenceSession):
            return ModelType.ONNX
        elif isinstance(model_or_path, tf.lite.Interpreter):
            return ModelType.TFLITE
        else:
            raise ValueError("Cannot determine model type from the given input.")
        
    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._is_loaded
    
    def get_model_type(self) -> ModelType:
        """Return the type of the model."""
        return self._model_type
    
    def get_model(self) -> Any:
        """Return the loaded model instance."""
        if not self._is_loaded:
            raise RuntimeError("Model is not loaded. Call `load_model` first.")
        return self.model
    
    def get_input_shape(self) -> Tuple[Any]:
        """Return the input shape of the model."""
        if not self._is_loaded:
            raise RuntimeError("Model is not loaded. Call `load_model` first.")
        return self.input_shape
    
    def get_output_shape(self) -> Tuple[Any]:
        """Return the output shape of the model."""
        if not self._is_loaded:
            raise RuntimeError("Model is not loaded. Call `load_model` first.")
        return self.output_shape

    def get_supported_formats(self) -> List[str]:
        """Return a list of supported model formats."""
        return self.supported_formats

    def _load(self, model_or_path:Any) -> Tuple[Any, Tuple[Any], Tuple[Any]]:
        """
        Load a model from the given path or instance based on its type or file extension.

        Returns:
            Tuple containing the loaded model, input shape, and output shape.

        Raises:
            ValueError: If the model format is not supported or the file does not exist.
        """

        if self._is_loaded:
            return self.model, self.input_shape, self.output_shape
        
        with suppress_stdout():
            if self._model_type == ModelType.TORCH:
                # Device configuration
                device_config = SaltupEnv.SALTUP_PYTORCH_DEVICE
                if device_config == "auto":
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                else:
                    device = torch.device(device_config)
                    
                if not isinstance(model_or_path, str):
                    self.model = model_or_path
                else:
                    # Load TORCH model
                    self.model_path = model_or_path
                    try:
                        self.model = torch.load(self.model_path, map_location=device, weights_only=False)  # Try standard loading first
                    except:
                        self.model = torch.jit.load(self.model_path, map_location=device)  # Generic TORCH model loading
                    # Model size
                    self.model_size_bytes = os.path.getsize(self.model_path)
                
                self.model.eval()  # Set the model to evaluation mode
                # Assuming the model has an attribute `input_shape` or similar
                if hasattr(self.model, 'input_shape'):
                    self.input_shape =  self.model.input_shape
                
                # Assuming the model has an attribute `output_shape` or similar
                if hasattr(self.model, 'output_shape'):
                    self.output_shape =  self.model.output_shape
                else:
                    # If no output_shape attribute, infer from a forward pass (requires example input)
                    if hasattr(self.model, 'input_shape') and self.model.input_shape is not None:
                        try:
                            example_input = torch.randn(1, *self.model.input_shape)  # Example input
                            with torch.no_grad():
                                output = self.model(example_input)
                            self.output_shape = output.shape[1:]  # Exclude batch size
                        except Exception as e:
                            raise RuntimeError(f"Failed to infer output shape from input_shape: {self.model.input_shape}. Error: {e}")
                    else:
                        raise RuntimeError("Cannot infer output shape: model does not have a valid 'input_shape' attribute. Please define an 'input_shape' attribute for the model")
                    self.output_shape =  output.shape[1:]  # Exclude batch size
                self._is_loaded = True
                # Model parameters
                self.num_parameters = sum(p.numel() for p in self.model.parameters())
                return self.model, self.input_shape, self.output_shape
                
            elif self._model_type == ModelType.KERAS:
                
                if not isinstance(model_or_path, str):
                    self.model = model_or_path
                else:
                    self.model_path = model_or_path
                    # Load TensorFlow/Keras model (supports both .keras and .h5 formats)
                    try:
                        self.model = keras.models.load_model(self.model_path, compile=False, safe_mode=False)
                    except:
                        raise RuntimeError(f"Failed to load Keras model from path: {self.model_path}")
                        #self.model = load_model(self.model_path, compile=False, safe_mode=False)
                    # Model size and parameters
                    self.model_size_bytes = os.path.getsize(self.model_path)
                self.input_shape = self.model.input_shape  # Exclude the batch size
                self.output_shape = self.model.output_shape  # Exclude the batch size
                self._is_loaded = True
                # Model parameters
                self.num_parameters = self.model.count_params()
                return self.model, self.input_shape, self.output_shape

            
            elif self._model_type == ModelType.ONNX:
                use_gpu = SaltupEnv.SALTUP_NN_MNG_USE_GPU
                providers = ort.get_available_providers()
                
                if use_gpu and "CUDAExecutionProvider" in providers:
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                else:
                    providers = ["CPUExecutionProvider"]
                
                if not isinstance(model_or_path, str):
                    self.model = model_or_path
                else:
                    self.model_path = model_or_path
                    # Load ONNX model with specified providers
                    self.model = ort.InferenceSession(self.model_path, providers=providers)
                    # Model size and parameters
                    self.model_size_bytes = os.path.getsize(self.model_path)
                    # Count ONNX parameters (sum all initializers)
                    model_proto = onnx.load(self.model_path)
                    self.num_parameters = sum(
                        int(np.prod(init.dims)) for init in model_proto.graph.initializer
                    )
                input_metadata = self.model.get_inputs()[0]
                self.input_shape = tuple(input_metadata.shape)
                output_metadata = self.model.get_outputs()[0]
                self.output_shape = tuple(output_metadata.shape)
                self._is_loaded = True
                return self.model, self.input_shape, self.output_shape

            elif self._model_type == ModelType.TFLITE:
                
                if not isinstance(model_or_path, str):
                    self.model = model_or_path
                else:
                    self.model_path = model_or_path
                    # Load TensorFlow Lite model
                    self.model = tf.lite.Interpreter(model_path=self.model_path)
                    # Model size (parameters not easily available)
                    self.model_size_bytes = os.path.getsize(self.model_path)
                
                self.model.allocate_tensors()  # Allocate tensors for inference
                # Get the input shape from the interpreter
                input_details = self.model.get_input_details()[0]
                self.input_shape = tuple(input_details['shape'])
            
                output_details = self.model.get_output_details()[0]
                self.output_shape =  tuple(output_details['shape'])
                self._is_loaded = True
                self.num_parameters = None
                return self.model, self.input_shape, self.output_shape

            else:
                raise ValueError(f"Unsupported model format. Supported formats are: {self.supported_formats}")

    def get_model_size_bytes(self) -> int:
        """Return the model size in bytes."""
        return self.model_size_bytes

    def get_num_parameters(self) -> int:
        """Return the number of parameters in the model."""
        return self.num_parameters

    def model_inference(self, input_data: Any) -> Any:
        """
        Perform inference using the loaded model and measure the inference time.

        Args:
            input_data: Preprocessed input data for the model.

        Returns:
            Raw output from the model.

        Raises:
            RuntimeError: If the model is not loaded.
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call `load_model` first.")

        start_time = time.time()  # Capture start time

        with suppress_stdout():
            if self._model_type == ModelType.TORCH:
                # TORCH inference
                with torch.no_grad():
                    # Device configuration
                    device_config = SaltupEnv.SALTUP_PYTORCH_DEVICE
                    if device_config == "auto":
                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    else:
                        device = torch.device(device_config)
                    if isinstance(input_data, np.ndarray):
                        input_data = torch.from_numpy(input_data)
                    input_data = input_data.to(torch.float32)  # Ensure float32 dtype
                    input_data = input_data.to(device)
                    self.model.to(device)
                    output = self.model(input_data)
                    output = output.cpu().numpy()
            elif self._model_type == ModelType.KERAS:
                # TensorFlow/Keras inference
                output = self.model.predict(input_data)
            elif self._model_type == ModelType.ONNX:
                if isinstance(input_data, torch.Tensor):
                    input_data = input_data.cpu().numpy()
                elif not isinstance(input_data, np.ndarray):
                    input_data = np.array(input_data)                
                # ONNX inference
                input_name = self.model.get_inputs()[0].name
                input_data = input_data.astype(np.float32)  # Ensure input is float32
                output = self.model.run(None, {input_name: input_data})[0]
            elif self._model_type == ModelType.TFLITE:
                # TensorFlow Lite inference
                input_details = self.model.get_input_details()
                output_details = self.model.get_output_details()
                self.model.allocate_tensors()

                input_index = input_details[0]['index']
                output_index = output_details[0]['index']
                
                if input_details[0]['dtype'] == np.float32:
                    input_data = input_data.astype(np.float32)
                else:
                    scale, zero_point = input_details[0]["quantization"]
                    dtype = input_details[0]['dtype']
                    input_data = (input_data / scale + zero_point).astype(dtype)

                if input_data.shape != tuple(input_details[0]['shape']):
                    print('Resizing input tensor to:', input_data.shape)
                    self.model.resize_tensor_input(input_index, input_data.shape)
                    self.model.allocate_tensors()

                self.model.set_tensor(input_index, input_data)
                self.model.invoke()
                
                # Get output tensor
                output = self.model.get_tensor(output_index)

            else:
                raise RuntimeError("Unsupported model type.")

            end_time = time.time()  # Capture end time
            self.inference_time_ms = (end_time - start_time) * 1000  # Calculate inference time in milliseconds

            return output

    def get_inference_time_ms(self) -> float:
        """
        Get the last measured inference time in milliseconds.

        Returns:
            The last inference time in milliseconds, or None if no inference has been performed yet.
        """
        return self.inference_time_ms