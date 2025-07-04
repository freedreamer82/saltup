from typing import List, Tuple, Any
import torch
import os
import onnxruntime as ort
import numpy as np

import tensorflow as tf
from tf_keras.saving import load_model
import time

from saltup.utils.misc import suppress_stdout
from saltup.saltup_env import SaltupEnv


class NeuralNetworkModel:
    """Class to manage loading and inference for different neural network model formats or instances."""

    def __init__(self, model_or_path: Any):
        self.model = None
        self.model_path = None
        self.supported_formats = [".pt", ".keras", ".h5", ".onnx", ".tflite"]
        self.inference_time_ms = None
        self._is_loaded = False
        self.input_shape = None
        self.output_shape = None
        self.model_size_bytes = None
        self.num_parameters = None

        # Accept either a path or an already loaded model
        if isinstance(model_or_path, str):
            if not os.path.exists(model_or_path):
                raise ValueError(f"Model file not found: {model_or_path}")
            self.model_path = model_or_path
        elif isinstance(model_or_path, torch.nn.Module):
            self.model = model_or_path
            self._is_loaded = True
            if hasattr(self.model, 'input_shape'):
                self.input_shape = self.model.input_shape
            if hasattr(self.model, 'output_shape'):
                self.output_shape = self.model.output_shape
            # PyTorch: count parameters
            self.num_parameters = sum(p.numel() for p in self.model.parameters())
            self.model_size_bytes = None  # Not available if loaded from instance
        elif hasattr(model_or_path, 'predict') and hasattr(model_or_path, 'input_shape'):
            # Assume Keras model
            self.model = model_or_path
            self.input_shape = self.model.input_shape
            self.output_shape = self.model.output_shape
            self._is_loaded = True
            # Keras: count parameters
            self.num_parameters = self.model.count_params()
            self.model_size_bytes = None  # Not available if loaded from instance
        else:
            raise ValueError("model_or_path must be a file path or a valid PyTorch/Keras model instance")
        
    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._is_loaded

    def get_supported_formats(self) -> List[str]:
        """Return a list of supported model formats."""
        return self.supported_formats

    def load(self) -> Tuple[Any, Tuple[Any], Tuple[Any]]:
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
            if self.model_path.endswith(".pt"):
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                # Load PyTorch model
                self.model = torch.jit.load(self.model_path, map_location=device)  # Generic PyTorch model loading
                self.model.eval()  # Set the model to evaluation mode
                # Assuming the model has an attribute `input_shape` or similar
                if hasattr(self.model, 'input_shape'):
                    self.input_shape =  self.model.input_shape
                
                # Assuming the model has an attribute `output_shape` or similar
                if hasattr(self.model, 'output_shape'):
                    self.output_shape =  self.model.output_shape
                else:
                    # If no output_shape attribute, infer from a forward pass (requires example input)
                    example_input = torch.randn(1, *self.model.input_shape)  # Example input
                    with torch.no_grad():
                        output = self.model(example_input)
                    self.output_shape =  output.shape[1:]  # Exclude batch size
                self._is_loaded = True
                # Model size and parameters
                self.model_size_bytes = os.path.getsize(self.model_path)
                self.num_parameters = sum(p.numel() for p in self.model.parameters())
                return self.model, self.input_shape, self.output_shape
                
            elif self.model_path.endswith(".keras") or self.model_path.endswith(".h5"):
                # Load TensorFlow/Keras model (supports both .keras and .h5 formats)
                try:
                    self.model = tf.keras.models.load_model(self.model_path, compile=False, safe_mode=False)
                except:
                    self.model = load_model(self.model_path, compile=False, safe_mode=False)
                self.input_shape = self.model.input_shape  # Exclude the batch size
                self.output_shape = self.model.output_shape  # Exclude the batch size
                self._is_loaded = True
                # Model size and parameters
                self.model_size_bytes = os.path.getsize(self.model_path)
                self.num_parameters = self.model.count_params()
                return self.model, self.input_shape, self.output_shape

            
            elif self.model_path.endswith(".onnx"):
                use_gpu = SaltupEnv.SALTUP_NN_MNG_USE_GPU
                providers = ort.get_available_providers()
                
                if use_gpu and "CUDAExecutionProvider" in providers:
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                else:
                    providers = ["CPUExecutionProvider"]
                
                # Load ONNX model with specified providers
                self.model = ort.InferenceSession(self.model_path, providers=providers)
                input_metadata = self.model.get_inputs()[0]
                self.input_shape = tuple(input_metadata.shape)
                output_metadata = self.model.get_outputs()[0]
                self.output_shape = tuple(output_metadata.shape)
                self._is_loaded = True
                # Model size and parameters
                self.model_size_bytes = os.path.getsize(self.model_path)
                # Count ONNX parameters (sum all initializers)
                import onnx
                model_proto = onnx.load(self.model_path)
                self.num_parameters = sum(
                    int(np.prod(init.dims)) for init in model_proto.graph.initializer
                )
                return self.model, self.input_shape, self.output_shape

            elif self.model_path.endswith(".tflite"):
                # Load TensorFlow Lite model
                self.model = tf.lite.Interpreter(model_path=self.model_path)
                self.model.allocate_tensors()  # Allocate tensors for inference
                # Get the input shape from the interpreter
                input_details = self.model.get_input_details()[0]
                self.input_shape = tuple(input_details['shape'])
            
                output_details = self.model.get_output_details()[0]
                self.output_shape =  tuple(output_details['shape'])
                self._is_loaded = True
                # Model size (parameters not easily available)
                self.model_size_bytes = os.path.getsize(self.model_path)
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
            if isinstance(self.model, torch.nn.Module):
                # PyTorch inference
                with torch.no_grad():
                    output = self.model(input_data)
            elif type(self.model).__name__ == 'Functional':  
                # TensorFlow/Keras inference
                output = self.model.predict(input_data)
            elif isinstance(self.model, ort.InferenceSession):
                # ONNX inference
                input_name = self.model.get_inputs()[0].name
                output = self.model.run(None, {input_name: input_data})
            elif isinstance(self.model, tf.lite.Interpreter):
                # TensorFlow Lite inference
                input_details = self.model.get_input_details()
                output_details = self.model.get_output_details()

                # Set input tensor
                self.model.set_tensor(input_details[0]['index'], input_data)

                # Run inference
                self.model.invoke()

                # Get output tensor
                output = self.model.get_tensor(output_details[0]['index'])
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