from typing import List, Any
import torch
import onnxruntime as ort
import tensorflow as tf
import numpy as np
from tf_keras.saving import load_model  # Importazione specifica di load_model da tf_keras.saving
import time

class NeuralNetworkManager:
    """Class to manage loading and inference for different neural network model formats."""
    
    def __init__(self):
        self.model = None
        self.supported_formats = [".pt", ".keras", ".h5", ".onnx", ".tflite"]
        self.inference_time_ms = None  # Attributo per memorizzare il tempo di inferenza

    def get_supported_formats(self) -> List[str]:
        """Return a list of supported model formats."""
        return self.supported_formats

    def load_model(self, model_path: str) -> Any:
        """
        Load a model from the given path based on its file extension.

        Args:
            model_path: Path to the model file.

        Returns:
            The loaded model.

        Raises:
            ValueError: If the model format is not supported.
        """
        if model_path.endswith(".pt"):
            # Load PyTorch model
            self.model = torch.load(model_path)  # Generic PyTorch model loading
            self.model.eval()  # Set the model to evaluation mode
        elif model_path.endswith(".keras") or model_path.endswith(".h5"):
            # Load TensorFlow/Keras model (supports both .keras and .h5 formats)
            self.model = load_model(model_path, compile=False, safe_mode=False)  # Usa tf_keras.saving.load_model
        elif model_path.endswith(".onnx"):
            # Load ONNX model
            self.model = ort.InferenceSession(model_path)
        elif model_path.endswith(".tflite"):
            # Load TensorFlow Lite model
            self.model = tf.lite.Interpreter(model_path=model_path)
            self.model.allocate_tensors()  # Allocate tensors for inference
        else:
            raise ValueError(f"Unsupported model format. Supported formats are: {self.supported_formats}")
        return self.model

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

        if isinstance(self.model, torch.nn.Module):
            # PyTorch inference
            with torch.no_grad():
                output = self.model(input_data)
        elif isinstance(self.model, tf.keras.Model):
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