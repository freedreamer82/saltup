from typing import List, Tuple, Any
import torch
import onnxruntime as ort
import tensorflow as tf
import numpy as np
from tf_keras.saving import load_model
import time

class NeuralNetworkManager:
    """Class to manage loading and inference for different neural network model formats."""
    
    def __init__(self):
        self.model = None
        self.supported_formats = [".pt", ".keras", ".h5", ".onnx", ".tflite"]
        self.inference_time_ms = None

    def get_supported_formats(self) -> List[str]:
        """Return a list of supported model formats."""
        return self.supported_formats

    def load_model(self, model_path: str) -> Tuple[Any, Tuple[Any], Tuple[Any]]:
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
            # Assuming the model has an attribute `input_shape` or similar
            if hasattr(self.model, 'input_shape'):
                model_input_shape =  self.model.input_shape
            
            # Assuming the model has an attribute `output_shape` or similar
            if hasattr(self.model, 'output_shape'):
                model_output_shape =  self.model.output_shape
            else:
                # If no output_shape attribute, infer from a forward pass (requires example input)
                example_input = torch.randn(1, *self.model.input_shape)  # Example input
                with torch.no_grad():
                    output = self.model(example_input)
                model_output_shape =  output.shape[1:]  # Exclude batch size
            
            return self.model, model_input_shape, model_output_shape
            
        elif model_path.endswith(".keras") or model_path.endswith(".h5"):
            # Load TensorFlow/Keras model (supports both .keras and .h5 formats)
            self.model = load_model(model_path, compile=False, safe_mode=False)  # Usa tf_keras.saving.load_model
            model_input_shape = self.model.input_shape[1:]  # Exclude the batch size
            
            model_output_shape = self.model.output_shape[1:]  # Exclude the batch size
            
            return self.model, model_input_shape, model_output_shape

        
        elif model_path.endswith(".onnx"):
            # Load ONNX model
            self.model = ort.InferenceSession(model_path)
            input_metadata = self.model.get_inputs()[0]
            model_input_shape = tuple(input_metadata.shape)
            output_metadata = self.model.get_outputs()[0]
            model_output_shape = tuple(output_metadata.shape)
            return self.model, model_input_shape, model_output_shape

        elif model_path.endswith(".tflite"):
            # Load TensorFlow Lite model
            self.model = tf.lite.Interpreter(model_path=model_path)
            self.model.allocate_tensors()  # Allocate tensors for inference
            # Get the input shape from the interpreter
            input_details = self.model.get_input_details()[0]
            model_input_shape = tuple(input_details['shape'])
        
            output_details = self.model.get_output_details()[0]
            model_output_shape =  tuple(output_details['shape'])
            
            return self.model, model_input_shape, model_output_shape

            
        else:
            raise ValueError(f"Unsupported model format. Supported formats are: {self.supported_formats}")


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
            output = self.model.run(None, {input_name: input_data})[0]
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