import os
import shutil
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt

import onnx
from onnx import numpy_helper, utils

import onnxruntime
from onnxruntime import InferenceSession
from onnxruntime import quantization
from onnxruntime.quantization import (
    QuantType,
    CalibrationDataReader,
    quantize_static,
    QuantFormat,
    quantize_dynamic,
)

from saltup.ai.base_dataformat.base_dataloader import BaseDataloader


class OnnxCalibrationDataReader(CalibrationDataReader):
    def __init__(self, dataloader: BaseDataloader, model_path: str, preprocess_fn: callable = lambda x : x):
        self.enum_data = None

        # Use inference session to get input shape.
        session = onnxruntime.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
        input_shape = session.get_inputs()[0].shape
        print(f"Model input shape: {input_shape}")

        # Extract height and width from input shape (assume NHWC or NCHW)
        if len(input_shape) == 4:
            # Try to infer format: if first dim is None or 1, assume NCHW or NHWC
            if input_shape[1] in [3, 1]:  # Likely NCHW
                height, width = input_shape[2], input_shape[3]
            else:  # Likely NHWC
                height, width = input_shape[1], input_shape[2]
        else:
            raise ValueError("Unexpected input shape for model: {}".format(input_shape))

        
        # Process data
        self.data_list = [preprocess_fn(image.get_data(), height, width) for image, label in dataloader]

        self.input_name = session.get_inputs()[0].name
        self.datasize = len(self.data_list)

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter([{self.input_name: data} for data in self.data_list])
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None


def print_model_info(model_path):
    """
    Prints detailed information about an ONNX model.

    Args:
        model_path (str): Path to the ONNX model file.

    Returns:
        tuple: A tuple containing the loaded ONNX model and the ONNX Runtime inference session.

    Example:
        model, session = print_model_info("model.onnx")
    """
    model = onnx.load(model_path)
    session = InferenceSession(model.SerializeToString())
    
    print("\nModel Details:")
    for input in session.get_inputs():
        print(f"Input '{input.name}':")
        print(f"  Shape: {input.shape}")
        print(f"  Type: {input.type}")
    
    # Print the full structure of the first input
    input_shape = model.graph.input[0].type.tensor_type.shape.dim
    print("\nDetailed structure of the first input:")
    for i, dim in enumerate(input_shape):
        print(f"Dimension {i}: {dim.dim_value}")
    
    return model, session