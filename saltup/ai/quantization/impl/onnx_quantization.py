import onnx
import os
import cv2
import random
import shutil
import numpy as np
import onnxruntime

from onnxruntime.quantization import QuantType, CalibrationDataReader, quantize_static, QuantFormat, quantize_dynamic


def quantize_onnx_model_dynamic(input_model_path:str, output_model_path:str, per_channel:bool=False):
    """quantize an onnx model

    Args:
        input_model_path (str): path of the input model
        output_model_path (str): path of the output model
    """
    # Load the ONNX model
    #model = onnx.load(input_model_path)
    
    # Verify the model
    #onnx.checker.check_model(model)
    
    # Perform dynamic quantization
    quantize_dynamic(
    model_input=input_model_path,
    model_output=output_model_path,
    per_channel=per_channel,  # Set to True if you want per-channel quantization (usually for Conv layers)
    weight_type=QuantType.QUInt8  # Quantize weights to uint8
    )
    
    print(f"Quantized model saved to: {output_model_path}")
    
    # Optional: Compare model sizes
    original_size = os.path.getsize(input_model_path) / (1024 * 1024)
    quantized_size = os.path.getsize(output_model_path) / (1024 * 1024)
    
    print(f"Original model size: {original_size:.2f} MB")
    print(f"Quantized model size: {quantized_size:.2f} MB")
    print(f"Size reduction: {(1 - quantized_size/original_size) * 100:.2f}%")
    
    