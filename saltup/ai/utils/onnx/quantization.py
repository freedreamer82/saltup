import numpy as np
from typing import Callable
from copy import deepcopy
import json
import os
import random

import onnx
import onnxruntime
from onnxruntime import InferenceSession

from saltup.ai.utils.onnx.onnx import print_model_info
from saltup.ai.base_dataformat.base_dataloader import BaseDataloader


def generate_quantization_config(sensitivity_results: dict) -> dict:
    """
    Generates a quantization configuration based on sensitivity analysis results.

    Args:
        sensitivity_results (dict): Dictionary mapping layer names to sensitivity metrics and quantization safety.

    Returns:
        dict: Configuration dictionary containing:
            - 'safe_to_quantize': List of layer names safe for quantization.
            - 'avoid_quantization': List of layer names to avoid quantizing.
            - 'statistics': Dictionary with total, quantizable, and sensitive layer counts.
    """
    config = {
        'safe_to_quantize': [],
        'avoid_quantization': [],
        'statistics': {
            'total_layers': len(sensitivity_results),
            'quantizable_layers': 0,
            'sensitive_layers': 0
        }
    }

    for layer_name, results in sensitivity_results.items():
        if results['safe_to_quantize']:
            config['safe_to_quantize'].append(layer_name)
            config['statistics']['quantizable_layers'] += 1
        else:
            config['avoid_quantization'].append(layer_name)
            config['statistics']['sensitive_layers'] += 1

    return config

def print_sensitivity_report(results):
    print("\nQuantization Sensitivity Report:")
    print("-" * 80)

    for node_name, metrics in results.items():
        print(f"\nNode: {node_name}")
        print(f"Type: {metrics['op_type']}")
        print(f"MSE: {metrics['mse']:.6f}")
        print(f"Maximum Difference: {metrics['max_difference']:.6f}")
        print(f"Cosine Similarity: {metrics['cosine_similarity']:.6f}")
        print(f"Safe to Quantize: {'Yes' if metrics['safe_to_quantize'] else 'No'}")
        print("-" * 40)

def analyze_layer_sensitivity(
    model_path: str,
    calibration_dataloader: BaseDataloader,
    num_samples: int = 100,
    preprocess_fn: Callable = lambda x: x,
    exclude_nodes=None
) -> dict:
    """
    Analyze the sensitivity of each layer in an ONNX model to simulated INT8 quantization.

    This function loads an ONNX model and runs inference on a batch of calibration images.
    For each layer (node), it simulates quantization of the layer's initializers (weights),
    runs inference, and compares the output to the original (float) model predictions.
    It computes metrics such as MSE, maximum absolute difference, and cosine similarity
    between the outputs. Layers that meet certain thresholds are considered "safe to quantize".

    Args:
        model_path (str): Path to the ONNX model file.
        calibration_data_path (str): Path to the folder containing calibration images.
        num_samples (int, optional): Number of images to use for calibration. Default is 100.
        exclude_nodes (list, optional): List of node names to skip during analysis.

    Returns:
        dict: Dictionary mapping node names to sensitivity metrics and quantization safety.
    """
    print("Analyzing model...")
    model, session = print_model_info(model_path)

    if exclude_nodes is None:
        exclude_nodes = []


    model_input = session.get_inputs()[0]
    input_name = model_input.name
    print(f"\nModel input name: {input_name}")
    _, height, width, channels = model_input.shape
    print(f"Expected input shape: {model_input.shape} (N, {height}, {width}, {channels})")
    
    for cnt, (image, _) in enumerate(calibration_dataloader):
        if preprocess_fn:
            image = preprocess_fn(image, height, width)
        
        if cnt >= num_samples:
            break
        
        try:
            input_data = {input_name: image.get_data()}
            print("\nTesting base inference...")
            base_preds = session.run(None, input_data)[0]
            print(f"Base predictions shape: {base_preds.shape}")
        except Exception as e:
            print(f"\nError during inference: {str(e)}")
            print(f"Expected shape: {session.get_inputs()[0].shape}")
            raise

        results = {}

        # Add problematic layer to exclusion list
        if exclude_nodes is None:
            exclude_nodes = []

        total_nodes = len([node for node in model.graph.node])
        print(f"\nAnalyzing {total_nodes} layers...")

        for i, node in enumerate(model.graph.node, 1):
            if node.name in exclude_nodes:
                print(f"Skipping node: {node.name}")
                continue

            print(f"Analyzing layer {i}/{total_nodes}: {node.name} ({node.op_type})")
            test_model = deepcopy(model)

            # Simulate INT8 quantization for the layer's initializers
            for init in test_model.graph.initializer:
                if init.name in node.input:
                    # Load tensor data
                    data = np.frombuffer(init.raw_data, dtype=np.float32)

                    # Check that tensor is not empty
                    if data.size == 0:
                        print(f"Warning: initializer {init.name} is empty, skipping quantization.")
                        continue  # Ignore empty initializers

                    # Determine signed/unsigned INT8 range
                    qmin, qmax = -127, 127  # Signed INT8
                    scale = max(np.max(np.abs(data)) / qmax, 1e-8)  # Minimum tolerance to avoid division by zero

                    # Simulated quantization (quantize and then de-quantize)
                    quant_data = np.clip(np.round(data / scale), qmin, qmax) * scale

                    # Update tensor with quantized data
                    init.raw_data = quant_data.astype(np.float32).tobytes()

            try:
                test_session = InferenceSession(test_model.SerializeToString())
                test_preds = test_session.run(None, input_data)[0]

                mse = np.mean((base_preds - test_preds) ** 2)
                max_diff = np.max(np.abs(base_preds - test_preds))
                cosine_sim = np.mean([
                    np.dot(p1.flatten(), p2.flatten()) /
                    (np.linalg.norm(p1.flatten()) * np.linalg.norm(p2.flatten()))
                    for p1, p2 in zip(base_preds, test_preds)
                ])

                # Convert numpy values to standard Python types
                results[node.name] = {
                    'mse': float(mse),
                    'max_difference': float(max_diff),
                    'cosine_similarity': float(cosine_sim),
                    'op_type': str(node.op_type),
                    'safe_to_quantize': bool(mse < 1e-4 and max_diff < 0.01 and cosine_sim > 0.99)
                }
            except Exception as e:
                print(f"Error analyzing node {node.name}: {str(e)}")
                continue

    return results
