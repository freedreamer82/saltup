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
from saltup.utils.data.image import Image


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
    preprocess_fn: Callable = lambda x, h, w: x,
    exclude_nodes=None,
    skip_ops=None,
    mse_threshold: float = 1e-3,
    max_diff_threshold: float = 0.1,
    cosine_sim_threshold: float = 0.99
) -> dict:
    """
        Analyzes the sensitivity of each layer in an ONNX model to quantization.

        Args:
            model_path (str): Path to the ONNX model file.
            calibration_dataloader (BaseDataloader): Dataloader providing calibration images.
            num_samples (int, optional): Number of samples to use for calibration. Default is 100.
            preprocess_fn (Callable, optional): Function to preprocess images. Default is identity.
            exclude_nodes (list, optional): List of node names to exclude from analysis.
            skip_ops (set, optional): Set of operation types to skip during analysis. 
                By default, 'Mul' (multiplication) and 'Add' (addition) are skipped because 
                these operations are typically less sensitive to quantization errors compared 
                to other layers, and quantizing them often has minimal impact on model accuracy.
            mse_threshold (float, optional): Threshold for mean squared error to consider quantization safe.
            max_diff_threshold (float, optional): Threshold for maximum difference to consider quantization safe.
            cosine_sim_threshold (float, optional): Threshold for cosine similarity to consider quantization safe.

        Returns:
            dict: Dictionary mapping node names to sensitivity metrics and quantization safety.
    """
    print("Analyzing model...")
    model, session = print_model_info(model_path)

    if exclude_nodes is None:
        exclude_nodes = []
    if skip_ops is None:
        skip_ops = {"Mul", "Add"}  # default

    model_input = session.get_inputs()[0]
    input_name = model_input.name
    _, height, width, channels = model_input.shape
    print(f"\nModel input name: {input_name}")
    print(f"Expected input shape: {model_input.shape} (N, {height}, {width}, {channels})")

    calibration_imgs = np.zeros((num_samples, height, width, channels), dtype=np.float32)
    for cnt, (image, _) in enumerate(calibration_dataloader):
        if cnt >= num_samples:
            break
        if preprocess_fn:
            image = preprocess_fn(image, height, width)
        if isinstance(image, Image):
            calibration_imgs[cnt] = image.get_data()
        elif isinstance(image, np.ndarray):
            calibration_imgs[cnt] = image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}. Expected PIL Image or numpy array.")

    print(f"Shape calibration data: {calibration_imgs.shape}")
    if 0 in calibration_imgs.shape:
        raise ValueError(f"Invalid dimension found in calibration_data: {calibration_imgs.shape}")

    input_data = {input_name: calibration_imgs}
    print("\nTesting base inference...")
    base_preds = session.run(None, input_data)[0]
    print(f"Base predictions shape: {base_preds.shape}")

    results = {}

    total_nodes = len([node for node in model.graph.node])
    print(f"\nAnalyzing {total_nodes} layers...")

    for i, node in enumerate(model.graph.node, 1):
        if node.name in exclude_nodes or node.op_type in skip_ops:
            print(f"Skipping node: {node.name} ({node.op_type})")
            continue

        # Check if node has any initializer among its inputs
        has_initializer = any(init.name in node.input for init in model.graph.initializer)
        if not has_initializer:
            print(f"Skipping node: {node.name} ({node.op_type}) (no initializer)")
            continue

        print(f"Analyzing layer {i}/{total_nodes}: {node.name} ({node.op_type})")
        test_model = deepcopy(model)

        # Simulate INT8 quantization for the layer's initializers
        for init in test_model.graph.initializer:
            if init.name in node.input:
                data = np.frombuffer(init.raw_data, dtype=np.float32)
                if data.size == 0:
                    print(f"Warning: initializer {init.name} is empty, skipping quantization.")
                    continue
                qmin, qmax = -127, 127
                scale = max(np.max(np.abs(data)) / qmax, 1e-8)
                quant_data = np.clip(np.round(data / scale), qmin, qmax) * scale
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

            results[node.name] = {
                'mse': float(mse),
                'max_difference': float(max_diff),
                'cosine_similarity': float(cosine_sim),
                'op_type': str(node.op_type),
                'safe_to_quantize': bool(
                    mse < mse_threshold and 
                    max_diff < max_diff_threshold and 
                    cosine_sim > cosine_sim_threshold
                )
            }
        except Exception as e:
            print(f"Error analyzing node {node.name}: {str(e)}")
            continue

    return results