import os
import numpy as np
import tensorflow as tf
from typing import Callable, Iterator, Any


def quantize(
    model_path: str,
    output_quantize_path: str,
    representative_data_gen_fnct: Callable[[], Iterator[Any]],
    input_type: tf.dtypes.DType = tf.uint8,
    output_type: tf.dtypes.DType = tf.float32
) -> str:
    """
    Quantizes a float32 TensorFlow model to a quantized TFLite model using representative data.

    Args:
        model_path (str): Path to the SavedModel directory or .tflite file.
        output_quantize_path (str): Path where the quantized TFLite model will be saved.
        representative_data_gen_fnct (Callable[[], Iterator[Any]]): Function yielding representative input samples for calibration.
        input_type (tf.dtypes.DType, optional): Input tensor type for inference. Defaults to tf.uint8.
        output_type (tf.dtypes.DType, optional): Output tensor type for inference. Defaults to tf.float32.

    Returns:
        str: Path to the quantized TFLite model.
    """
    # Check if tflite_path is a valid SavedModel directory or .tflite file
    if not (os.path.isdir(model_path) or (os.path.isfile(model_path) and model_path.endswith('.tflite'))):
        raise ValueError(f"Provided path '{model_path}' is not a valid SavedModel directory or .tflite file.")

    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen_fnct
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    if input_type is not None:
        converter.inference_input_type = input_type
    if output_type is not None:
        converter.inference_output_type = output_type

    tflite_quant_model = converter.convert()

    os.makedirs(os.path.dirname(output_quantize_path), exist_ok=True)
    with open(output_quantize_path, "wb") as f:
        f.write(tflite_quant_model)
    print(f"Quantized model saved at {output_quantize_path}")
    return output_quantize_path