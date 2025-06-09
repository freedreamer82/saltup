import tensorflow as tf
import os
import keras
import numpy as np
from typing import Callable, Iterator, Any


def quantize(
    model_path: str,
    output_quantize_path: str,
    representative_data_gen_fnct: Callable[[], Iterator[Any]],
    input_type: tf.dtypes.DType = tf.uint8,
    output_type: tf.dtypes.DType = tf.float32
) -> str:
    """
    Quantizes a Keras model to a quantized TFLite model using representative data.

    Args:
        model_path (str): Path to the Keras model file (.h5 or SavedModel directory).
        output_quantize_path (str): Path where the quantized TFLite model will be saved.
        representative_data_gen_fnct (Callable[[], Iterator[Any]]): Function yielding representative input samples for calibration.
        input_type (tf.dtypes.DType or None, optional): Input tensor type for inference. If None, input quantization is not applied. Defaults to tf.uint8.
        output_type (tf.dtypes.DType or None, optional): Output tensor type for inference. If None, output quantization is not applied. Defaults to tf.float32.

    Returns:
        str: Path to the quantized TFLite model.
    """
    # Check if model_path is a valid Keras model file (.h5, .keras) or directory (SavedModel)
    valid_extensions = [".h5", ".keras"]
    if os.path.isfile(model_path):
        _, ext = os.path.splitext(model_path)
        if ext.lower() not in valid_extensions:
            raise ValueError(f"File '{model_path}' does not have a valid Keras model extension ({valid_extensions}).")
    elif os.path.isdir(model_path):
        # Check for SavedModel signature file
        if not os.path.exists(os.path.join(model_path, "saved_model.pb")):
            raise ValueError(f"Directory '{model_path}' does not contain a SavedModel (missing 'saved_model.pb').")
    else:
        raise ValueError(f"Provided path '{model_path}' is not a valid Keras model file or directory.")

    # Load Keras model
    try:
        if os.path.isdir(model_path):
            model = keras.models.load_model(model_path, compile=False)
        else:
            model = keras.models.load_model(model_path, compile=False)
    except Exception as e:
        raise ValueError(f"File at {model_path} is not a valid Keras model: {e}")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
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