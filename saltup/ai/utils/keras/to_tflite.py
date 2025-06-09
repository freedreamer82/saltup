import tensorflow as tf
import os
import keras
from typing import Any


def tflite_conversion(
    model_path: str,
    output_tflite_path: str,
) -> str:
    """
    Converts a Keras model to TFLite format.

    Args:
        model_path (str): Path to the Keras model file (.h5 or SavedModel).
        output_tflite_path (str): Path where the converted TFLite model will be saved.

    Returns:
        str: Path to the saved TFLite model.
    """
    model = keras.models.load_model(model_path, compile=False)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    tflite_model_quantInt = converter.convert()

    # Check input and output types
    interpreter = tf.lite.Interpreter(model_content=tflite_model_quantInt)
    input_type = interpreter.get_input_details()[0]['dtype']
    print('input: ', input_type)
    output_type = interpreter.get_output_details()[0]['dtype']
    print('output: ', output_type)

    # Save model and weights
    os.makedirs(os.path.dirname(output_tflite_path), exist_ok=True)
    with open(output_tflite_path, 'wb') as f:
        f.write(tflite_model_quantInt)
    print('Saved trained model at %s ' % output_tflite_path)
    return output_tflite_path
