import os
import numpy as np
import tensorflow as tf
import keras
from saltup.ai.utils.keras.quantization import quantize


def test_tflite_quantization(tmp_path):
    # Create a mock Keras model
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Save the mock model
    golden_model_path = str(tmp_path / "mock_model.keras")
    model.save(golden_model_path)

    # Define output path for the TFLite model
    output_tflite_path = str(tmp_path / "mock_model_quantized.tflite")

    def representative_data_gen():
        for _ in range(100):
            yield [np.random.rand(1, 28, 28).astype("float32")]

    # Call the quantize function
    quantized_model_path = quantize(
        model_path=golden_model_path,
        output_quantize_path=output_tflite_path,
        representative_data_gen_fnct=representative_data_gen,
        input_type=tf.uint8,
        output_type=tf.uint8
    )

    # Assertions
    assert os.path.exists(quantized_model_path)
    assert quantized_model_path.endswith(".tflite")

    # Verify the input and output types of the quantized model
    interpreter = tf.lite.Interpreter(model_path=quantized_model_path)
    input_type = interpreter.get_input_details()[0]['dtype']
    output_type = interpreter.get_output_details()[0]['dtype']
    assert input_type == np.uint8, "Input type is not quantized to uint8"
    assert output_type == np.uint8, "Output type is not quantized to uint8"