import os
import numpy as np
import tensorflow as tf
from saltup.ai.keras_utils.keras_to_tflite_quantization import tflite_quantization

def test_tflite_quantization(tmp_path):
    # Create a mock Keras model
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Save the mock model
    golden_model_path = str(tmp_path / "mock_model.keras")
    model.save(golden_model_path)

    # Generate mock training data
    x_train = np.random.rand(1000, 28, 28).astype("float32")

    # Define output path for the TFLite model
    output_tflite_path = str(tmp_path / "mock_model_quantized.tflite")

    # Call the tflite_quantization function
    quantized_model_path = tflite_quantization(
        golden_model_path=golden_model_path,
        output_tflite_path=output_tflite_path,
        x_train=x_train,
        quantizing_input=True,
        quantizing_output=True
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