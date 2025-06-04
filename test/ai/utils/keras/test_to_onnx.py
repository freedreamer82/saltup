import pytest
import numpy as np
import os
import tensorflow as tf
from saltup.ai.utils.keras.to_onnx import (
    convert_keras_to_onnx, verify_onnx_model
)


def test_convert_keras_to_onnx_and_verify(tmp_path):
    # Create a mock Keras model using the functional API
    inputs = tf.keras.Input(shape=(28, 28))
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Save the mock Keras model
    keras_model_path = str(tmp_path / "mock_model.keras")
    model.save(keras_model_path)

    # Define the ONNX output path
    onnx_model_path = str(tmp_path / "mock_model.onnx")

    # Convert the Keras model to ONNX
    onnx_model, keras_model = convert_keras_to_onnx(keras_model_path, onnx_model_path)

    # Assertions for ONNX model file
    assert os.path.exists(onnx_model_path)
    assert onnx_model_path.endswith(".onnx")

    # Verify the ONNX model against the Keras model
    keras_pred, onnx_pred = verify_onnx_model(onnx_model_path, keras_model)

    # Assertions for predictions
    assert keras_pred.shape == onnx_pred.shape
    np.testing.assert_allclose(keras_pred, onnx_pred, rtol=1e-5, atol=1e-5)