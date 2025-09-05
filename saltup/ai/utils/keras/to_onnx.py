import tensorflow as tf
#from tf_keras.saving import load_model
#from tensorflow.keras.models import load_model
import keras
import tf2onnx
import onnx
import numpy as np
import sys
import onnxruntime as ort


# Monkey-patch for non-standard tf_keras imports
# Some saved models reference 'tf_keras.src.models.functional'.
# We alias it to the public TensorFlow Keras functional module.
try:
    import tensorflow.keras.models.functional as _functional_module
    sys.modules['tf_keras.src.models.functional'] = _functional_module
except ImportError:
    # If aliasing fails, proceed; load_model may still work with custom_objects
    pass

def convert_keras_to_onnx(
    keras_model_path: str,
    onnx_path: str,
    opset: int = 16
) -> tuple[onnx.ModelProto, keras.Model]:
    """
    Converts a Keras model (.keras) to ONNX format

    Args:
        keras_model_path: Path to the .keras model to convert
        onnx_path: Path where to save the ONNX model
        opset: ONNX opset version to use (default: 16)
    Returns:
        Tuple of (ONNX model proto, loaded Keras model)
    """
    print(f"Converting Keras model '{keras_model_path}' to ONNX format at '{onnx_path}' with opset {opset}...")
    # 1. Load Keras model
    # Don't load compiler to avoid issues with custom loss
    model: keras.Model = keras.models.load_model(
        keras_model_path,
        compile=False  
    )

    # 2. Get input shape
    input_shape = model.input_shape[1:]  # Remove batch size

    # 3. Convert to ONNX
    spec = (tf.TensorSpec(shape=(None,) + input_shape, dtype=tf.float32, name="input"),)

    # Model conversion
    model_proto, _ = tf2onnx.convert.from_keras(
        model, 
        input_signature=spec,
        opset=opset,  # Use recent opset version
        target=["onnxruntime"],
        custom_ops=None,
        custom_op_handlers=None,
        custom_rewriter=None,
        inputs_as_nchw=None,
        extra_opset=None,
        shape_override=None
    )

    # 4. Save ONNX model
    onnx.save(model_proto, onnx_path)
    # Return both ONNX and Keras models for testing
    return model_proto, model

def verify_onnx_model(
    onnx_path: str,
    keras_model_path: str,
    test_input: np.ndarray = None,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    verbose: bool = True
) -> tuple[dict, np.ndarray, np.ndarray]:
    """
    Verify that the ONNX model produces outputs close to the original Keras model.

    Args:
        onnx_path: Path to the ONNX model.
        keras_model_path: Path to the original Keras model for comparison.
        test_input: Optional test input (numpy array). If None, a random input is generated.
        rtol: Relative tolerance for comparison.
        atol: Absolute tolerance for comparison.
        verbose: If True, prints verification status and stats.

    Returns:
        Tuple of (stats_dict, keras_pred, onnx_pred)
    """
    # Load Keras model
    keras_model: keras.Model = keras.models.load_model(keras_model_path, compile=False)

    # Create ONNX session
    providers = ort.get_available_providers()
    session = ort.InferenceSession(onnx_path, providers=providers)

    # Prepare test input
    if test_input is None:
        input_shape = keras_model.input_shape[1:]
        test_input = np.random.randn(1, *input_shape).astype(np.float32)

    # Get input name for ONNX
    input_name = session.get_inputs()[0].name

    # Run inference
    keras_pred: np.ndarray = keras_model.predict(test_input)
    onnx_pred: np.ndarray = session.run(None, {input_name: test_input.astype(np.float32)})[0]

    # Calculate differences
    abs_diff = np.abs(keras_pred - onnx_pred)
    rel_diff = abs_diff / (np.abs(keras_pred) + 1e-8)
    max_abs_diff = np.max(abs_diff)
    max_rel_diff = np.max(rel_diff)

    # Check if within tolerance
    within_tolerance = np.allclose(keras_pred, onnx_pred, rtol=rtol, atol=atol)

    if verbose:
        if within_tolerance:
            print("ONNX model verification PASSED: outputs are close within tolerance.")
        else:
            print("ONNX model verification FAILED: outputs differ beyond tolerance.")
            print("Max absolute difference:", max_abs_diff)
            print("Max relative difference:", max_rel_diff)

    # Calculate percentage of elements outside tolerance
    not_close = ~np.isclose(keras_pred, onnx_pred, rtol=rtol, atol=atol)
    total_elements = np.prod(keras_pred.shape)
    num_not_close = np.sum(not_close)
    percent_error = (num_not_close / total_elements) * 100

    stats = {
        "within_tolerance": within_tolerance,
        "max_abs_diff": max_abs_diff,
        "max_rel_diff": max_rel_diff,
        "percent_error": percent_error
    }

    return stats, keras_pred, onnx_pred
