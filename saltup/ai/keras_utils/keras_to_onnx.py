import tensorflow as tf
#from tf_keras.saving import load_model
from tensorflow.keras.models import load_model
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

def convert_keras_to_onnx(keras_model_path, onnx_path,opset = 16):
    """
    Converts a Keras model (.keras) to ONNX format
    
    Args:
        keras_model_path: Path to the .keras model to convert
        onnx_path: Path where to save the ONNX model
    """
    # 1. Load Keras model
    # Don't load compiler to avoid issues with custom loss
    model = load_model(
        keras_model_path,
        compile=False  
    )
    
    # 2. Get input shape
    input_shape = model.input_shape[1:]  # Remove batch size
    
    # 3. Convert to ONNX
    spec = (tf.TensorSpec(shape=(1,) + input_shape, dtype=tf.float32, name="input"),)
    
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

def verify_onnx_model(onnx_path, keras_model, test_input=None):
    """
    Verify that the ONNX model works correctly
    
    Args:
        onnx_path: Path to the ONNX model
        keras_model: Original Keras model for comparison
        test_input: Optional test input (numpy array)
    """
    
    # Create ONNX session
    session = ort.InferenceSession(onnx_path)
    
    # If no test input provided, create one
    if test_input is None:
        input_shape = keras_model.input_shape[1:]
        test_input = np.random.randn(1, *input_shape).astype(np.float32)
    
    # Get input name
    input_name = session.get_inputs()[0].name
    
    # Run inference with both models
    keras_pred = keras_model.predict(test_input)
    onnx_pred = session.run(None, {input_name: test_input.astype(np.float32)})[0]
    
    # Verify predictions are similar
    np.testing.assert_allclose(keras_pred, onnx_pred, rtol=1e-5, atol=1e-5)
    print("Verification test completed successfully!")
    
    return keras_pred, onnx_pred