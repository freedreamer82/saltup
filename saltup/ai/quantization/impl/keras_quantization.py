import os
import tensorflow as tf
import numpy as np
import cv2


def quantize_model_to_tflite(
    keras_model_path:str,
    tflite_model_path:str,
    calibration_data_path:str,
    preprocess_function:str,
    custom_objects:dict[str]=None,
    quantize_input:bool=True,
    quantize_output:bool=True,
    layers_to_skip:list[str]=None,
    **preprocess_kwargs
) -> None:
    """
    Quantizes a model to a TensorFlow Lite model with full integer quantization.

    Parameters:
        keras_model_path (str): Path to the saved Keras model (saved in .h5 or SavedModel format).
        tflite_model_path (str): Path to save the quantized TensorFlow Lite model.
        calibration_data_path (str): Path to the directory containing calibration images.
        preprocess_function (callable): Function to preprocess the raw data to the model's input format.
        custom_objects (dict, optional): Dictionary of custom objects for loading the Keras model.
        quantize_input (bool): If True, quantizes the input tensors to `uint8` (or `int8`).
        quantize_output (bool): If True, quantizes the output tensors to `uint8` (or `int8`).
        layers_to_skip (list, optional): List of layer names to exclude from quantization.
        preprocess_kwargs (dict): Additional arguments to pass to the preprocess function.

    Usage:
        def preprocess(image, target_size=(416, 416), normalize=True):
            # Example: Resize and normalize
            return cv2.resize(image, target_size) / 255.0

        quantize_yolo_model_to_tflite(
            keras_model_path="model.keras",
            tflite_model_path="quantized_model.tflite",
            calibration_data_path="path/to/calibration/images",
            preprocess_function=preprocess,
            custom_objects={"CustomLayer": CustomLayerClass},
            quantize_input=True,
            quantize_output=False,
            layers_to_skip=["output_layer"],
            target_size=(416, 416),
            normalize=True
        )
    """

    def representative_dataset():
        """
        Generator function that yields preprocessed representative data samples.
        """
        for image_name in os.listdir(calibration_data_path):
            image_path = os.path.join(calibration_data_path, image_name)
            
            # Preprocess the image using the user-defined function
            preprocessed_image = preprocess_function(image_path, **preprocess_kwargs)
            
            yield [preprocessed_image.astype(np.float32)]

    # Load the Keras model, with custom objects if specified
    model = tf.keras.models.load_model(keras_model_path, custom_objects=custom_objects)

    # If skipping layers, set those layers as non-trainable (to exclude them from quantization)
    if layers_to_skip:
        for layer in model.layers:
            if layer.name in layers_to_skip:
                layer.trainable = False

    # Convert the model to TensorFlow Lite format with quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset

    # Configure input and output quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    if quantize_input:
        converter.inference_input_type = tf.uint8  # or tf.int8
    if quantize_output:
        converter.inference_output_type = tf.uint8  # or tf.int8

    # Convert and save the quantized model
    tflite_quantized_model = converter.convert()
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_quantized_model)
    
    print(f"Quantized model saved at: {tflite_model_path}")


