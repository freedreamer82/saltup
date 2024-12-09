import onnx
import os
import cv2
import random
import shutil
import numpy as np
import onnxruntime

from onnxruntime.quantization import quantize_dynamic, quantize_static, CalibrationDataReader, QuantType, QuantFormat


def quantize_onnx_model_dynamic(input_model_path:str, output_model_path:str, per_channel:bool=False):
    """quantize an onnx model

    Args:
        input_model_path (str): path of the input model
        output_model_path (str): path of the output model
    """
    # Load the ONNX model
    #model = onnx.load(input_model_path)
    
    # Verify the model
    #onnx.checker.check_model(model)
    
    # Perform dynamic quantization
    quantize_dynamic(
    model_input=input_model_path,
    model_output=output_model_path,
    per_channel=per_channel,  # Set to True if you want per-channel quantization (usually for Conv layers)
    weight_type=QuantType.QUInt8  # Quantize weights to uint8
    )
    
    print(f"Quantized model saved to: {output_model_path}")
    
    # Optional: Compare model sizes
    original_size = os.path.getsize(input_model_path) / (1024 * 1024)
    quantized_size = os.path.getsize(output_model_path) / (1024 * 1024)
    
    print(f"Original model size: {original_size:.2f} MB")
    print(f"Quantized model size: {quantized_size:.2f} MB")
    print(f"Size reduction: {(1 - quantized_size/original_size) * 100:.2f}%")
    


class yolo_calibration_data_reader(CalibrationDataReader):
    def __init__(self, calibration_image_folder: str, model_path: str, preprocess_function):
        """
        Calibration data reader for model. This class reads calibration images and prepares them
        for quantization.
        
        Args:
            calibration_image_folder (str): Path to the folder containing calibration images.
            model_path (str): Path to the model (ONNX format).
            preprocess_function (callable): User-defined function for preprocessing images.
        """
        self.enum_data = None
        self.preprocess_function = preprocess_function

        # Use inference session to get input shape.
        session = onnxruntime.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
        (_, height, width, _) = session.get_inputs()[0].shape
        print(f"Input shape: {session.get_inputs()[0].shape}")
        
        # Convert images to input data using the user-defined preprocessing function
        self.nhwc_data_list = self._load_and_preprocess_images(
            calibration_image_folder, height, width
        )
        
        self.input_name = session.get_inputs()[0].name
        self.datasize = len(self.nhwc_data_list)

    def _load_and_preprocess_images(self, images_folder, height, width, size_limit=0):
        """
        Loads a batch of images and preprocesses them using the provided preprocessing function.
        
        Args:
            images_folder (str): Path to the folder storing images.
            height (int): Image height in pixels.
            width (int): Image width in pixels.
            size_limit (int, optional): Number of images to load. Default is 0 which means all images are loaded.
        
        Returns:
            np.ndarray: A numpy array containing preprocessed images.
        """
        image_names = [elt for elt in os.listdir(images_folder) if elt.endswith('.jpg')]
        if size_limit > 0 and len(image_names) >= size_limit:
            batch_filenames = [image_names[i] for i in range(size_limit)]
        else:
            batch_filenames = image_names
        unconcatenated_batch_data = []

        for image_name in batch_filenames:
            image_filepath = os.path.join(images_folder, image_name)
            # Preprocess the image using the provided preprocessing function
            preprocessed_image = self.preprocess_function(image_filepath, height, width)
            unconcatenated_batch_data.append(preprocessed_image)

        batch_data = np.concatenate(
            np.expand_dims(unconcatenated_batch_data, axis=0), axis=0
        )
        return batch_data

    def get_next(self):
        """Yield next calibration image."""
        if self.enum_data is None:
            self.enum_data = iter(
                [{self.input_name: nhwc_data} for nhwc_data in self.nhwc_data_list]
            )
        return next(self.enum_data, None)

    def rewind(self):
        """Reset the iterator."""
        self.enum_data = None


def quantize_yolo_model_to_onnx(
    model_path:str,
    output_path:str,
    calibration_data_path:str,
    preprocess_function:str,
    preprocess_kwargs:dict=None,
    quant_format=QuantFormat.QOperator,
    activation_type=QuantType.QUInt8,
    weight_type=QuantType.QInt8,
    nodes_to_exclude:list[str]=None,
    per_channel:bool=False,
    reduce_range:bool=True,
) -> None:
    """
    Quantizes a model in ONNX format with static quantization.

    Parameters:
        model_path (str): Path to the input and output ONNX model (the same path for both).
        output_path (str): Path to save the quantized ONNX model.
        calibration_data_path (str): Path to the directory containing calibration images.
        preprocess_function (callable): Function to preprocess calibration images.
        quant_format (QuantFormat): Quantization format (default: QOperator).
        activation_type (QuantType): Activation quantization type (default: QUInt8).
        weight_type (QuantType): Weight quantization type (default: QInt8).
        nodes_to_exclude (list, optional): List of nodes to exclude from quantization.
        per_channel (bool): Whether to use per-channel quantization (default: False).
        reduce_range (bool): Whether to reduce the range for quantization (default: True).

    """


    # Create the calibration data reader instance
    dr = yolo_calibration_data_reader(calibration_data_path, model_path, preprocess_function, preprocess_kwargs)
    
    # Perform static quantization
    quantize_static(
        input_path=model_path,
        output_path=output_path,
        calibration_data_reader=dr,
        quant_format=quant_format,
        activation_type=activation_type,
        weight_type=weight_type,
        nodes_to_exclude=nodes_to_exclude,
        per_channel=per_channel,
        reduce_range=reduce_range
    )
    
    print(f"Quantized model saved to: {output_path}")

    # Optional: Compare model sizes
    original_size = os.path.getsize(model_path) / (1024 * 1024)
    quantized_size = os.path.getsize(output_path) / (1024 * 1024)

    print(f"Original model size: {original_size:.2f} MB")
    print(f"Quantized model size: {quantized_size:.2f} MB")
    print(f"Size reduction: {(1 - quantized_size / original_size) * 100:.2f}%")