import pytest
import numpy as np
import os
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto
import tempfile
from saltup.ai.object_detection.yolo.yolo import YoloType
from saltup.ai.object_detection.utils.bbox import BBox, BBoxFormat
from saltup.utils.data.image.image_utils import Image
from saltup.ai.object_detection.yolo.impl.yolo_damo import YoloDamo
from saltup.ai.nn_model import NeuralNetworkModel

# Function to create a simple ONNX model (mimicking YOLO output)
def create_simple_yolo_onnx_model(output_path):
    input_shape = [1, 3, 320, 320]
    output_shape = [1, 8, 100]

    input1 = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)

    # Define a dummy graph (identity operation for simplicity)
    identity_node = helper.make_node('Identity', inputs=['input'], outputs=['output'])

    # Create the graph
    graph = helper.make_graph(
        [identity_node],
        'simple_yolo_graph',
        [input1],
        [output]
    )

    # Create the model with opset version 21
    model = helper.make_model(graph, producer_name='onnx-yolo-example', opset_imports=[helper.make_opsetid("", 16)])

    # Save the model to the output path
    onnx.save(model, output_path)

# Fixture for creating a YoloAnchorsBased instance with a temporary ONNX model
@pytest.fixture
def yolo_damo_with_temp_model():
    # Create a temporary file for the ONNX model
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as temp_file:
        temp_file_path = temp_file.name

    try:
        # Generate a simple ONNX model and save it to the temporary file
        create_simple_yolo_onnx_model(temp_file_path)

        # Initialize YoloAnchorsBased with the temporary model
        yolot = YoloType.DAMO  # Replace with the appropriate YoloType
        
        number_class = 4  # Replace with the actual number of classes
        yolo_instance = YoloDamo(yolot, NeuralNetworkModel(temp_file_path), number_class)

        yield yolo_instance

    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# Test preprocess method
def test_preprocess(yolo_damo_with_temp_model):
    # Generate a random image tensor
    raw_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    image = Image(raw_image)
    # Define target dimensions
    target_height, target_width = 320, 320
    
    # Preprocess the image
    processed_image = yolo_damo_with_temp_model.preprocess(image, target_height, target_width)
    
    # Check the shape of the processed image
    assert processed_image.shape == (1, 3, target_height, target_width)
    
    # Check the type of the processed image
    assert processed_image.dtype == np.float32

# Test postprocess method
def test_postprocess(yolo_damo_with_temp_model):

    # Generate the first tensor with scores between 0 and 1
    scores = np.random.rand(1, 100, 4)  # Random values between 0 and 1

    # Generate the second tensor with bounding box values between 0 and max pixel coordonate value
    bboxes = np.random.randint(0, 20, (1, 100, 4))  # Random integers between 0 and max pixel coordonate value
    
    # Ensure the last two values are greater than the first two
    bboxes[..., 2] = bboxes[..., 0] + np.random.randint(1, 10, bboxes[..., 0].shape)  # width
    bboxes[..., 3] = bboxes[..., 1] + np.random.randint(1, 10, bboxes[..., 1].shape)  # height
    
    # Combine into a list
    raw_output = [scores, bboxes]
    
    # Define image dimensions
    image_height, image_width = 480, 640
    
    # Postprocess the raw output
    result = yolo_damo_with_temp_model.postprocess(raw_output, image_height, image_width)
    
    # Check the type of the result
    assert isinstance(result, list)
    
    # Check the elements of the result
    for box_object, class_id, score in result:
        
        assert isinstance(box_object, BBox)
        assert isinstance(class_id, int)
        assert 0 <= score <= 1

# Run the tests
if __name__ == "__main__":
    pytest.main()