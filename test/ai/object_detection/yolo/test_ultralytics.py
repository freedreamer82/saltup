import pytest
import numpy as np
import os
from typing import Iterator
os.environ["SALTUP_BACKEND"] = "keras_tensorflow"
import onnx
from onnx import helper, TensorProto
import tempfile
from saltup.ai.object_detection.utils.bbox import BBox
from saltup.utils.data.image.image_utils import Image
from saltup.ai.object_detection.yolo.impl.yolo_ultralytics import YoloUltralytics
from saltup.ai.nn_model import NeuralNetworkModel

# Function to create a simple ONNX model (mimicking YOLO output)
def create_simple_yolo_onnx_model(output_path: str) -> None:
    input_shape = [1, 3, 256, 320]
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
def yolo_ultralytics_with_temp_model() -> Iterator[YoloUltralytics]:
    # Create a temporary file for the ONNX model
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as temp_file:
        temp_file_path = temp_file.name

    try:
        # Generate a simple ONNX model and save it to the temporary file
        create_simple_yolo_onnx_model(temp_file_path)

        # Initialize YoloAnchorsBased with the temporary model
        number_class = 4  # Replace with the actual number of classes
        yolo_instance = YoloUltralytics(NeuralNetworkModel(temp_file_path), number_class)

        yield yolo_instance

    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# Test preprocess method
def test_preprocess(yolo_ultralytics_with_temp_model: YoloUltralytics) -> None:
    # Generate a random image tensor
    image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    image = Image(image)
    # Define target dimensions
    target_height, target_width = 256, 320
    
    # Preprocess the image
    processed_image = yolo_ultralytics_with_temp_model.preprocess(image, target_height, target_width)
    
    # Check the shape of the processed image
    assert processed_image.shape == (1, 3, target_height, target_width)
    
    # Check the type of the processed image
    assert processed_image.dtype == np.float32

# Test postprocess method
def test_postprocess(yolo_ultralytics_with_temp_model: YoloUltralytics) -> None:

    # Generate random values for the format (8, 100)
    # xcenter, ycenter, width, height (all between 0 and 640 pixels)
    bbox_values = np.random.uniform(0, 321, (4, 100))  # 4 rows for bbox values
    
    # Scores for 4 classes (values between 0 and 1)
    class_scores = np.random.rand(4, 100)  # 4 rows for class scores

    # Concatenate to form the final tensor of shape (8, 100)
    raw_output = [np.concatenate((bbox_values, class_scores), axis=0)]
    
    # Define image dimensions
    image_height, image_width = 480, 640
    
    # Postprocess the raw output
    result = yolo_ultralytics_with_temp_model.postprocess(raw_output, image_height, image_width)
    
    # Check the type of the result
    assert isinstance(result, list)
    
    # Check the elements of the result
    for box_object, class_id, score in result:
        
        assert isinstance(box_object, BBox)
        assert isinstance(class_id, int)
        assert 0 <= score <= 1

def test_postprocess_class_agnostic_nms(yolo_ultralytics_with_temp_model: YoloUltralytics) -> None:
    # Two identical boxes: class 0 scores 0.9, class 1 scores 0.8; every other column is empty
    raw = np.zeros((8, 100))
    raw[:4, 0] = raw[:4, 1] = [160, 128, 50, 50]
    raw[4, 0] = 0.9
    raw[5, 1] = 0.8

    yolo = yolo_ultralytics_with_temp_model
    assert yolo.class_agnostic_nms is False
    per_class = yolo.postprocess([raw], 480, 640)
    assert sorted(c for _, c, _ in per_class) == [0, 1]

    yolo.class_agnostic_nms = True
    agnostic = yolo.postprocess([raw], 480, 640)
    assert [c for _, c, _ in agnostic] == [0]


# Run the tests
if __name__ == "__main__":
    pytest.main()