import pytest
from typing import List, Tuple, Optional
import numpy as np
from saltup.ai.object_detection.yolo.yolo import YoloOutput
from saltup.ai.object_detection.utils.bbox  import BBoxFormat, BBox, BBoxClassIdScore
from saltup.utils.data.image.image_utils import Image


def test_yolo_output():
    # Test constructor
    boxes = [(BBox(coordinates=(10, 20, 30, 40), fmt=BBoxFormat.CORNERS_ABSOLUTE, img_height=100, img_width=100), 1, 0.9)]
    image = Image(np.zeros((100, 100, 3)))
    yolo_output = YoloOutput(boxes, image)
    
    # Test get_boxes
    assert yolo_output.get_boxes() == boxes
    
    # Test set_boxes
    new_boxes = [(BBox(coordinates=(50, 60, 70, 80), fmt=BBoxFormat.CORNERS_ABSOLUTE, img_height=100, img_width=100), 2, 0.8)]
    yolo_output.set_boxes(new_boxes)
    assert yolo_output.get_boxes() == new_boxes
    
    # Test get_image
    assert np.array_equal(yolo_output.get_image(), image)
    
    # Test set_image
    new_image = Image(np.ones((50, 50, 3)))
    yolo_output.set_image(new_image)
    assert np.array_equal(yolo_output.get_image(), new_image)
    
    # Test inference time
    yolo_output.set_inference_time(50.5)
    assert yolo_output.get_inference_time() == 50.5
    
    # Test preprocessing time
    yolo_output.set_preprocessing_time(20.2)
    assert yolo_output.get_preprocessing_time() == 20.2
    
    # Test postprocessing time
    yolo_output.set_postprocessing_time(30.3)
    assert yolo_output.get_postprocessing_time() == 30.3
    
    # Test total processing time
    assert yolo_output.get_total_processing_time() == 50.5 + 20.2 + 30.3
    
    # Test get_property
    assert yolo_output.get_property("boxes") == new_boxes
    assert yolo_output.get_property("image") == new_image
    assert yolo_output.get_property("inference_time") == 50.5
    assert yolo_output.get_property("preprocessing_time") == 20.2
    assert yolo_output.get_property("postprocessing_time") == 30.3
    with pytest.raises(AttributeError):
        yolo_output.get_property("invalid_property")
    
    # Test set_property
    yolo_output.set_property("inference_time", 99.9)
    assert yolo_output.get_inference_time() == 99.9
    
    yolo_output.set_property("preprocessing_time", 10.1)
    assert yolo_output.get_preprocessing_time() == 10.1
    
    with pytest.raises(AttributeError):
        yolo_output.set_property("invalid_property", 123)
    
    # Test __repr__
    assert "YoloOutput" in repr(yolo_output)
