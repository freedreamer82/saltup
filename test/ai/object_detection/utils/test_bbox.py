import pytest
import numpy as np
from saltup.ai.object_detection.utils.bbox import BBox, BBoxFormat, NotationFormat, IoUType
from saltup.ai.object_detection.utils.bbox import nms, convert_matrix_boxes
# Test data
TEST_IMAGE_WIDTH = 640
TEST_IMAGE_HEIGHT = 480


# Test cases for convert_matrix_boxes function
def test_convert_matrix_boxes():
    # Test case 1: Single bounding box
    box_xy = np.array([[50, 50]])  # Center coordinates (x_center, y_center)
    box_wh = np.array([[40, 60]])  # Width and height (width, height)

    expected_corners = np.array([[30, 20, 70, 80]])  # Expected corners (xmin, ymin, xmax, ymax)
    expected_centers = np.array([[50, 50, 40, 60]])  # Expected centers (x, y, w, h)

    corners, centers = convert_matrix_boxes(box_xy, box_wh)

    # Assert that the output matches the expected values
    assert np.array_equal(corners, expected_corners), "Corners do not match expected output"
    assert np.array_equal(centers, expected_centers), "Centers do not match expected output"

    # Test case 2: Multiple bounding boxes
    box_xy = np.array([[50, 50], [100, 100]])  # Center coordinates for two boxes
    box_wh = np.array([[40, 60], [80, 120]])   # Width and height for two boxes

    expected_corners = np.array([
        [30, 20, 70, 80],   # Box 1 corners
        [60, 40, 140, 160]  # Box 2 corners
    ])
    expected_centers = np.array([
        [50, 50, 40, 60],   # Box 1 centers
        [100, 100, 80, 120] # Box 2 centers
    ])

    corners, centers = convert_matrix_boxes(box_xy, box_wh)

    # Assert that the output matches the expected values
    assert np.array_equal(corners, expected_corners), "Corners do not match expected output for multiple boxes"
    assert np.array_equal(centers, expected_centers), "Centers do not match expected output for multiple boxes"

    # Test case 3: Edge case with zero width and height
    box_xy = np.array([[50, 50]])
    box_wh = np.array([[0, 0]])

    expected_corners = np.array([[50, 50, 50, 50]])  # All corners should be the center point
    expected_centers = np.array([[50, 50, 0, 0]])    # Width and height should be zero

    corners, centers = convert_matrix_boxes(box_xy, box_wh)

    # Assert that the output matches the expected values
    assert np.array_equal(corners, expected_corners), "Corners do not match expected output for zero width/height"
    assert np.array_equal(centers, expected_centers), "Centers do not match expected output for zero width/height"

    # Test case 4: Negative width and height (should still work)
    box_xy = np.array([[50, 50]])
    box_wh = np.array([[-40, -60]])

    expected_corners = np.array([[70, 80, 30, 20]])  # Corners should flip due to negative width/height
    expected_centers = np.array([[50, 50, -40, -60]])

    corners, centers = convert_matrix_boxes(box_xy, box_wh)

    # Assert that the output matches the expected values
    assert np.array_equal(corners, expected_corners), "Corners do not match expected output for negative width/height"
    assert np.array_equal(centers, expected_centers), "Centers do not match expected output for negative width/height"
    

# Test cases for BBox class
def test_bbox_initialization():
    # Test initialization with CORNERS format
    bbox = BBox(TEST_IMAGE_HEIGHT, TEST_IMAGE_WIDTH, [100, 150, 200, 250], BBoxFormat.CORNERS)
    assert bbox.get_coordinates() == pytest.approx([0.15625, 0.3125, 0.3125, 0.5208333333333334])
    assert bbox.get_img_width() == TEST_IMAGE_WIDTH
    assert bbox.get_img_height() == TEST_IMAGE_HEIGHT

    # Test initialization with CENTER format
    bbox = BBox(TEST_IMAGE_HEIGHT, TEST_IMAGE_WIDTH, [150, 200, 100, 100], BBoxFormat.CENTER)
    assert bbox.get_coordinates() == pytest.approx([0.15625, 0.3125, 0.3125, 0.5208333333333334])

    # Test initialization with TOPLEFT format
    bbox = BBox(TEST_IMAGE_HEIGHT, TEST_IMAGE_WIDTH, [100, 150, 100, 100], BBoxFormat.TOPLEFT)
    assert bbox.get_coordinates() == pytest.approx([0.15625, 0.3125, 0.3125, 0.5208333333333334])

def test_bbox_copy():
    bbox = BBox(TEST_IMAGE_HEIGHT, TEST_IMAGE_WIDTH, [100, 150, 200, 250])
    bbox_copy = bbox.copy()
    assert bbox_copy.get_coordinates() == bbox.get_coordinates()
    assert bbox_copy.get_img_width() == bbox.get_img_width()
    assert bbox_copy.get_img_height() == bbox.get_img_height()

def test_bbox_is_normalized():
    # Test normalized coordinates
    input_coordinates = [0.1, 0.2, 0.3, 0.4]
    assert BBox.is_normalized(input_coordinates, BBoxFormat.CORNERS) == True

    # Test non-normalized coordinates
    pixel_input_coordinates = [100, 150, 200, 250]
    
    assert BBox.is_normalized(pixel_input_coordinates, BBoxFormat.CORNERS) == False

def test_bbox_pascalvoc_to_yolo_bbox():
    voc_bbox = [100, 150, 200, 250]
    yolo_bbox = BBox.pascalvoc_to_yolo_bbox(voc_bbox, TEST_IMAGE_WIDTH, TEST_IMAGE_HEIGHT)
    expected_yolo_bbox = (0.234375, 0.4166666666666667, 0.15625, 0.20833333333333334)
    assert yolo_bbox == pytest.approx(expected_yolo_bbox)

def test_bbox_corners_to_center_format():
    corners = [100, 150, 200, 250]
    center = BBox.corners_to_center_format(corners)
    expected_center = (150.0, 200.0, 100.0, 100.0)
    assert center == expected_center

def test_bbox_center_to_corners_format():
    center = [150, 200, 100, 100]
    corners = BBox.center_to_corners_format(center)
    expected_corners = (100.0, 150.0, 200.0, 250.0)
    assert corners == expected_corners

def test_bbox_normalize():
    input_coordinates = [100, 150, 200, 250]
    img_width = TEST_IMAGE_WIDTH
    img_height = TEST_IMAGE_HEIGHT
    coordinates = BBox.normalize(input_coordinates, img_width, img_height, BBoxFormat.CORNERS)
    assert coordinates == pytest.approx([0.15625, 0.3125, 0.3125, 0.5208333333333334])

def test_bbox_absolute():
    input_coordinates = [0.1, 0.2, 0.3, 0.4]
    img_width = TEST_IMAGE_WIDTH
    img_height = TEST_IMAGE_HEIGHT
    absolute_coords = BBox.absolute(input_coordinates, img_width, img_height, BBoxFormat.CORNERS)
    expected_coords = (64, 96, 192, 192)
    assert absolute_coords == expected_coords

def test_bbox_to_yolo():
    bbox = BBox(TEST_IMAGE_HEIGHT, TEST_IMAGE_WIDTH, [100, 150, 200, 250])
    yolo_coords = bbox.to_yolo()
    expected_yolo_coords = (0.234375, 0.4166666666666667, 0.15625, 0.20833333333333334)
    assert yolo_coords == pytest.approx(expected_yolo_coords)

def test_bbox_to_coco():
    bbox = BBox(TEST_IMAGE_HEIGHT, TEST_IMAGE_WIDTH, [100, 150, 200, 250])
    coco_coords = bbox.to_coco()
    expected_coco_coords = (100, 150, 100, 100)
    assert coco_coords == expected_coco_coords

def test_bbox_to_pascal_voc():
    bbox = BBox(TEST_IMAGE_HEIGHT, TEST_IMAGE_WIDTH, [100, 150, 200, 250])
    pascal_coords = bbox.to_pascal_voc()
    expected_pascal_coords = (100, 150, 200, 250)
    assert pascal_coords == expected_pascal_coords

def test_bbox_compute_iou():
    bbox1 = BBox(TEST_IMAGE_HEIGHT, TEST_IMAGE_WIDTH, [100, 150, 200, 250])
    bbox2 = BBox(TEST_IMAGE_HEIGHT, TEST_IMAGE_WIDTH, [150, 200, 250, 300])
    iou = bbox1.compute_iou(bbox2)
    expected_iou = 0.14285714285714285
    assert iou == pytest.approx(expected_iou)

# Test cases for utility functions
def test_nms():
    bboxes = [
        BBox(TEST_IMAGE_HEIGHT, TEST_IMAGE_WIDTH, [100, 150, 200, 250]),
        BBox(TEST_IMAGE_HEIGHT, TEST_IMAGE_WIDTH, [150, 200, 250, 300]),
        BBox(TEST_IMAGE_HEIGHT, TEST_IMAGE_WIDTH, [50, 100, 150, 200])
    ]
    scores = [0.9, 0.8, 0.7]
    iou_threshold = 0.5
    selected_bboxes = nms(bboxes, scores, iou_threshold)
    assert isinstance(selected_bboxes, list)

# Add more test cases as needed

if __name__ == "__main__":
    pytest.main()