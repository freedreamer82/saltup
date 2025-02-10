import pytest
import json
import numpy as np
from typing import Tuple

from saltup.ai.object_detection.utils.bbox import BBox, BBoxFormat, NotationFormat, IoUType
from saltup.ai.object_detection.utils.bbox import nms, convert_matrix_boxes

# Test data
TEST_IMAGE_WIDTH = 640
TEST_IMAGE_HEIGHT = 480

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

def test_corners_to_center_format():
    corners = [100, 150, 200, 250]
    center = BBox.corners_to_center_format(corners)
    expected_center = (150.0, 200.0, 100.0, 100.0)
    assert center == expected_center

def test_corners_to_topleft_format():
    topleft_bbox = BBox.corners_to_topleft_format([100, 150, 300, 400])
    expected_topleft_bbox = (100, 150, 300 - 100, 400 - 150)
    assert topleft_bbox == expected_topleft_bbox

def test_center_to_corners_format():
    center = [150, 200, 100, 100]
    corners = BBox.center_to_corners_format(center)
    expected_corners = (100.0, 150.0, 200.0, 250.0)
    assert corners == expected_corners
    
def test_center_to_topleft_format():
    topleft_bbox = BBox.center_to_topleft_format([200, 275, 200, 250])
    expected_topleft_bbox = (200 - 100, 275 - 125, 200, 250)
    assert topleft_bbox == expected_topleft_bbox

def test_topleft_to_center_format():
    center_bbox = BBox.topleft_to_center_format([100, 150, 200, 250])
    expected_center_bbox = (100 + 100, 150 + 125, 200, 250)
    assert center_bbox == expected_center_bbox

def test_topleft_to_corners_format():
    corners_bbox = BBox.topleft_to_corners_format([100, 150, 200, 250])
    expected_corners_bbox = (100, 150, 100 + 200, 150 + 250)
    assert corners_bbox == expected_corners_bbox

def test_normalize():
    normalized_bbox = BBox.normalize([100, 150, 300, 400], TEST_IMAGE_WIDTH, TEST_IMAGE_HEIGHT, BBoxFormat.CORNERS)
    expected_normalized_bbox = (100 / TEST_IMAGE_WIDTH, 150 / TEST_IMAGE_HEIGHT, 300 / TEST_IMAGE_WIDTH, 400 / TEST_IMAGE_HEIGHT)
    assert normalized_bbox == expected_normalized_bbox

def test_absolute():
    input_coordinates = [0.1, 0.2, 0.3, 0.4]
    img_width = TEST_IMAGE_WIDTH
    img_height = TEST_IMAGE_HEIGHT
    absolute_coords = BBox.absolute(input_coordinates, img_width, img_height, BBoxFormat.CORNERS)
    expected_coords = (int(0.1 * TEST_IMAGE_WIDTH), int(0.2 * TEST_IMAGE_HEIGHT), int(0.3 * TEST_IMAGE_WIDTH), int(0.4 * TEST_IMAGE_HEIGHT))
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


class TestBBoxFromFile:
    def test_from_yolo_file(self, tmp_path):
        # Setup test file
        bbox0 = "0 0.794531 0.352083 0.207813 0.704167"
        bbox1 = "1 0.946936 0.432773 0.103601 0.464586"
        bbox_content = f"{bbox0}\n{bbox1}"
        yolo_file = tmp_path / "test_yolo.txt"
        yolo_file.write_text(bbox_content)

        # Test reading
        bboxes, class_ids = BBox.from_yolo_file(str(yolo_file), TEST_IMAGE_HEIGHT, TEST_IMAGE_WIDTH)

        # Verify results
        assert len(bboxes) == 2
        assert class_ids == [0, 1]
        
        # Verify bounding box coordinates in CORNERS_NORMALIZED format
        x_center, y_center, w, h = map(float, bbox0.split()[1:])
        exp_bbox0 = [
            max(0, (x_center - w / 2)),
            max(0, (y_center - h / 2)),
            max(0, (x_center + w / 2)),
            max(0, (y_center + h / 2))
        ]
        assert bboxes[0].get_coordinates() == pytest.approx(exp_bbox0, abs=1e-4)
        
        x_center, y_center, w, h = map(float, bbox1.split()[1:])
        exp_bbox1 = [
            max(0, (x_center - w / 2)), 
            max(0, (y_center - h / 2)), 
            max(0, (x_center + w / 2)), 
            max(0, (y_center + h / 2))
        ]
        assert bboxes[1].get_coordinates() == pytest.approx(exp_bbox1, abs=1e-4)
        
        # Verify bounding box coordinates in YOLO format
        exp_bbox0 = list(map(float, bbox0.split()[1:]))
        assert bboxes[0].get_coordinates(NotationFormat.YOLO) == pytest.approx(exp_bbox0, abs=1e-4)
        
        exp_bbox1 = list(map(float, bbox1.split()[1:]))
        assert bboxes[1].get_coordinates(NotationFormat.YOLO) == pytest.approx(exp_bbox1, abs=1e-4)
    
    def test_from_coco_file(self, tmp_path):
        # Setup test file
        coco_data = {
            "annotations": [
                {"image_id": 1, "bbox": [100, 150, 200, 250], "category_id": 1},
                {"image_id": 1, "bbox": [196, 150, 132, 83], "category_id": 2},
                {"image_id": 2, "bbox": [200, 250, 300, 350]} 
            ]
        }
        coco_file = tmp_path / "test_coco.json"
        coco_file.write_text(json.dumps(coco_data))

        # Test reading
        bboxes = BBox.from_coco_file(str(coco_file), 1, TEST_IMAGE_HEIGHT, TEST_IMAGE_WIDTH)
        assert len(bboxes) == 2

        # Verify bounding box coordinates in CORNERS_NORMALIZED format
        x1, y1, w, h = coco_data["annotations"][0]["bbox"]
        exp_bbox0 = [
            x1 / TEST_IMAGE_WIDTH, 
            y1 / TEST_IMAGE_HEIGHT, 
            (x1 + w) / TEST_IMAGE_WIDTH, 
            (y1 + h) / TEST_IMAGE_HEIGHT
        ]
        assert bboxes[0].get_coordinates() == pytest.approx(exp_bbox0, abs=1e-4)
        
        x1, y1, w, h = coco_data["annotations"][1]["bbox"]
        exp_bbox1 = [
            x1 / TEST_IMAGE_WIDTH, 
            y1 / TEST_IMAGE_HEIGHT, 
            (x1 + w) / TEST_IMAGE_WIDTH, 
            (y1 + h) / TEST_IMAGE_HEIGHT
        ]
        assert bboxes[1].get_coordinates() == pytest.approx(exp_bbox1, abs=1e-4)
        
        # Verify bounding box coordinates in COCO format
        exp_bbox0 = coco_data["annotations"][0]["bbox"]
        assert bboxes[0].get_coordinates(NotationFormat.COCO) == pytest.approx(exp_bbox0)
        
        exp_bbox1 = coco_data["annotations"][1]["bbox"]
        assert bboxes[1].get_coordinates(NotationFormat.COCO) == pytest.approx(exp_bbox1)

    def test_from_pascal_voc_file(self, tmp_path):
        # Setup test file
        voc_content = '''
        <annotation>
            <object>
                <name>person</name>
                <bndbox>
                    <xmin>100</xmin>
                    <ymin>150</ymin>
                    <xmax>300</xmax>
                    <ymax>400</ymax>
                </bndbox>
            </object>
            <object>
                <name>car</name>
                <bndbox>
                    <xmin>350</xmin>
                    <ymin>200</ymin>
                    <xmax>500</xmax>
                    <ymax>350</ymax>
                </bndbox>
            </object>
        </annotation>
        '''
        pascal_voc_file = tmp_path / "test_pascal_voc.xml"
        pascal_voc_file.write_text(voc_content)

        # Test reading
        bboxes = BBox.from_pascal_voc_file(str(pascal_voc_file), TEST_IMAGE_HEIGHT, TEST_IMAGE_WIDTH)
        assert len(bboxes) == 2

        # Verify bounding box coordinates in CORNERS_NORMALIZED format
        xmin0, ymin0, xmax0, ymax0 = 100, 150, 300, 400
        exp_bbox0 = [
            xmin0 / TEST_IMAGE_WIDTH,
            ymin0 / TEST_IMAGE_HEIGHT,
            xmax0 / TEST_IMAGE_WIDTH,
            ymax0 / TEST_IMAGE_HEIGHT
        ]
        assert bboxes[0].get_coordinates() == pytest.approx(exp_bbox0, abs=1e-4)

        xmin1, ymin1, xmax1, ymax1 = 350, 200, 500, 350
        exp_bbox1 = [
            xmin1 / TEST_IMAGE_WIDTH,
            ymin1 / TEST_IMAGE_HEIGHT,
            xmax1 / TEST_IMAGE_WIDTH,
            ymax1 / TEST_IMAGE_HEIGHT
        ]
        assert bboxes[1].get_coordinates() == pytest.approx(exp_bbox1, abs=1e-4)

        # Verify bounding box coordinates in Pascal VOC format
        assert bboxes[0].get_coordinates(NotationFormat.PASCALVOC) == pytest.approx([xmin0, ymin0, xmax0, ymax0])
        assert bboxes[1].get_coordinates(NotationFormat.PASCALVOC) == pytest.approx([xmin1, ymin1, xmax1, ymax1])
        
    def test_empty_files(self, tmp_path):
        # Test empty YOLO file
        yolo_file = tmp_path / "empty.txt"
        yolo_file.write_text("")
        bboxes, class_ids = BBox.from_yolo_file(str(yolo_file), TEST_IMAGE_HEIGHT, TEST_IMAGE_WIDTH)
        assert len(bboxes) == 0
        assert len(class_ids) == 0

        # Test empty COCO file
        coco_file = tmp_path / "empty.json"
        coco_file.write_text('{"annotations": []}')
        bboxes = BBox.from_coco_file(str(coco_file), 1, TEST_IMAGE_HEIGHT, TEST_IMAGE_WIDTH)
        assert len(bboxes) == 0

        # Test empty Pascal VOC file
        voc_file = tmp_path / "empty.xml"
        voc_file.write_text('<annotation></annotation>')
        bboxes = BBox.from_pascal_voc_file(str(voc_file), TEST_IMAGE_HEIGHT, TEST_IMAGE_WIDTH)
        assert len(bboxes) == 0


class TestComputeIoU:
    """Test class for computing IoU using BBox objects."""

    @pytest.fixture
    def fully_overlapping_boxes(self):
        """Fixture for fully overlapping bounding boxes."""
        box1 = [0, 0, 10, 10]
        box2 = [0, 0, 10, 10]
        return box1, box2

    @pytest.fixture
    def partially_overlapping_boxes(self):
        """Fixture for partially overlapping bounding boxes."""
        box1 = [0, 0, 10, 10]
        box2 = [5, 5, 15, 15]
        return box1, box2

    @pytest.fixture
    def non_overlapping_boxes(self):
        """Fixture for non-overlapping bounding boxes."""
        box1 = [0, 0, 10, 10]
        box2 = [15, 15, 20, 20]
        return box1, box2

    @pytest.fixture
    def center_format_boxes(self):
        """Fixture for bounding boxes in center format."""
        box1 = [5, 5, 10, 10]  # center (5,5), width 10, height 10
        box2 = [5, 5, 10, 10]  # center (5,5), width 10, height 10
        return box1, box2

    @pytest.fixture
    def topleft_format_boxes(self):
        """Fixture for bounding boxes in top-left format."""
        box1 = [0, 0, 10, 10]  # top-left (0,0), width 10, height 10
        box2 = [0, 0, 10, 10]  # top-left (0,0), width 10, height 10
        return box1, box2

    def test_iou_fully_overlapping_boxes(self, fully_overlapping_boxes):
        """Test the case where bounding boxes fully overlap."""
        box1, box2 = fully_overlapping_boxes
        iou = BBox.compute_iou(box1, box2, format=BBoxFormat.CORNERS, iou_type=IoUType.IOU)
        assert iou == pytest.approx(1.0)

    def test_iou_partially_overlapping_boxes(self, partially_overlapping_boxes):
        """Test the case where bounding boxes partially overlap."""
        box1, box2 = partially_overlapping_boxes
        iou = BBox.compute_iou(box1, box2, format=BBoxFormat.CORNERS, iou_type=IoUType.IOU)
        assert iou == pytest.approx(0.1429, abs=1e-4)

    def test_iou_no_overlapping_boxes(self, non_overlapping_boxes):
        """Test the case where bounding boxes do not overlap."""
        box1, box2 = non_overlapping_boxes
        iou = BBox.compute_iou(box1, box2, format=BBoxFormat.CORNERS, iou_type=IoUType.IOU)
        assert iou == pytest.approx(0.0)

    def test_iou_center_format(self, center_format_boxes):
        """Test the case where bounding boxes are in center format."""
        box1, box2 = center_format_boxes
        iou = BBox.compute_iou(box1, box2, format=BBoxFormat.CENTER, iou_type=IoUType.IOU)
        assert iou == pytest.approx(1.0)

    def test_iou_topleft_format(self, topleft_format_boxes):
        """Test the case where bounding boxes are in top-left format."""
        box1, box2 = topleft_format_boxes
        iou = BBox.compute_iou(box1, box2, format=BBoxFormat.TOPLEFT, iou_type=IoUType.IOU)
        assert iou == pytest.approx(1.0)

    def test_diou(self, partially_overlapping_boxes):
        """Test the case of DIoU."""
        box1, box2 = partially_overlapping_boxes
        diou = BBox.compute_iou(box1, box2, format=BBoxFormat.CORNERS, iou_type=IoUType.DIOU)
        assert diou == pytest.approx(0.031746, abs=1e-4)

    def test_ciou(self, partially_overlapping_boxes):
        """Test the case of CIoU."""
        box1, box2 = partially_overlapping_boxes
        ciou = BBox.compute_iou(box1, box2, format=BBoxFormat.CORNERS, iou_type=IoUType.CIOU)
        assert ciou <= 0.1428  # CIoU is always <= IoU

    def test_giou(self, partially_overlapping_boxes):
        """Test the case of GIoU."""
        box1, box2 = partially_overlapping_boxes
        giou = BBox.compute_iou(box1, box2, format=BBoxFormat.CORNERS, iou_type=IoUType.GIOU)
        assert giou <= 0.1428  # GIoU is always <= IoU

    def test_invalid_format(self, fully_overlapping_boxes):
        """Test the case where the format is invalid."""
        box1, box2 = fully_overlapping_boxes
        with pytest.raises(TypeError):
            BBox.compute_iou(box1, box2, format="invalid_format", iou_type=IoUType.IOU)

    def test_invalid_iou_type(self, fully_overlapping_boxes):
        """Test the case where the IoU type is invalid."""
        box1, box2 = fully_overlapping_boxes
        with pytest.raises(TypeError):
            BBox.compute_iou(box1, box2, format=BBoxFormat.CORNERS, iou_type="invalid_type")

# Test cases for BBox class
class TestComputeIoUBBox:
    """Test class for IoU calculations using BBox objects."""

    @pytest.fixture
    def bbox_setup(self) -> Tuple[BBox, BBox]:
        """Fixture to set up common BBox objects for IoU tests."""
        img_height = 100
        img_width = 200
        box1 = [0, 0, 10, 10]
        box2 = [5, 5, 15, 15]
        bbox1 = BBox(img_height, img_width, box1, BBoxFormat.CORNERS)
        bbox2 = BBox(img_height, img_width, box2, BBoxFormat.CORNERS)
        return bbox1, bbox2

    def test_iou_calculation(self, bbox_setup: Tuple[BBox, BBox]):
        """Test the IoU calculation between two bounding boxes."""
        bbox1, bbox2 = bbox_setup
        iou = bbox1.compute_iou(bbox2)
        assert iou == pytest.approx(0.1429, abs=1e-4)

    def test_diou_calculation(self, bbox_setup:Tuple[BBox, BBox]):
        """Test the DIoU calculation between two bounding boxes."""
        bbox1, bbox2 = bbox_setup
        diou = bbox1.compute_iou(bbox2, iou_type=IoUType.DIOU)
        assert diou == pytest.approx(0.031746, abs=1e-4)

    def test_ciou_calculation(self, bbox_setup:Tuple[BBox, BBox]):
        """Test the CIoU calculation between two bounding boxes."""
        bbox1, bbox2 = bbox_setup
        ciou = bbox1.compute_iou(bbox2, iou_type=IoUType.CIOU)
        assert ciou <= 0.1428  # CIoU is always <= IoU

    def test_giou_calculation(self, bbox_setup:Tuple[BBox, BBox]):
        """Test the GIoU calculation between two bounding boxes."""
        bbox1, bbox2 = bbox_setup
        giou = bbox1.compute_iou(bbox2, iou_type=IoUType.GIOU)
        assert giou <= 0.1428  # GIoU is always <= IoU

if __name__ == "__main__":
    pytest.main()