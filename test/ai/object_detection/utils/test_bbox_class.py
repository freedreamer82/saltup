import pytest
from saltup.ai.object_detection.utils.bbox import BBox, BBoxFormat, IoUType
import numpy as np
from copy import deepcopy
import json
import xml.etree.ElementTree as ET

# Test data
IMG_WIDTH = 640
IMG_HEIGHT = 480

# Test cases for BBox initialization
def test_bbox_initialization():
    bbox = BBox(img_height=IMG_HEIGHT, img_width=IMG_WIDTH, coordinates=[0.1, 0.2, 0.3, 0.4])
    assert bbox.get_coordinates() == [0.1, 0.2, 0.3, 0.4]

# Test cases for copy method
def test_bbox_copy():
    bbox = BBox(img_height=IMG_HEIGHT, img_width=IMG_WIDTH, coordinates=[0.1, 0.2, 0.3, 0.4])
    bbox_copy = bbox.copy()
    assert bbox.get_coordinates() == bbox_copy.get_coordinates()

# Test cases for is_normalized method
def test_is_normalized():
    assert BBox.is_normalized([0.1, 0.2, 0.3, 0.4], BBoxFormat.CORNERS) == True
    assert BBox.is_normalized([1.1, 0.2, 0.3, 0.4], BBoxFormat.CORNERS) == False

# Test cases for pascalvoc_to_yolo_bbox method
def test_pascalvoc_to_yolo_bbox():
    yolo_bbox = BBox.pascalvoc_to_yolo_bbox([100, 150, 300, 400], IMG_WIDTH, IMG_HEIGHT)
    expected_yolo_bbox = ((100 + 300) / (2.0 * IMG_WIDTH), (150 + 400) / (2.0 * IMG_HEIGHT), (300 - 100) / IMG_WIDTH, (400 - 150) / IMG_HEIGHT)
    assert yolo_bbox == expected_yolo_bbox

# Test cases for corners_to_center_format method
def test_corners_to_center_format():
    center_bbox = BBox.corners_to_center_format([100, 150, 300, 400])
    expected_center_bbox = ((100 + 300) / 2, (150 + 400) / 2, 300 - 100, 400 - 150)
    assert center_bbox == expected_center_bbox

# Test cases for corners_to_topleft_format method
def test_corners_to_topleft_format():
    topleft_bbox = BBox.corners_to_topleft_format([100, 150, 300, 400])
    expected_topleft_bbox = (100, 150, 300 - 100, 400 - 150)
    assert topleft_bbox == expected_topleft_bbox

# Test cases for center_to_corners_format method
def test_center_to_corners_format():
    corners_bbox = BBox.center_to_corners_format([200, 275, 200, 250])
    expected_corners_bbox = (200 - 100, 275 - 125, 200 + 100, 275 + 125)
    assert corners_bbox == expected_corners_bbox

# Test cases for center_to_topleft_format method
def test_center_to_topleft_format():
    topleft_bbox = BBox.center_to_topleft_format([200, 275, 200, 250])
    expected_topleft_bbox = (200 - 100, 275 - 125, 200, 250)
    assert topleft_bbox == expected_topleft_bbox

# Test cases for topleft_to_center_format method
def test_topleft_to_center_format():
    center_bbox = BBox.topleft_to_center_format([100, 150, 200, 250])
    expected_center_bbox = (100 + 100, 150 + 125, 200, 250)
    assert center_bbox == expected_center_bbox

# Test cases for topleft_to_corners_format method
def test_topleft_to_corners_format():
    corners_bbox = BBox.topleft_to_corners_format([100, 150, 200, 250])
    expected_corners_bbox = (100, 150, 100 + 200, 150 + 250)
    assert corners_bbox == expected_corners_bbox

# Test cases for normalize method
def test_normalize():
    normalized_bbox = BBox.normalize([100, 150, 300, 400], IMG_WIDTH, IMG_HEIGHT, BBoxFormat.CORNERS)
    expected_normalized_bbox = (100 / IMG_WIDTH, 150 / IMG_HEIGHT, 300 / IMG_WIDTH, 400 / IMG_HEIGHT)
    assert normalized_bbox == expected_normalized_bbox

# Test cases for absolute method
def test_absolute():
    absolute_bbox = BBox.absolute([0.1, 0.2, 0.3, 0.4], IMG_WIDTH, IMG_HEIGHT, BBoxFormat.CORNERS)
    expected_absolute_bbox = (int(0.1 * IMG_WIDTH), int(0.2 * IMG_HEIGHT), int(0.3 * IMG_WIDTH), int(0.4 * IMG_HEIGHT))
    assert absolute_bbox == expected_absolute_bbox

# Test cases for compute_iou method
def test_compute_iou():
    bbox1 = BBox(img_height=IMG_HEIGHT, img_width=IMG_WIDTH, coordinates=[0.1, 0.1, 0.3, 0.3])
    bbox2 = BBox(img_height=IMG_HEIGHT, img_width=IMG_WIDTH, coordinates=[0.2, 0.2, 0.4, 0.4])
    iou = bbox1.compute_iou(bbox2)
    assert 0 <= iou <= 1

# Test cases for from_yolo_file method
def test_from_yolo_file(tmp_path):
    yolo_file = tmp_path / "test_yolo.txt"
    yolo_file.write_text("0 0.5 0.5 0.2 0.2\n")
    bboxes, class_ids = BBox.from_yolo_file(str(yolo_file), IMG_HEIGHT, IMG_WIDTH)
    assert len(bboxes) == 1
    assert class_ids[0] == 0

# Test cases for from_coco_file method
def test_from_coco_file(tmp_path):
    coco_file = tmp_path / "test_coco.json"
    coco_data = {
        "annotations": [
            {"image_id": 1, "bbox": [100, 150, 200, 250]},
            {"image_id": 2, "bbox": [200, 250, 300, 350]}
        ]
    }
    coco_file.write_text(json.dumps(coco_data))
    bboxes = BBox.from_coco_file(str(coco_file), 1, IMG_HEIGHT, IMG_WIDTH)
    assert len(bboxes) == 1

# Test cases for from_pascal_voc_file method
def test_from_pascal_voc_file(tmp_path):
    pascal_voc_file = tmp_path / "test_pascal_voc.xml"
    pascal_voc_file.write_text('''
    <annotation>
        <object>
            <bndbox>
                <xmin>100</xmin>
                <ymin>150</ymin>
                <xmax>300</xmax>
                <ymax>400</ymax>
            </bndbox>
        </object>
    </annotation>
    ''')
    bboxes = BBox.from_pascal_voc_file(str(pascal_voc_file), IMG_HEIGHT, IMG_WIDTH)
    assert len(bboxes) == 1

# Test cases for to_yolo method
def test_to_yolo():
    bbox = BBox(img_height=IMG_HEIGHT, img_width=IMG_WIDTH, coordinates=[0.1, 0.2, 0.3, 0.4])
    yolo_bbox = bbox.to_yolo()
    expected_yolo_bbox = BBox.pascalvoc_to_yolo_bbox([0.1 * IMG_WIDTH, 0.2 * IMG_HEIGHT, 0.3 * IMG_WIDTH, 0.4 * IMG_HEIGHT], IMG_WIDTH, IMG_HEIGHT)
    assert yolo_bbox == expected_yolo_bbox

# Test cases for to_coco method
def test_to_coco():
    bbox = BBox(img_height=IMG_HEIGHT, img_width=IMG_WIDTH, coordinates=[0.1, 0.2, 0.3, 0.4])
    coco_bbox = bbox.to_coco()
    expected_coco_bbox = (0.1 * IMG_WIDTH, 0.2 * IMG_HEIGHT, (0.3 - 0.1) * IMG_WIDTH, (0.4 - 0.2) * IMG_HEIGHT)
    assert coco_bbox == pytest.approx(expected_coco_bbox)

# Test cases for to_pascal_voc method
def test_to_pascal_voc():
    bbox = BBox(img_height=IMG_HEIGHT, img_width=IMG_WIDTH, coordinates=[0.1, 0.2, 0.3, 0.4])
    pascal_voc_bbox = bbox.to_pascal_voc()
    expected_pascal_voc_bbox = (0.1 * IMG_WIDTH, 0.2 * IMG_HEIGHT, 0.3 * IMG_WIDTH, 0.4 * IMG_HEIGHT)
    assert pascal_voc_bbox == expected_pascal_voc_bbox

# Test cases for set_coordinates method
def test_set_coordinates():
    bbox = BBox(img_height=IMG_HEIGHT, img_width=IMG_WIDTH, coordinates=[0.1, 0.2, 0.3, 0.4])
    bbox.set_coordinates(IMG_HEIGHT, IMG_WIDTH, [0.2, 0.3, 0.4, 0.5], BBoxFormat.CORNERS)
    assert bbox.get_coordinates() == [0.2, 0.3, 0.4, 0.5]

# Test cases for __repr__ method
def test_repr():
    bbox = BBox(img_height=IMG_HEIGHT, img_width=IMG_WIDTH, coordinates=[0.1, 0.2, 0.3, 0.4])
    repr_str = repr(bbox)
    assert "BBox" in repr_str
    assert "coordinates=[0.1, 0.2, 0.3, 0.4]" in repr_str
    assert f"img_width={IMG_WIDTH}" in repr_str
    assert f"img_height={IMG_HEIGHT}" in repr_str

# Run the tests
if __name__ == "__main__":
    pytest.main()