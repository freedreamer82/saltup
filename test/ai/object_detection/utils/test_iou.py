import unittest
import numpy as np
from saltup.ai.object_detection.utils.bbox import BBox, BBoxFormat
from saltup.ai.object_detection.utils.bbox import compute_iou, IoUType

class TestComputeIoU(unittest.TestCase):
    def test_iou_fully_overlapping_boxes(self):
        """Test the case where bounding boxes fully overlap."""
        box1 = [0, 0, 10, 10]
        box2 = [0, 0, 10, 10]
        iou = compute_iou(box1, box2, format=BBoxFormat.CORNERS, iou_type=IoUType.IOU)
        self.assertAlmostEqual(iou, 1.0)

    def test_iou_partially_overlapping_boxes(self):
        """Test the case where bounding boxes partially overlap."""
        box1 = [0, 0, 10, 10]
        box2 = [5, 5, 15, 15]
        iou = compute_iou(box1, box2, format=BBoxFormat.CORNERS, iou_type=IoUType.IOU)
        iou_rounded = round(iou, 4)  # Round to 4 decimal places
        self.assertAlmostEqual(iou_rounded, 0.1429, places=4)  # Update the expected value

    def test_iou_no_overlapping_boxes(self):
        """Test the case where bounding boxes do not overlap."""
        box1 = [0, 0, 10, 10]
        box2 = [15, 15, 20, 20]
        iou = compute_iou(box1, box2, format=BBoxFormat.CORNERS, iou_type=IoUType.IOU)
        self.assertAlmostEqual(iou, 0.0)

    def test_iou_center_format(self):
        """Test the case where bounding boxes are in center format."""
        box1 = [5, 5, 10, 10]  # center (5,5), width 10, height 10
        box2 = [5, 5, 10, 10]  # center (5,5), width 10, height 10
        iou = compute_iou(box1, box2, format=BBoxFormat.CENTER, iou_type=IoUType.IOU)
        self.assertAlmostEqual(iou, 1.0)

    def test_iou_topleft_format(self):
        """Test the case where bounding boxes are in top-left format."""
        box1 = [0, 0, 10, 10]  # top-left (0,0), width 10, height 10
        box2 = [0, 0, 10, 10]  # top-left (0,0), width 10, height 10
        iou = compute_iou(box1, box2, format=BBoxFormat.TOPLEFT, iou_type=IoUType.IOU)
        self.assertAlmostEqual(iou, 1.0)

    def test_diou(self):
        """Test the case of DIoU."""
        box1 = [0, 0, 10, 10]
        box2 = [5, 5, 15, 15]
        diou = compute_iou(box1, box2, format=BBoxFormat.CORNERS, iou_type=IoUType.DIOU)
        self.assertAlmostEqual(diou, 0.031746, places=4)  # Update the expected value

    def test_ciou(self):
        """Test the case of CIoU."""
        box1 = [0, 0, 10, 10]
        box2 = [5, 5, 15, 15]
        ciou = compute_iou(box1, box2, format=BBoxFormat.CORNERS, iou_type=IoUType.CIOU)
        self.assertLessEqual(ciou, 0.1428)  # CIoU is always <= IoU

    def test_giou(self):
        """Test the case of GIoU."""
        box1 = [0, 0, 10, 10]
        box2 = [5, 5, 15, 15]
        giou = compute_iou(box1, box2, format=BBoxFormat.CORNERS, iou_type=IoUType.GIOU)
        self.assertLessEqual(giou, 0.1428)  # GIoU is always <= IoU

    def test_invalid_format(self):
        """Test the case where the format is invalid."""
        box1 = [0, 0, 10, 10]
        box2 = [0, 0, 10, 10]
        with self.assertRaises(TypeError):
            compute_iou(box1, box2, format="invalid_format", iou_type=IoUType.IOU)

    def test_invalid_iou_type(self):
        """Test the case where the IoU type is invalid."""
        box1 = [0, 0, 10, 10]
        box2 = [0, 0, 10, 10]
        with self.assertRaises(TypeError):
            compute_iou(box1, box2, format=BBoxFormat.CORNERS, iou_type="invalid_type")

if __name__ == "__main__":
    unittest.main()