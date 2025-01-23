import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import AUC  # type: ignore
import unittest
from saltup.ai.object_detection.utils.bbox import BBox, BBoxFormat
from saltup.ai.object_detection.utils.metrics import compute_ap, compute_ap_for_threshold, compute_map_50_95


class TestAveragePrecision(unittest.TestCase):
    def setUp(self):
        # Create more challenging test cases
        self.gt_bbox1 = BBox(img_height=100, img_width=100, coordinates=[10, 10, 20, 20], format=BBoxFormat.CORNERS)
        self.gt_bbox2 = BBox(img_height=100, img_width=100, coordinates=[30, 30, 40, 40], format=BBoxFormat.CORNERS)
        
        # Good match but not perfect
        self.pred_bbox1 = BBox(img_height=100, img_width=100, coordinates=[12, 12, 21, 21], format=BBoxFormat.CORNERS)
        
        # Clear false positive - completely different location
        self.pred_bbox2 = BBox(img_height=100, img_width=100, coordinates=[50, 50, 60, 60], format=BBoxFormat.CORNERS)
        
        # Partial overlap but below threshold false positive
        self.pred_bbox3 = BBox(img_height=100, img_width=100, coordinates=[15, 15, 25, 25], format=BBoxFormat.CORNERS)
        
        # Perfect match for gt_bbox2
        self.pred_bbox4 = BBox(img_height=100, img_width=100, coordinates=[30, 30, 40, 40], format=BBoxFormat.CORNERS)

    def test_false_positive(self):
        """Test case with false positives"""
        gt_bboxes = [self.gt_bbox1]  # Only one ground truth box
        
        # Test with false positives having higher confidence than true positive
        pred_bboxes_scores = [
            (self.pred_bbox2, 0.99),  # False positive with highest confidence
            (self.pred_bbox3, 0.98),  # Another false positive
            (self.pred_bbox1, 0.95),  # Good match but lower confidence
        ]
        
        ap = compute_ap_for_threshold(gt_bboxes, pred_bboxes_scores, threshold=0.5)
        print(f"False positive test AP: {ap}")
        print("IoUs:")
        for pred, score in pred_bboxes_scores:
            iou = pred.compute_iou(self.gt_bbox1)
            print(f"Prediction (score={score:.2f}) IoU: {iou:.3f}")
        
        # AP should be significantly lower due to high confidence false positives
        self.assertLess(ap, 0.5)

    def test_confidence_ordering(self):
        """Test that confidence scores affect AP correctly"""
        gt_bboxes = [self.gt_bbox1]
        
        # Good match first
        pred_scores_good_first = [
            (self.pred_bbox1, 0.99),  # Good match with highest confidence
            (self.pred_bbox2, 0.8),   # False positive
            (self.pred_bbox3, 0.7),   # False positive
        ]
        
        # False positives first
        pred_scores_bad_first = [
            (self.pred_bbox2, 0.99),  # False positive with highest confidence
            (self.pred_bbox3, 0.98),  # False positive
            (self.pred_bbox1, 0.7),   # Good match but low confidence
        ]
        
        ap_good_first = compute_ap_for_threshold(gt_bboxes, pred_scores_good_first, threshold=0.5)
        ap_bad_first = compute_ap_for_threshold(gt_bboxes, pred_scores_bad_first, threshold=0.5)
        
        print(f"AP with good match first: {ap_good_first}")
        print(f"AP with false positives first: {ap_bad_first}")
        
        # AP should be higher when good matches have higher confidence
        self.assertGreater(ap_good_first, ap_bad_first)

    def test_perfect_match(self):
        """Test case with a perfect match"""
        gt_bboxes = [self.gt_bbox1]
        pred_bboxes_scores = [(self.pred_bbox1, 0.99)]  # Perfect match
        
        ap = compute_ap_for_threshold(gt_bboxes, pred_bboxes_scores, threshold=0.5)
        self.assertEqual(ap, 1.0)

    def test_no_predictions(self):
        """Test case with no predictions"""
        gt_bboxes = [self.gt_bbox1]
        pred_bboxes_scores = []  # No predictions
        
        ap = compute_ap_for_threshold(gt_bboxes, pred_bboxes_scores, threshold=0.5)
        self.assertEqual(ap, 0.0)

    def test_multiple_gt_and_pred(self):
        """Test case with multiple ground truths and predictions"""
        gt_bboxes = [self.gt_bbox1, self.gt_bbox2]
        
        pred_bboxes_scores = [
            (self.pred_bbox1, 0.99),  # Good match for gt_bbox1
            (self.pred_bbox4, 0.98),  # Perfect match for gt_bbox2
            (self.pred_bbox2, 0.97),  # False positive
        ]
        
        ap = compute_map_50_95(gt_bboxes, pred_bboxes_scores)
        print(f"AP for multiple ground truths and predictions: {ap}")
        self.assertGreater(ap, 0.3)  # Updated expected value

    def test_compute_ap(self):
        """Test the compute_ap function directly"""
        recall = np.array([0.0, 0.5, 1.0])
        precision = np.array([1.0, 0.5, 0.0])
        
        ap = compute_ap(recall, precision)
        self.assertEqual(ap, 0.5)  # Correct expected value


if __name__ == '__main__':
    unittest.main()