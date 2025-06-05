import unittest
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict, Union
from saltup.ai.object_detection.yolo.yolo import BaseYolo, BBox, YoloOutput, BBoxFormat
from saltup.utils.data.image import ColorMode, ImageFormat, Image
import onnx
from onnx import helper, TensorProto
import tempfile
import os
from typing import Any, Dict, List, Tuple
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from saltup.ai.nn_model import NeuralNetworkModel


class TestEvaluate(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.onnx_model_path = os.path.join(self.tmp_dir, "test_model.onnx")
        self.create_simple_onnx_model(self.onnx_model_path)

        class MockYolo(BaseYolo):
            def preprocess(self, 
                   image: Image,
                   target_height: int, 
                   target_width: int,        
                   normalize_method: callable = lambda x: x.astype(np.float32) / 255.0,
                   apply_padding: bool = True,
                   **kwargs: Any
                   ) -> np.ndarray:
                return image

            def get_input_info(self) -> Tuple[tuple, ColorMode, ImageFormat]:
                input_shape = self._model_input_shape[1:]  # Remove batch size
                return (
                    input_shape,  # Shape: (480, 640, 1)
                    ColorMode.RGB,
                    ImageFormat.HWC
                )
            
            def postprocess(self, raw_output: np.ndarray,
                    image_height: int, image_width: int, confidence_thr: float = 0.5, 
                            iou_threshold: float = 0.5) -> List[Tuple[BBox, int, float]]:
                return []
        
        self.yolo = MockYolo(yolot=None, model=NeuralNetworkModel(self.onnx_model_path), number_class=1)

    def create_simple_onnx_model(self, model_path: str):
        # Create a simple ONNX model with a single Identity node
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 224, 224])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 224, 224])
        node = helper.make_node('Identity', ['input'], ['output'])
        graph = helper.make_graph([node], 'simple_model', [input], [output])
        model = helper.make_model(graph, producer_name='onnx-yolo-example', opset_imports=[helper.make_opsetid("", 16)])
        onnx.save(model, model_path)

    def tearDown(self):
        if os.path.exists(self.onnx_model_path):
            os.remove(self.onnx_model_path)
        os.rmdir(self.tmp_dir)

    def test_evaluate_perfect_match(self):
        """Test the case where all predictions correspond exactly to ground truths."""
        bbox1 = BBox(img_height=100, img_width=100, coordinates=[0, 0, 10, 10], fmt=BBoxFormat.CORNERS_ABSOLUTE)  
        bbox2 = BBox(img_height=100, img_width=100, coordinates=[20, 20, 30, 30], fmt=BBoxFormat.CORNERS_ABSOLUTE)  
        predictions = YoloOutput([(bbox1, 0, 0.9), (bbox2, 1, 0.8)])   
        ground_truth = [(bbox1, 0), (bbox2, 1)]

        metrics = self.yolo.evaluate(predictions, ground_truth)
        self.assertAlmostEqual(metrics["precision"], 1.0)
        self.assertAlmostEqual(metrics["recall"], 1.0)
        self.assertAlmostEqual(metrics["f1"], 1.0)
        self.assertAlmostEqual(metrics["mAP"], 1.0)
        self.assertAlmostEqual(metrics["mAP@50-95"], 1.0)

    def test_evaluate_false_positives(self):
        """Test the case where there are false positives."""
        bbox1 = BBox(img_height=100, img_width=100, coordinates=[0, 0, 10, 10], fmt=BBoxFormat.CORNERS_ABSOLUTE)  
        bbox2 = BBox(img_height=100, img_width=100, coordinates=[20, 20, 30, 30], fmt=BBoxFormat.CORNERS_ABSOLUTE)  
        bbox3 = BBox(img_height=100, img_width=100, coordinates=[40, 40, 50, 50], fmt=BBoxFormat.CORNERS_ABSOLUTE)   
        predictions = YoloOutput([(bbox1, 0, 0.9), (bbox2, 1, 0.8), (bbox3, 1, 0.7)])   
        ground_truth = [(bbox1, 0), (bbox2, 1)]

        metrics = self.yolo.evaluate(predictions, ground_truth)
        self.assertAlmostEqual(metrics["precision"], 2 / 3)
        self.assertAlmostEqual(metrics["recall"], 1.0)
        self.assertAlmostEqual(metrics["f1"], 0.8)
        self.assertAlmostEqual(metrics["mAP"], 1.0)
        self.assertAlmostEqual(metrics["mAP@50-95"], 1.0)

    def test_evaluate_false_negatives(self):
        """Test the case where there are false negatives."""
        bbox1 = BBox(img_height=100, img_width=100, coordinates=[0, 0, 10, 10], fmt=BBoxFormat.CORNERS_ABSOLUTE)
        bbox2 = BBox(img_height=100, img_width=100, coordinates=[20, 20, 30, 30], fmt=BBoxFormat.CORNERS_ABSOLUTE)
        bbox3 = BBox(img_height=100, img_width=100, coordinates=[40, 40, 50, 50], fmt=BBoxFormat.CORNERS_ABSOLUTE)
        predictions = YoloOutput([(bbox1, 0, 0.9), (bbox2, 1, 0.8)])
        ground_truth = [(bbox1, 0), (bbox2, 1), (bbox3, 1)]

        metrics = self.yolo.evaluate(predictions, ground_truth)
        self.assertAlmostEqual(metrics["precision"], 1.0)
        self.assertAlmostEqual(metrics["recall"], 2 / 3)
        self.assertAlmostEqual(metrics["f1"], 0.8)
        self.assertAlmostEqual(metrics["mAP"], 0.875)
        self.assertAlmostEqual(metrics["mAP@50-95"], 0.833, delta=0.0005)

    def test_evaluate_no_predictions(self):
        """Test the case where there are no predictions."""
        bbox1 = BBox(img_height=100, img_width=100, coordinates=[0, 0, 10, 10], fmt=BBoxFormat.CORNERS_ABSOLUTE)  
        bbox2 = BBox(img_height=100, img_width=100, coordinates=[20, 20, 30, 30], fmt=BBoxFormat.CORNERS_ABSOLUTE)  
        predictions = YoloOutput([])  
        ground_truth = [(bbox1, 0), (bbox2, 1)]

        metrics = self.yolo.evaluate(predictions, ground_truth)
        self.assertAlmostEqual(metrics["precision"], 0.0)
        self.assertAlmostEqual(metrics["recall"], 0.0)
        self.assertAlmostEqual(metrics["f1"], 0.0)
        self.assertAlmostEqual(metrics["mAP"], 0.0)
        self.assertAlmostEqual(metrics["mAP@50-95"], 0.0)

    def test_evaluate_no_ground_truth(self):
        """Test the case where there are no ground truths."""
        bbox1 = BBox(img_height=100, img_width=100, coordinates=[0, 0, 10, 10], fmt=BBoxFormat.CORNERS_ABSOLUTE)   
        bbox2 = BBox(img_height=100, img_width=100, coordinates=[20, 20, 30, 30], fmt=BBoxFormat.CORNERS_ABSOLUTE)  
        predictions = YoloOutput([(bbox1, 0, 0.9), (bbox2, 1, 0.8)])   
        ground_truth = []

        metrics = self.yolo.evaluate(predictions, ground_truth)
        self.assertAlmostEqual(metrics["precision"], 0.0)
        self.assertAlmostEqual(metrics["recall"], 0.0)
        self.assertAlmostEqual(metrics["f1"], 0.0)
        self.assertAlmostEqual(metrics["mAP"], 0.0)
        self.assertAlmostEqual(metrics["mAP@50-95"], 0.0)

    def test_evaluate_partial_overlap(self):
        """Test the case where predictions have partial overlap with ground truths."""
        bbox1 = BBox(img_height=100, img_width=100, coordinates=[0, 0, 10, 10], fmt=BBoxFormat.CORNERS_ABSOLUTE)
        bbox2 = BBox(img_height=100, img_width=100, coordinates=[1, 1, 11, 11], fmt=BBoxFormat.CORNERS_ABSOLUTE)
        predictions = YoloOutput([(bbox2, 0, 0.9)])
        ground_truth = [(bbox1, 0)]

        metrics = self.yolo.evaluate(predictions, ground_truth, threshold_iou=0.5)
        self.assertAlmostEqual(metrics["precision"], 1.0)   
        self.assertAlmostEqual(metrics["recall"], 1.0)      
        self.assertAlmostEqual(metrics["f1"], 1.0)         
        self.assertAlmostEqual(metrics["mAP"], 1.0)         
        self.assertAlmostEqual(metrics["mAP@50-95"], 0.4, delta=0.1)

    def test_evaluate_comparison_with_torchmetrics(self):
        """
        Compare the results of our implementation with torchmetrics.
        Verify that both implementations produce similar results.
        """
        # Create test data
        bbox1 = BBox(img_height=100, img_width=100, coordinates=[0, 0, 10, 10], fmt=BBoxFormat.CORNERS_ABSOLUTE)
        bbox2 = BBox(img_height=100, img_width=100, coordinates=[20, 20, 30, 30], fmt=BBoxFormat.CORNERS_ABSOLUTE)
        bbox3 = BBox(img_height=100, img_width=100, coordinates=[40, 40, 50, 50], fmt=BBoxFormat.CORNERS_ABSOLUTE)
        
        predictions = YoloOutput([
            (bbox1, 0, 0.9),
            (bbox2, 1, 0.8),
            (bbox3, 0, 0.7)
        ])
        
        ground_truth = [(bbox1, 0), (bbox2, 1), (bbox3, 0)]
        
        # Calculate metrics with our implementation
        our_metrics = self.yolo.evaluate(predictions, ground_truth)
        
        # Prepare data for torchmetrics
        pred_boxes = []
        pred_scores = []
        pred_labels = []
        
        # Access YoloOutput data using get_boxes()
        for box, cls, score in predictions.get_boxes():
            coords = box.get_coordinates(fmt=BBoxFormat.CORNERS_NORMALIZED)
            pred_boxes.append([
                coords[0],
                coords[1],
                coords[2],
                coords[3]
            ])
            pred_scores.append(score)
            pred_labels.append(cls + 1)  # torchmetrics uses 1-based labels
        
        gt_boxes = []
        gt_labels = []
        
        for box, cls in ground_truth:
            coords = box.get_coordinates(fmt=BBoxFormat.CORNERS_NORMALIZED)
            gt_boxes.append([
                coords[0],
                coords[1],
                coords[2],
                coords[3]
            ])
            gt_labels.append(cls + 1)
        
        preds = [{
            'boxes': torch.tensor(pred_boxes, dtype=torch.float32),
            'scores': torch.tensor(pred_scores, dtype=torch.float32),
            'labels': torch.tensor(pred_labels, dtype=torch.int32)
        }]
        
        target = [{
            'boxes': torch.tensor(gt_boxes, dtype=torch.float32),
            'labels': torch.tensor(gt_labels, dtype=torch.int32)
        }]
        
        # Calculate metrics with torchmetrics
        metric = MeanAveragePrecision(box_format='xyxy')
        metric.update(preds, target)
        torch_metrics = metric.compute()
        
        # Compare the results of the two implementations
        self.assertAlmostEqual(
            our_metrics["mAP"],
            torch_metrics['map'].item(),
            places=2,
            msg="mAP metrics differ significantly between implementations"
        )
        
        self.assertAlmostEqual(
            our_metrics["mAP@50-95"],
            torch_metrics['map_50'].item(),
            places=2,
            msg="mAP@50-95 metrics differ significantly between implementations"
        )
        
        # Check also the other metrics of our implementation
        self.assertGreater(our_metrics["precision"], 0.8)
        self.assertGreater(our_metrics["recall"], 0.8)
        self.assertGreater(our_metrics["f1"], 0.8)
            
if __name__ == "__main__":
    unittest.main()