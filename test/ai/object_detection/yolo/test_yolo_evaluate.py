import unittest
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict
from saltup.ai.object_detection.yolo.yolo import BaseYolo, BBox, YoloOutput,BBoxFormat
 
import onnx
from onnx import helper, TensorProto
import tempfile
import os
from typing import Any, Dict, List ,Tuple


class TestEvaluate(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.onnx_model_path = os.path.join(self.tmp_dir, "test_model.onnx")
        self.create_simple_onnx_model(self.onnx_model_path)

        class MockYolo(BaseYolo):
            def preprocess(self, image: np.array, target_height: int, target_width: int) -> np.ndarray:
                return image

            def postprocess(self, raw_output: np.ndarray, image_height: int, image_width: int, confidence_thr: float = 0.5, iou_threshold: float = 0.5) -> List[Tuple[BBox, int, float]]:
                return []

        self.yolo = MockYolo(yolot=None, model_path=self.onnx_model_path, number_class=1)

    def create_simple_onnx_model(self, model_path: str):
        # Creiamo un semplice modello ONNX con un solo nodo Identity
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 224, 224])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 224, 224])
        node = helper.make_node('Identity', ['input'], ['output'])
        graph = helper.make_graph([node], 'simple_model', [input], [output])
        model = helper.make_model(graph)
        onnx.save(model, model_path)

    def tearDown(self):
        if os.path.exists(self.onnx_model_path):
            os.remove(self.onnx_model_path)
        os.rmdir(self.tmp_dir)

    def test_evaluate_perfect_match(self):
        """Testa il caso in cui tutte le previsioni corrispondono esattamente alle verità fondamentali."""
        bbox1 = BBox(coordinates=[0, 0, 10, 10], format=BBoxFormat.CORNERS)  
        bbox2 = BBox(coordinates=[20, 20, 30, 30], format=BBoxFormat.CORNERS)  
        predictions = YoloOutput([(bbox1, 0, 0.9), (bbox2, 1, 0.8)])   
        ground_truth = [(bbox1, 0), (bbox2, 1)]

        metrics = self.yolo.evaluate(predictions, ground_truth)
        self.assertAlmostEqual(metrics["precision"], 1.0)
        self.assertAlmostEqual(metrics["recall"], 1.0)
        self.assertAlmostEqual(metrics["f1"], 1.0)
        self.assertAlmostEqual(metrics["mAP"], 1.0)
        self.assertAlmostEqual(metrics["mAP@50-95"], 1.0)

    def test_evaluate_false_positives(self):
        """Testa il caso in cui ci sono falsi positivi."""
        bbox1 = BBox(coordinates=[0, 0, 10, 10], format=BBoxFormat.CORNERS)  
        bbox2 = BBox(coordinates=[20, 20, 30, 30], format=BBoxFormat.CORNERS)  
        bbox3 = BBox(coordinates=[40, 40, 50, 50], format=BBoxFormat.CORNERS)   
        predictions = YoloOutput([(bbox1, 0, 0.9), (bbox2, 1, 0.8), (bbox3, 1, 0.7)])   
        ground_truth = [(bbox1, 0), (bbox2, 1)]

        metrics = self.yolo.evaluate(predictions, ground_truth)
        self.assertAlmostEqual(metrics["precision"], 2 / 3)
        self.assertAlmostEqual(metrics["recall"], 1.0)
        self.assertAlmostEqual(metrics["f1"], 0.8)
        self.assertAlmostEqual(metrics["mAP"], 1.0)
        self.assertAlmostEqual(metrics["mAP@50-95"], 1.0)

    def test_evaluate_false_negatives(self):
        """Testa il caso in cui ci sono falsi negativi."""
        bbox1 = BBox(coordinates=[0, 0, 10, 10], format=BBoxFormat.CORNERS)  
        bbox2 = BBox(coordinates=[20, 20, 30, 30], format=BBoxFormat.CORNERS)  
        bbox3 = BBox(coordinates=[40, 40, 50, 50], format=BBoxFormat.CORNERS)   
        predictions = YoloOutput([(bbox1, 0, 0.9), (bbox2, 1, 0.8)])  
        ground_truth = [(bbox1, 0), (bbox2, 1), (bbox3, 1)]

        metrics = self.yolo.evaluate(predictions, ground_truth)
        self.assertAlmostEqual(metrics["precision"], 1.0)
        self.assertAlmostEqual(metrics["recall"], 2 / 3)
        self.assertAlmostEqual(metrics["f1"], 0.8)
        self.assertAlmostEqual(metrics["mAP"], 1.0)
        self.assertAlmostEqual(metrics["mAP@50-95"], 1.0)

    def test_evaluate_no_predictions(self):
        """Testa il caso in cui non ci sono previsioni."""
        bbox1 = BBox(coordinates=[0, 0, 10, 10], format=BBoxFormat.CORNERS)  
        bbox2 = BBox(coordinates=[20, 20, 30, 30], format=BBoxFormat.CORNERS)  
        predictions = YoloOutput([])  
        ground_truth = [(bbox1, 0), (bbox2, 1)]

        metrics = self.yolo.evaluate(predictions, ground_truth)
        self.assertAlmostEqual(metrics["precision"], 0.0)
        self.assertAlmostEqual(metrics["recall"], 0.0)
        self.assertAlmostEqual(metrics["f1"], 0.0)
        self.assertAlmostEqual(metrics["mAP"], 0.0)
        self.assertAlmostEqual(metrics["mAP@50-95"], 0.0)

    def test_evaluate_no_ground_truth(self):
        """Testa il caso in cui non ci sono verità fondamentali."""
        bbox1 = BBox(coordinates=[0, 0, 10, 10], format=BBoxFormat.CORNERS)   
        bbox2 = BBox(coordinates=[20, 20, 30, 30], format=BBoxFormat.CORNERS)  
        predictions = YoloOutput([(bbox1, 0, 0.9), (bbox2, 1, 0.8)])   
        ground_truth = []

        metrics = self.yolo.evaluate(predictions, ground_truth)
        self.assertAlmostEqual(metrics["precision"], 0.0)
        self.assertAlmostEqual(metrics["recall"], 0.0)
        self.assertAlmostEqual(metrics["f1"], 0.0)
        self.assertAlmostEqual(metrics["mAP"], 0.0)
        self.assertAlmostEqual(metrics["mAP@50-95"], 0.0)

    def test_evaluate_partial_overlap(self):
        """Testa il caso in cui le previsioni hanno una sovrapposizione parziale con le verità fondamentali."""
        bbox1 = BBox(coordinates=[0, 0, 10, 10], format=BBoxFormat.CORNERS)   
        bbox2 = BBox(coordinates=[1, 1, 11, 11], format=BBoxFormat.CORNERS)  
        predictions = YoloOutput([(bbox2, 0, 0.9)])  
        ground_truth = [(bbox1, 0)]

        metrics = self.yolo.evaluate(predictions, ground_truth, threshold_iou=0.5)
        self.assertAlmostEqual(metrics["precision"], 1.0)  # 1 TP / (1 TP + 0 FP)
        self.assertAlmostEqual(metrics["recall"], 1.0)     # 1 TP / (1 TP + 0 FN)
        self.assertAlmostEqual(metrics["f1"], 1.0)         # 2 * (precision * recall) / (precision + recall)
        self.assertAlmostEqual(metrics["mAP"], 1.0)        # Placeholder
        self.assertAlmostEqual(metrics["mAP@50-95"], 1.0)  # Placeholder
    
if __name__ == "__main__":
    unittest.main()