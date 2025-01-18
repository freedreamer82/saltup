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
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision


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
        self.assertAlmostEqual(metrics["mAP"], 0.875)
        self.assertAlmostEqual(metrics["mAP@50-95"], 0.833, delta=0.0005)

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
        self.assertAlmostEqual(metrics["precision"], 1.0)   
        self.assertAlmostEqual(metrics["recall"], 1.0)      
        self.assertAlmostEqual(metrics["f1"], 1.0)         
        self.assertAlmostEqual(metrics["mAP"], 1.0)         
        self.assertAlmostEqual(metrics["mAP@50-95"], 0.4, delta=0.1)

   

    def test_evaluate_comparison_with_torchmetrics(self):
        """
        Confronta i risultati della nostra implementazione con quella di torchmetrics.
        Verifica che entrambe le implementazioni producano risultati simili.
        """
        # Crea i dati di test
        bbox1 = BBox(coordinates=[0, 0, 10, 10], format=BBoxFormat.CORNERS)
        bbox2 = BBox(coordinates=[20, 20, 30, 30], format=BBoxFormat.CORNERS)
        bbox3 = BBox(coordinates=[40, 40, 50, 50], format=BBoxFormat.CORNERS)
        
        predictions = YoloOutput([
            (bbox1, 0, 0.9),
            (bbox2, 1, 0.8),
            (bbox3, 0, 0.7)
        ])
        
        ground_truth = [(bbox1, 0), (bbox2, 1), (bbox3, 0)]
        
        # Calcola le metriche con la nostra implementazione
        our_metrics = self.yolo.evaluate(predictions, ground_truth)
        
        # Prepara i dati per torchmetrics
        pred_boxes = []
        pred_scores = []
        pred_labels = []
        
        # Accedi ai dati di YoloOutput usando get_boxes()
        for box, cls, score in predictions.get_boxes():
            pred_boxes.append([
                box.coordinates[0],
                box.coordinates[1],
                box.coordinates[2],
                box.coordinates[3]
            ])
            pred_scores.append(score)
            pred_labels.append(cls + 1)  # torchmetrics usa 1-based
        
        gt_boxes = []
        gt_labels = []
        
        for box, cls in ground_truth:
            gt_boxes.append([
                box.coordinates[0],
                box.coordinates[1],
                box.coordinates[2],
                box.coordinates[3]
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
        
        # Calcola le metriche con torchmetrics
        metric = MeanAveragePrecision(box_format='xyxy')
        metric.update(preds, target)
        torch_metrics = metric.compute()
        
        # Confronta i risultati delle due implementazioni
        self.assertAlmostEqual(
            our_metrics["mAP"],
            torch_metrics['map'].item(),
            places=2,
            msg="Le metriche mAP differiscono significativamente tra le implementazioni"
        )
        
        self.assertAlmostEqual(
            our_metrics["mAP@50-95"],
            torch_metrics['map_50'].item(),
            places=2,
            msg="Le metriche mAP@50-95 differiscono significativamente tra le implementazioni"
        )
        
        # Verifica anche le altre metriche della nostra implementazione
        self.assertGreater(our_metrics["precision"], 0.8)
        self.assertGreater(our_metrics["recall"], 0.8)
        self.assertGreater(our_metrics["f1"], 0.8)
            
if __name__ == "__main__":
    unittest.main()