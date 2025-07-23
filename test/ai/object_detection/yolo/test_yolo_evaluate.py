import unittest
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict, Union
from saltup.ai.object_detection.yolo.yolo import BaseYolo, BBox, YoloOutput, BBoxFormat, evaluate, evaluate_image
from saltup.ai.object_detection.yolo.yolo_type import YoloType
from saltup.utils.data.image import ColorMode, ImageFormat, Image
from saltup.ai.base_dataformat.base_dataloader import BaseDataloader
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
            def __init__(self, model_path, number_class):
                # Initialize with required parameters including YoloType
                super().__init__(
                    yolot=YoloType.ULTRALYTICS,  # Added YoloType parameter
                    model=NeuralNetworkModel(model_path), 
                    number_class=number_class
                )
            
            def preprocess(self, 
                   image: Image,
                   target_height: int, 
                   target_width: int,        
                   normalize_method: callable = lambda x: x.astype(np.float32) / 255.0,
                   apply_padding: bool = True,
                   **kwargs: Any
                   ) -> np.ndarray:
                return np.random.random((target_height, target_width, 3)).astype(np.float32)

            def get_input_info(self) -> Tuple[tuple, ColorMode, ImageFormat]:
                return (
                    (224, 224, 3),  # Shape: (H, W, C)
                    ColorMode.RGB,
                    ImageFormat.HWC
                )
            
            def postprocess(self, raw_output: np.ndarray,
                    image_height: int, image_width: int, confidence_thr: float = 0.5, 
                            iou_threshold: float = 0.5) -> List[Tuple[BBox, int, float]]:
                return []
        
        self.yolo = MockYolo(self.onnx_model_path, number_class=2)

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
        if os.path.exists(self.tmp_dir):
            os.rmdir(self.tmp_dir)

    def test_evaluate_image_perfect_match(self):
        """Test the evaluate_image function with perfect matches."""
        bbox1 = BBox(img_height=100, img_width=100, coordinates=[0, 0, 10, 10], fmt=BBoxFormat.CORNERS_ABSOLUTE)  
        bbox2 = BBox(img_height=100, img_width=100, coordinates=[20, 20, 30, 30], fmt=BBoxFormat.CORNERS_ABSOLUTE)  
        predictions = YoloOutput([(bbox1, 0, 0.9), (bbox2, 1, 0.8)])   
        ground_truth = [(bbox1, 0), (bbox2, 1)]

        metrics = evaluate_image(predictions, ground_truth, iou_thres=0.5)
        
        # Check that we have metrics for both classes
        self.assertIn(0, metrics)
        self.assertIn(1, metrics)
        
        # Class 0: 1 TP, 0 FP, 0 FN
        self.assertEqual(metrics[0].getTP(), 1)
        self.assertEqual(metrics[0].getFP(), 0)
        self.assertEqual(metrics[0].getFN(), 0)
        
        # Class 1: 1 TP, 0 FP, 0 FN
        self.assertEqual(metrics[1].getTP(), 1)
        self.assertEqual(metrics[1].getFP(), 0)
        self.assertEqual(metrics[1].getFN(), 0)

    def test_evaluate_image_false_positives(self):
        """Test the evaluate_image function with false positives."""
        bbox1 = BBox(img_height=100, img_width=100, coordinates=[0, 0, 10, 10], fmt=BBoxFormat.CORNERS_ABSOLUTE)  
        bbox2 = BBox(img_height=100, img_width=100, coordinates=[20, 20, 30, 30], fmt=BBoxFormat.CORNERS_ABSOLUTE)  
        bbox3 = BBox(img_height=100, img_width=100, coordinates=[40, 40, 50, 50], fmt=BBoxFormat.CORNERS_ABSOLUTE)   
        predictions = YoloOutput([(bbox1, 0, 0.9), (bbox2, 1, 0.8), (bbox3, 1, 0.7)])   
        ground_truth = [(bbox1, 0), (bbox2, 1)]

        metrics = evaluate_image(predictions, ground_truth, iou_thres=0.5)
        
        # Class 0: 1 TP, 0 FP, 0 FN
        self.assertEqual(metrics[0].getTP(), 1)
        self.assertEqual(metrics[0].getFP(), 0)
        self.assertEqual(metrics[0].getFN(), 0)
        
        # Class 1: 1 TP, 1 FP, 0 FN (bbox3 is false positive)
        self.assertEqual(metrics[1].getTP(), 1)
        self.assertEqual(metrics[1].getFP(), 1)
        self.assertEqual(metrics[1].getFN(), 0)

    def test_evaluate_image_false_negatives(self):
        """Test the evaluate_image function with false negatives."""
        bbox1 = BBox(img_height=100, img_width=100, coordinates=[0, 0, 10, 10], fmt=BBoxFormat.CORNERS_ABSOLUTE)
        bbox2 = BBox(img_height=100, img_width=100, coordinates=[20, 20, 30, 30], fmt=BBoxFormat.CORNERS_ABSOLUTE)
        bbox3 = BBox(img_height=100, img_width=100, coordinates=[40, 40, 50, 50], fmt=BBoxFormat.CORNERS_ABSOLUTE)
        predictions = YoloOutput([(bbox1, 0, 0.9), (bbox2, 1, 0.8)])
        ground_truth = [(bbox1, 0), (bbox2, 1), (bbox3, 1)]

        metrics = evaluate_image(predictions, ground_truth, iou_thres=0.5)
        
        # Class 0: 1 TP, 0 FP, 0 FN
        self.assertEqual(metrics[0].getTP(), 1)
        self.assertEqual(metrics[0].getFP(), 0)
        self.assertEqual(metrics[0].getFN(), 0)
        
        # Class 1: 1 TP, 0 FP, 1 FN (bbox3 is false negative)
        self.assertEqual(metrics[1].getTP(), 1)
        self.assertEqual(metrics[1].getFP(), 0)
        self.assertEqual(metrics[1].getFN(), 1)

    def test_evaluate_image_no_predictions(self):
        """Test the evaluate_image function with no predictions."""
        bbox1 = BBox(img_height=100, img_width=100, coordinates=[0, 0, 10, 10], fmt=BBoxFormat.CORNERS_ABSOLUTE)  
        bbox2 = BBox(img_height=100, img_width=100, coordinates=[20, 20, 30, 30], fmt=BBoxFormat.CORNERS_ABSOLUTE)  
        predictions = YoloOutput([])  
        ground_truth = [(bbox1, 0), (bbox2, 1)]

        metrics = evaluate_image(predictions, ground_truth, iou_thres=0.5)
        
        # Should have metrics for both classes with all FN
        self.assertEqual(metrics[0].getTP(), 0)
        self.assertEqual(metrics[0].getFP(), 0)
        self.assertEqual(metrics[0].getFN(), 1)
        
        self.assertEqual(metrics[1].getTP(), 0)
        self.assertEqual(metrics[1].getFP(), 0)
        self.assertEqual(metrics[1].getFN(), 1)

    def test_evaluate_image_no_ground_truth(self):
        """Test the evaluate_image function with no ground truth."""
        bbox1 = BBox(img_height=100, img_width=100, coordinates=[0, 0, 10, 10], fmt=BBoxFormat.CORNERS_ABSOLUTE)   
        bbox2 = BBox(img_height=100, img_width=100, coordinates=[20, 20, 30, 30], fmt=BBoxFormat.CORNERS_ABSOLUTE)  
        predictions = YoloOutput([(bbox1, 0, 0.9), (bbox2, 1, 0.8)])   
        ground_truth = []

        metrics = evaluate_image(predictions, ground_truth, iou_thres=0.5)
        
        # Should have metrics for both classes with all FP
        self.assertEqual(metrics[0].getTP(), 0)
        self.assertEqual(metrics[0].getFP(), 1)
        self.assertEqual(metrics[0].getFN(), 0)
        
        self.assertEqual(metrics[1].getTP(), 0)
        self.assertEqual(metrics[1].getFP(), 1)
        self.assertEqual(metrics[1].getFN(), 0)

    def test_evaluate_image_partial_overlap(self):
        """Test the evaluate_image function with partial overlap."""
        bbox1 = BBox(img_height=100, img_width=100, coordinates=[0, 0, 10, 10], fmt=BBoxFormat.CORNERS_ABSOLUTE)
        bbox2 = BBox(img_height=100, img_width=100, coordinates=[1, 1, 11, 11], fmt=BBoxFormat.CORNERS_ABSOLUTE)
        predictions = YoloOutput([(bbox2, 0, 0.9)])
        ground_truth = [(bbox1, 0)]

        # Test with high IoU threshold (should not match)
        metrics_high = evaluate_image(predictions, ground_truth, iou_thres=0.9)
        self.assertEqual(metrics_high[0].getTP(), 0)
        self.assertEqual(metrics_high[0].getFP(), 1)
        self.assertEqual(metrics_high[0].getFN(), 1)
        
        # Test with low IoU threshold (should match)
        metrics_low = evaluate_image(predictions, ground_truth, iou_thres=0.3)
        self.assertEqual(metrics_low[0].getTP(), 1)
        self.assertEqual(metrics_low[0].getFP(), 0)
        self.assertEqual(metrics_low[0].getFN(), 0)

    def test_evaluate_with_dataloader(self):
        """Test the evaluate function with a mock dataloader."""
        
        # Create mock dataloader
        class MockDataloader(BaseDataloader):
            def __init__(self, data):
                self.data = data
                self.index = 0
                
            def __iter__(self):
                self.index = 0
                return self
                
            def __next__(self):
                if self.index >= len(self.data):
                    raise StopIteration
                item = self.data[self.index]
                self.index += 1
                return item
                
            def __len__(self):
                return len(self.data)
                
            def split(self, ratio):
                # Simple implementation for testing
                return [self, self]
                
            @staticmethod
            def merge(dl1, dl2):
                # Simple implementation for testing
                return dl1
        
        # Create mock YOLO model that returns predictable results
        class MockEvaluationYolo(BaseYolo):
            def __init__(self, predictions_per_image):
                # Initialize with minimal required parameters including YoloType
                model_path = self.create_temp_model()
                super().__init__(
                    yolot=YoloType.ULTRALYTICS,  # Added YoloType parameter
                    model=NeuralNetworkModel(model_path), 
                    number_class=2
                )
                self.predictions_per_image = predictions_per_image
                self.current_image_idx = 0
                
            def create_temp_model(self):
                # Create temporary ONNX model for testing
                tmp_dir = tempfile.mkdtemp()
                model_path = os.path.join(tmp_dir, "temp_model.onnx")
                input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 224, 224])
                output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 224, 224])
                node = helper.make_node('Identity', ['input'], ['output'])
                graph = helper.make_graph([node], 'simple_model', [input], [output])
                model = helper.make_model(graph, producer_name='onnx-test', opset_imports=[helper.make_opsetid("", 16)])
                onnx.save(model, model_path)
                return model_path
                
            def run(self, image, confidence_thr=0.5, iou_threshold=0.5, preprocess=None, postprocess=None):
                # Return predefined predictions for each image
                predictions = self.predictions_per_image[self.current_image_idx % len(self.predictions_per_image)]
                self.current_image_idx += 1
                return YoloOutput(predictions)
                
            def get_input_info(self):
                return ((100, 100, 3), ColorMode.RGB, ImageFormat.HWC)
                
            def preprocess(self, image, target_height, target_width, **kwargs):
                return np.zeros((target_height, target_width, 3))
                
            def postprocess(self, raw_output, image_height, image_width, confidence_thr=0.5, iou_threshold=0.5):
                return []
        
        # Test data: 2 images with ground truth and predictions
        # Image 1: 1 GT, 1 correct prediction
        bbox1_gt = BBox(img_height=100, img_width=100, coordinates=[10, 10, 30, 30], fmt=BBoxFormat.CORNERS_ABSOLUTE)
        bbox1_pred = BBox(img_height=100, img_width=100, coordinates=[12, 12, 28, 28], fmt=BBoxFormat.CORNERS_ABSOLUTE)
        
        # Image 2: 1 GT, 1 correct prediction + 1 false positive
        bbox2_gt = BBox(img_height=100, img_width=100, coordinates=[40, 40, 60, 60], fmt=BBoxFormat.CORNERS_ABSOLUTE)
        bbox2_pred1 = BBox(img_height=100, img_width=100, coordinates=[41, 41, 59, 59], fmt=BBoxFormat.CORNERS_ABSOLUTE)
        bbox2_pred2 = BBox(img_height=100, img_width=100, coordinates=[70, 70, 90, 90], fmt=BBoxFormat.CORNERS_ABSOLUTE)  # False positive
        
        # Create mock image (doesn't matter for this test)
        mock_image = Image(np.zeros((100, 100, 3), dtype=np.uint8), ColorMode.RGB)
        
        # Create dataloader data: (image, ground_truth_labels)
        dataloader_data = [
            (mock_image, [(bbox1_gt, 0)]),  # Image 1: class 0
            (mock_image, [(bbox2_gt, 1)])   # Image 2: class 1
        ]
        
        # Create predictions for each image
        predictions_per_image = [
            [(bbox1_pred, 0, 0.9)],                                    # Image 1: 1 correct prediction
            [(bbox2_pred1, 1, 0.85), (bbox2_pred2, 1, 0.7)]          # Image 2: 1 correct + 1 false positive
        ]
        
        dataloader = MockDataloader(dataloader_data)
        yolo = MockEvaluationYolo(predictions_per_image)
        
        # Test evaluate function
        metrics_per_class, overall_metric, overall_map_50_95 = evaluate(
            yolo=yolo,
            dataloader=dataloader,
            iou_threshold=0.5,
            confidence_threshold=0.5
        )
        
        # Verify results
        # Should have metrics for classes 0 and 1
        self.assertIn(0, metrics_per_class)
        self.assertIn(1, metrics_per_class)
        
        # Class 0: 1 TP, 0 FP, 0 FN
        self.assertEqual(metrics_per_class[0].getTP(), 1)
        self.assertEqual(metrics_per_class[0].getFP(), 0)
        self.assertEqual(metrics_per_class[0].getFN(), 0)
        
        # Class 1: 1 TP, 1 FP, 0 FN  
        self.assertEqual(metrics_per_class[1].getTP(), 1)
        self.assertEqual(metrics_per_class[1].getFP(), 1)
        self.assertEqual(metrics_per_class[1].getFN(), 0)
        
        # Overall metrics: 2 TP, 1 FP, 0 FN
        self.assertEqual(overall_metric.getTP(), 2)
        self.assertEqual(overall_metric.getFP(), 1)
        self.assertEqual(overall_metric.getFN(), 0)
        
        # Overall precision should be 2/3 ≈ 0.667
        self.assertAlmostEqual(overall_metric.getPrecision(), 2/3, places=3)
        
        # Overall recall should be 2/2 = 1.0
        self.assertAlmostEqual(overall_metric.getRecall(), 1.0, places=3)
        
        # F1 should be 2 * (2/3 * 1) / (2/3 + 1) = 4/5 = 0.8
        expected_f1 = 2 * (2/3 * 1) / (2/3 + 1)
        self.assertAlmostEqual(overall_metric.getF1Score(), expected_f1, places=3)
        
        # mAP@50-95 should be a reasonable value (0-1)
        self.assertGreaterEqual(overall_map_50_95, 0.0)
        self.assertLessEqual(overall_map_50_95, 1.0)
        
        print(f"Test completed successfully:")
        print(f"  Overall precision: {overall_metric.getPrecision():.3f}")
        print(f"  Overall recall: {overall_metric.getRecall():.3f}")
        print(f"  Overall F1: {overall_metric.getF1Score():.3f}")
        print(f"  Overall mAP@50-95: {overall_map_50_95:.3f}")

    def test_evaluate_with_empty_dataloader(self):
        """Test the evaluate function with an empty dataloader."""
        
        class EmptyDataloader(BaseDataloader):
            def __iter__(self):
                return iter([])
                
            def __next__(self):
                raise StopIteration
                
            def __len__(self):
                return 0
                
            def split(self, ratio):
                return [self, self]
                
            @staticmethod
            def merge(dl1, dl2):
                return dl1
        
        class MockEmptyYolo(BaseYolo):
            def __init__(self):
                # Create temporary ONNX model for testing
                tmp_dir = tempfile.mkdtemp()
                model_path = os.path.join(tmp_dir, "temp_model.onnx")
                input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 224, 224])
                output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 224, 224])
                node = helper.make_node('Identity', ['input'], ['output'])
                graph = helper.make_graph([node], 'simple_model', [input], [output])
                model = helper.make_model(graph, producer_name='onnx-test', opset_imports=[helper.make_opsetid("", 16)])
                onnx.save(model, model_path)
                
                # Initialize with required parameters including YoloType
                super().__init__(
                    yolot=YoloType.ULTRALYTICS,  # Added YoloType parameter
                    model=NeuralNetworkModel(model_path), 
                    number_class=2
                )
                
            def run(self, image, confidence_thr=0.5, iou_threshold=0.5, preprocess=None, postprocess=None):
                return YoloOutput([])
                
            def get_input_info(self):
                return ((100, 100, 3), ColorMode.RGB, ImageFormat.HWC)
                
            def preprocess(self, image, target_height, target_width, **kwargs):
                return np.zeros((target_height, target_width, 3))
                
            def postprocess(self, raw_output, image_height, image_width, confidence_thr=0.5, iou_threshold=0.5):
                return []
        
        dataloader = EmptyDataloader()
        yolo = MockEmptyYolo()
        
        # Test evaluate function with empty dataloader
        metrics_per_class, overall_metric, overall_map_50_95 = evaluate(
            yolo=yolo,
            dataloader=dataloader,
            iou_threshold=0.5,
            confidence_threshold=0.5
        )
        
        # Should have initialized metrics for all classes but with zero values
        self.assertEqual(len(metrics_per_class), 2)
        for class_id in range(2):
            self.assertEqual(metrics_per_class[class_id].getTP(), 0)
            self.assertEqual(metrics_per_class[class_id].getFP(), 0)
            self.assertEqual(metrics_per_class[class_id].getFN(), 0)
        
        # Overall metrics should be zero
        self.assertEqual(overall_metric.getTP(), 0)
        self.assertEqual(overall_metric.getFP(), 0)
        self.assertEqual(overall_metric.getFN(), 0)
        
        # mAP@50-95 should be 0 for empty dataset
        self.assertEqual(overall_map_50_95, 0.0)

    def test_evaluate_with_output_streams(self):
        """Test the evaluate function with output streams."""
        import io
        
        # Create a string buffer to capture output
        output_buffer = io.StringIO()
        
        # Create minimal test setup
        class SimpleDataloader(BaseDataloader):
            def __init__(self):
                self.data = []
                bbox_gt = BBox(img_height=100, img_width=100, coordinates=[10, 10, 30, 30], fmt=BBoxFormat.CORNERS_ABSOLUTE)
                mock_image = Image(np.zeros((100, 100, 3), dtype=np.uint8), ColorMode.RGB)
                self.data.append((mock_image, [(bbox_gt, 0)]))
                self.index = 0

            def __iter__(self):
                self.index = 0
                return self

            def __next__(self):
                if self.index >= len(self.data):
                    raise StopIteration
                item = self.data[self.index]
                self.index += 1
                return item

            def __len__(self):
                return len(self.data)

            def split(self, ratio):
                # For testing, just return two copies
                return [self, self]

            @staticmethod
            def merge(dl1, dl2):
                # For testing, just return dl1
                return dl1
        
        
        class SimpleYolo(BaseYolo):
            def __init__(self):
                tmp_dir = tempfile.mkdtemp()
                model_path = os.path.join(tmp_dir, "temp_model.onnx")
                input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 224, 224])
                output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 224, 224])
                node = helper.make_node('Identity', ['input'], ['output'])
                graph = helper.make_graph([node], 'simple_model', [input], [output])
                model = helper.make_model(graph, producer_name='onnx-test', opset_imports=[helper.make_opsetid("", 16)])
                onnx.save(model, model_path)
                
                super().__init__(
                    yolot=YoloType.ULTRALYTICS,
                    model=NeuralNetworkModel(model_path), 
                    number_class=2
                )
                
            def run(self, image, confidence_thr=0.5, iou_threshold=0.5, preprocess=None, postprocess=None):
                # Return a matching prediction
                bbox_pred = BBox(img_height=100, img_width=100, coordinates=[12, 12, 28, 28], fmt=BBoxFormat.CORNERS_ABSOLUTE)
                return YoloOutput([(bbox_pred, 0, 0.9)])
                
            def get_input_info(self):
                return ((100, 100, 3), ColorMode.RGB, ImageFormat.HWC)
                
            def preprocess(self, image, target_height, target_width, **kwargs):
                return np.zeros((target_height, target_width, 3))
                
            def postprocess(self, raw_output, image_height, image_width, confidence_thr=0.5, iou_threshold=0.5):
                return []
        
        dataloader = SimpleDataloader()
        yolo = SimpleYolo()
        
        # Test evaluate function with output stream
        metrics_per_class, overall_metric, overall_map_50_95 = evaluate(
            yolo=yolo,
            dataloader=dataloader,
            iou_threshold=0.5,
            confidence_threshold=0.5,
            output_streams=[output_buffer]
        )
        
        # Check that output was written to the buffer
        output_content = output_buffer.getvalue()
        self.assertIn("Model type:", output_content)
        self.assertIn("Overall Precision:", output_content)
        self.assertIn("Overall Recall:", output_content)
        self.assertIn("Overall F1 Score:", output_content)
        self.assertIn("Overall mAP@50-95", output_content)
        
        # Verify metrics are correct (should be perfect match)
        self.assertEqual(overall_metric.getTP(), 1)
        self.assertEqual(overall_metric.getFP(), 0)
        self.assertEqual(overall_metric.getFN(), 0)
        self.assertAlmostEqual(overall_metric.getPrecision(), 1.0)
        self.assertAlmostEqual(overall_metric.getRecall(), 1.0)
        

if __name__ == "__main__":
    unittest.main()