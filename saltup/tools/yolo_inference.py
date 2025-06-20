#!/usr/bin/env python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable OneDNN optimizations
import time
import argparse
from argparse import Namespace
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict, Union
from tqdm import tqdm

from saltup.ai.object_detection.yolo.yolo import BaseYolo, YoloOutput
from saltup.ai.object_detection.yolo.yolo_type import YoloType
from saltup.ai.object_detection.yolo.yolo_factory import YoloFactory
from saltup.ai.object_detection.utils.bbox import BBox, draw_boxes_on_image_with_labels_score, BBoxFormat , print_bbox_info
from saltup.utils.data.image.image_utils import ColorMode, ColorsBGR
from saltup.ai.object_detection.utils.metrics import Metric
from saltup.utils.data.image.image_utils import generate_random_bgr_colors
import signal
import sys

def signal_handler(sig, frame):
    """Handle Ctrl+C signal to gracefully exit the program."""
    print("\nProgram interrupted with Ctrl + C!")
    sys.exit(0)  # Terminate the script
 

def robust_mean(times):
    """Calculate the robust mean by excluding the smallest and largest values."""
    if len(times) <= 2:
        return np.mean(times) if times else 0
    times_sorted = sorted(times)
    return np.mean(times_sorted[1:-1])

def process_image(yolo: BaseYolo, image_path: str, args: Namespace):
    """
    Process an image using the YOLO model.
    
    Args:
        yolo: The YOLO model instance.
        image_path: Path to the input image.
        args: Command-line arguments.
    
    Returns:
        boxes: Detected bounding boxes.
        class_ids: Class IDs of the detected objects.
    """
    image = BaseYolo.load_image(image_path, ColorMode.GRAY if yolo.get_number_image_channel() == 1 else ColorMode.RGB)
    img_height, img_width, _ = image.get_shape()
    
    yoloOut = yolo.run(image, img_height, img_width, args.conf_thres, args.iou_thres)
    bboxes_with_labels_score = yoloOut.get_boxes()
    
    if args.gui:
        image_with_boxes = draw_boxes_on_image_with_labels_score(image, bboxes_with_labels_score)
        image_with_boxes.show()    
    
    boxes = [bbox for bbox, _, _ in bboxes_with_labels_score]
    class_ids = [class_id for _, class_id, _ in bboxes_with_labels_score]
    
    return boxes, class_ids

# def evaluate_predictions(predictions: YoloOutput, ground_truth: List[Tuple[BBox, int]], iou_thres: float) -> Dict[str, float]:
#     """Evaluate predictions against ground truth and compute metrics.
    
#     Args:
#         predictions: YOLO model predictions.
#         ground_truth: Ground truth bounding boxes and class IDs.
#         iou_thres: IoU threshold for matching predictions to ground truth.
    
#     Returns:
#         Dictionary containing evaluation metrics (TP, FP, FN, precision, recall, F1-score).
#     """
#     # Extract predicted boxes, class IDs, and scores
#     pred_boxes_with_info = predictions.get_boxes()
#     pred_boxes = [bbox for bbox, _, _ in pred_boxes_with_info]
#     pred_classes = [class_id for _, class_id, _ in pred_boxes_with_info]

#     # Initialize counters
#     tp = 0  # True Positives
#     fp = 0  # False Positives
#     fn = 0  # False Negatives

#     # Match predictions to ground truth
#     matched_gt = [False] * len(ground_truth)  # Track which ground truth boxes have been matched

#     for pred_box, pred_class in zip(pred_boxes, pred_classes):
#         best_iou = 0
#         best_match_idx = -1

#         # Find the best matching ground truth box
#         for j, (gt_box, gt_class) in enumerate(ground_truth):
#             if matched_gt[j]:
#                 continue  # Skip already matched ground truth boxes

#             iou = pred_box.compute_iou(gt_box)
#             if iou > best_iou:
#                 best_iou = iou
#                 best_match_idx = j

#         # Determine if the prediction is a TP or FP
#         if best_iou >= iou_thres and pred_class == ground_truth[best_match_idx][1]:
#             tp += 1  # True Positive
#             matched_gt[best_match_idx] = True
#         else:
#             fp += 1  # False Positive

#     # Count False Negatives (unmatched ground truth boxes)
#     fn = sum(not matched for matched in matched_gt)

#     # Calculate precision, recall, and F1-score
#     precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
#     recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
#     f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

#     return {
#         "tp": tp,
#         "fp": fp,
#         "fn": fn,
#         "precision": precision,
#         "recall": recall,
#         "f1_score": f1_score,
#     }
    

def evaluate_predictions(predictions: YoloOutput, ground_truth: List[Tuple[BBox, int]], iou_thres: float) -> Dict[str, Union[float, int, Dict[int, Dict[str, float]]]]:
    """Evaluate predictions against ground truth and compute overall and per-class metrics.
    
    Args:
        predictions: YOLO model predictions.
        ground_truth: Ground truth bounding boxes and class IDs.
        iou_thres: IoU threshold for matching predictions to ground truth.
    
    Returns:
        Dictionary containing overall metrics and a nested dictionary with per-class metrics.
    """
    pred_boxes_with_info = predictions.get_boxes()
    pred_boxes = [bbox for bbox, _, _ in pred_boxes_with_info]
    pred_classes = [class_id for _, class_id, _ in pred_boxes_with_info]

    # Initialize counters
    tp = 0
    fp = 0
    fn = 0

    matched_gt = [False] * len(ground_truth)

    # Initialize per-class stats
    per_class_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})

    for pred_box, pred_class in zip(pred_boxes, pred_classes):
        best_iou = 0
        best_match_idx = -1

        for j, (gt_box, gt_class) in enumerate(ground_truth):
            if matched_gt[j]:
                continue

            iou = pred_box.compute_iou(gt_box)
            if iou > best_iou:
                best_iou = iou
                best_match_idx = j

        if best_iou >= iou_thres and pred_class == ground_truth[best_match_idx][1]:
            tp += 1
            matched_gt[best_match_idx] = True
            per_class_stats[pred_class]['tp'] += 1
        else:
            fp += 1
            per_class_stats[pred_class]['fp'] += 1

    # Count False Negatives
    for (gt_box, gt_class), matched in zip(ground_truth, matched_gt):
        if not matched:
            fn += 1
            per_class_stats[gt_class]['fn'] += 1

    # Calculate overall metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Calculate per-class precision, recall, F1
    per_class_metrics = {}
    for class_id, stats in per_class_stats.items():
        c_tp = stats['tp']
        c_fp = stats['fp']
        c_fn = stats['fn']

        c_precision = c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 0.0
        c_recall = c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 0.0
        c_f1 = 2 * (c_precision * c_recall) / (c_precision + c_recall) if (c_precision + c_recall) > 0 else 0.0

        per_class_metrics[class_id] = {
            "tp": c_tp,
            "fp": c_fp,
            "fn": c_fn,
            "precision": c_precision,
            "recall": c_recall,
            "f1_score": c_f1
        }

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "per_class": per_class_metrics
    }


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Input NN model.")
    parser.add_argument("--type", type=str, help="Input your YOLO model type.")
    parser.add_argument("--img", type=str, default="bus.jpg", help="Path to input image or folder.")
    parser.add_argument("--anchors", type=str, default="", help="Path to the anchors if needed.")
    parser.add_argument("--label", type=str, help="Path to image label or folder.")
    parser.add_argument("--gui", action='store_true', help="Open GUI to draw bounding boxes.")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="Confidence threshold.")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="NMS IoU threshold.")
    parser.add_argument("--num_class", type=int, help="Number of the classes.")
    parser.add_argument("--show_bbox", type=str, help="Print bounding box information.")
    parser.add_argument("--max_images", type=int, default=0, help="Maximum number of images to process.")
    parser.add_argument("--cls_name", type=str, default="", help="Comma-separated list of class names.")

        # Analizza gli argomenti
    args = parser.parse_args(args)
    
    signal.signal(signal.SIGINT, signal_handler)


    if args.cls_name:
        class_labels = args.cls_name.split(',')
    else:
        class_labels = []

    yolotype = YoloType.from_string(args.type)
    if args.anchors:
        yolo = YoloFactory.create(yolotype, args.model, args.num_class, anchors=args.anchors)
    else:
        yolo = YoloFactory.create(yolotype, args.model, args.num_class)
    
    if os.path.isdir(args.img):
        image_paths = [
            os.path.join(args.img, fname) 
            for fname in os.listdir(args.img) 
            if fname.lower().endswith(('.jpg', '.png', '.jpeg'))
        ]
    else:
        image_paths = [args.img]
    
    if args.max_images > 0:
        image_paths = image_paths[:args.max_images]
    
    all_predictions = []
    all_labels = []
    metric = Metric()
    
    if args.cls_name:
        dict_class_metric = {}
        for i, class_label in enumerate(class_labels):
            dict_class_metric[i] = Metric()
    else:
        dict_class_metric = {i: Metric() for i in range(args.num_class)}
    
    start_time = time.time()
    num_images = len(image_paths)

    preprocess_times = []
    inference_times = []
    postprocess_times = []

    pbar = tqdm(image_paths, desc="Processing images", position=0, leave=True, dynamic_ncols=True)
    for i, image_path in enumerate(pbar):
        try:
            # Process the image
            image = BaseYolo.load_image(image_path, ColorMode.GRAY if yolo.get_number_image_channel() == 1 else ColorMode.RGB)
            
            yoloOut = yolo.run(image, args.conf_thres, args.iou_thres)
                
            preprocess_times.append(yoloOut.get_preprocessing_time())
            inference_times.append(yoloOut.get_inference_time())
            postprocess_times.append(yoloOut.get_postprocessing_time())

            boxes_with_info = yoloOut.get_boxes()
            boxes = [bbox for bbox, _, _ in boxes_with_info]
            class_ids = [class_id for _, class_id, _ in boxes_with_info]
            scores = [score for _, _, score in boxes_with_info]
            
            all_predictions.append((boxes, class_ids))
            
            num_classes = len(args.cls_name.split(',')) if args.cls_name else max(class_ids) + 1 if class_ids else 1
            class_colors = generate_random_bgr_colors(num_classes)
            
            class_colors_dict = {i: color for i, color in enumerate(class_colors)}
            
            if args.cls_name:
                class_labels_dict = {i: label for i, label in enumerate(args.cls_name.split(','))}
            else:
                class_labels_dict = {i: f"class_{i}" for i in range(num_classes)}

            if args.label:
                label_path = os.path.join(
                    args.label, 
                    os.path.basename(image_path).replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt')
                )
                if os.path.exists(label_path):
                    yolo_bboxes, yolo_class_ids = BBox.from_yolo_file(label_path, img_width=image.get_width(), 
                                                                      img_height=image.get_height())
                    all_labels.append((yolo_bboxes, yolo_class_ids))
                    
                    # Evaluate predictions for the current image
                    metrics = evaluate_predictions(yoloOut, list(zip(yolo_bboxes, yolo_class_ids)), args.iou_thres)
                    
                    for class_id in metrics['per_class'].keys():
                        
                        dict_class_metric[class_id].addTP(metrics['per_class'][class_id]['tp'])
                        dict_class_metric[class_id].addFP(metrics['per_class'][class_id]['fp'])
                        dict_class_metric[class_id].addFN(metrics['per_class'][class_id]['fn'])
                    
                    # Update cumulative metrics
                    metric.addTP(metrics['tp'])
                    metric.addFP(metrics['fp'])
                    metric.addFN(metrics['fn'])
                    
                    # Update progress bar with current metrics
                    pbar.set_postfix(**metric.get_metrics())
            else:
                print(f"Image {i+1}:{image_path}")
                print_bbox_info(boxes, class_ids, scores, class_labels_dict)

            if args.gui:               
                # Draw bounding boxes on the image
                image_with_boxes = draw_boxes_on_image_with_labels_score(
                    image, 
                    yoloOut.get_boxes(),
                    class_colors_bgr=class_colors_dict,
                    class_labels=class_labels_dict
                )
                
                image_with_boxes.show()
     
        except Exception as e:
            tqdm.write(f"Error processing image {image_path}: {e}")
            continue
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_image = total_time * 1000 / len(image_paths) if len(image_paths) > 0 else 0

    avg_preprocess_time = robust_mean(preprocess_times)
    avg_inference_time = robust_mean(inference_times)
    avg_postprocess_time = robust_mean(postprocess_times)
 
    print("\n")
    print("="*80)
    print(f"{'INFERENCE SUMMARY':^80}")
    print("="*80)
    print("\n")
    print(f"Model path: {args.model}")
    print(f"Model type: {yolotype.name}")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Average time per image: {avg_time_per_image:.2f} ms")
    num_params = yolo.get_model().get_num_parameters()
    if num_params >= 1_000_000:
        num_params_str = f"{num_params:,} ({num_params/1_000_000:.1f}M)"
    elif num_params >= 1_000:
        num_params_str = f"{num_params:,} ({num_params/1_000:.1f}k)"
    else:
        num_params_str = f"{num_params}"
    print(f"Number of model parameters : {num_params_str}")
    print(f"Size of model : { yolo.get_model().get_model_size_bytes() / (1024 * 1024):.2f} MB")


    print("Timings (per image, robust mean):")
    print(f"  - {'Pre-processing:':<25} {avg_preprocess_time:.2f} ms")
    print(f"  - {'Inference:':<25} {avg_inference_time:.2f} ms")
    print(f"  - {'Post-processing:':<25} {avg_postprocess_time:.2f} ms")
    
    if args.label:
        print("\n")
        print("="*80)
        print(f"{'METRICS SUMMARY':^80}")
        print("="*80)
        print("\n")
        print(f"{'Images processed:':<20} {num_images}")
        print(f"\nPer class:")
        print("+"*80)
        if args.cls_name:
            for class_id, class_label in enumerate(class_labels):
                if class_id == 0:
                    print(f"     {'label':<12} | {'Precision':<12} {'':>12} {'Recall':<12} {'':>6}{'F1 Score':<12}")
                    print("+"*80)
                print(f"  {class_label:<12} | {dict_class_metric[class_id].getPrecision():.4f} {'':<12}| {dict_class_metric[class_id].getRecall():.4f} {'':<12}| {dict_class_metric[class_id].getF1Score():.4f} {'':<12}")
                print("-"*80)
        else:
            for class_id in range(args.num_class):
                if class_id == 0:
                    print(f"     {'id':<12} | {'Precision':<12} {'':>12} {'Recall':<12} {'':>6}{'F1 Score':<12}")
                    print("+"*80)
                print(f"  {class_id:<12} | {dict_class_metric[class_id].getPrecision():.4f} {'':<12}| {dict_class_metric[class_id].getRecall():.4f} {'':<12}| {dict_class_metric[class_id].getF1Score():.4f}")
                print("-"*80)
        
        print("\nOverall:")
        print(f"  - {'True Positives (TP):':<25} {metric.getTP()}")
        print(f"  - {'False Positives (FP):':<25} {metric.getFP()}")
        print(f"  - {'False Negatives (FN):':<25} {metric.getFN()}")
        print(f"  - {'Overall Precision:':<25} {metric.getPrecision():.4f}")
        print(f"  - {'Overall Recall:':<25} {metric.getRecall():.4f}")
        print(f"  - {'Overall F1 Score:':<25} {metric.getF1Score():.4f}")
        print("="*80)
    
if __name__ == "__main__":
    main()