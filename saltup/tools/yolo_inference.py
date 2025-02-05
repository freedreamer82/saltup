#!/usr/bin/env python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable OneDNN optimizations
import time
import argparse
import numpy as np
from typing import List, Tuple, Dict
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

def process_image(yolo, image_path, args):
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

def evaluate_predictions(predictions: YoloOutput, ground_truth: List[Tuple[BBox, int]], iou_thres: float) -> Dict[str, float]:
    """Evaluate predictions against ground truth and compute metrics.
    
    Args:
        predictions: YOLO model predictions.
        ground_truth: Ground truth bounding boxes and class IDs.
        iou_thres: IoU threshold for matching predictions to ground truth.
    
    Returns:
        Dictionary containing evaluation metrics (TP, FP, FN, precision, recall, F1-score).
    """
    # Extract predicted boxes, class IDs, and scores
    pred_boxes_with_info = predictions.get_boxes()
    pred_boxes = [bbox for bbox, _, _ in pred_boxes_with_info]
    pred_classes = [class_id for _, class_id, _ in pred_boxes_with_info]

    # Initialize counters
    tp = 0  # True Positives
    fp = 0  # False Positives
    fn = 0  # False Negatives

    # Match predictions to ground truth
    matched_gt = [False] * len(ground_truth)  # Track which ground truth boxes have been matched

    for pred_box, pred_class in zip(pred_boxes, pred_classes):
        best_iou = 0
        best_match_idx = -1

        # Find the best matching ground truth box
        for j, (gt_box, gt_class) in enumerate(ground_truth):
            if matched_gt[j]:
                continue  # Skip already matched ground truth boxes

            iou = pred_box.compute_iou(gt_box)
            if iou > best_iou:
                best_iou = iou
                best_match_idx = j

        # Determine if the prediction is a TP or FP
        if best_iou >= iou_thres and pred_class == ground_truth[best_match_idx][1]:
            tp += 1  # True Positive
            matched_gt[best_match_idx] = True
        else:
            fp += 1  # False Positive

    # Count False Negatives (unmatched ground truth boxes)
    fn = sum(not matched for matched in matched_gt)

    # Calculate precision, recall, and F1-score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
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
    
    start_time = time.time()
    num_images = len(image_paths)

    preprocess_times = []
    inference_times = []
    postprocess_times = []

    pbar = tqdm(image_paths, desc="Processing images", position=0, leave=True, dynamic_ncols=True)
    for image_path in pbar:
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
                    
                    # Update cumulative metrics
                    metric.addTP(metrics['tp'])
                    metric.addFP(metrics['fp'])
                    metric.addFN(metrics['fn'])
                    
                    # Update progress bar with current metrics
                    pbar.set_postfix(**metric.get_metrics())
            else:
                print_bbox_info(boxes, class_ids, scores, class_labels_dict)

            if args.gui:               
                # Draw bounding boxes on the image
                image_with_boxes = draw_boxes_on_image_with_labels_score(image, 
                                                                        yoloOut.get_boxes(format=BBoxFormat.CORNERS),
                                                                        class_colors_bgr=class_colors_dict,
                                                                        class_labels=class_labels_dict)
                
                image_with_boxes.show()
     
        except Exception as e:
            tqdm.write(f"Error processing image {image_path}: {e}")
            continue
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_image = total_time / len(image_paths) if len(image_paths) > 0 else 0

    avg_preprocess_time = robust_mean(preprocess_times)
    avg_inference_time = robust_mean(inference_times)
    avg_postprocess_time = robust_mean(postprocess_times)
 
    print("\n--- Summary ---")
    print(f"Model path: {args.model}")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Average time per image: {avg_time_per_image:.2f} seconds")

    print("Timings (per image, robust mean):")
    print(f"  - Pre-processing: {avg_preprocess_time:.2f} ms")
    print(f"  - Inference: {avg_inference_time:.2f} ms")
    print(f"  - Post-processing: {avg_postprocess_time:.2f} ms")

    print("Metrics:")
    print(f"  - Images processed: {num_images}")
    print(f"  - TP: {metric.getTP()}")
    print(f"  - FP: {metric.getFP()}")
    print(f"  - FN: {metric.getFN()}")
    print(f"  - Precision: {metric.getPrecision():.4f}")
    print(f"  - Recall: {metric.getRecall():.4f}")
    print(f"  - F1 Score: {metric.getF1Score():.4f}")
    



if __name__ == "__main__":
    main()