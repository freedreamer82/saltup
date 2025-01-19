import argparse
import os
import cv2
import numpy as np
from typing import List, Tuple
from tqdm import tqdm
from saltup.ai.object_detection.yolo.yolo import BaseYolo, ColorMode
from saltup.ai.object_detection.yolo.yolo_type import YoloType
from saltup.ai.object_detection.yolo.yolo_factory import YoloFactory
from saltup.ai.object_detection.utils.bbox import BBox, draw_boxes_on_image_with_labels_score,BBoxFormat

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
    # Load the image in grayscale or RGB mode based on the model's input channels
    image = BaseYolo.load_image(image_path, ColorMode.GRAY if yolo.get_number_image_channel() == 1 else ColorMode.RGB)
    img_height, img_width, _ = image.shape
    
    # Run the YOLO model on the image
    yoloOut = yolo.run(image, img_height, img_width, args.conf_thres, args.iou_thres)
    
    # Get bounding boxes, class IDs, and scores in a single call
    bboxes_with_labels_score = yoloOut.get_boxes()  # Assuming "CENTER" format is used
    
    # If GUI mode is enabled, draw bounding boxes, labels, and scores on the image
    if args.gui:
        image_with_boxes = draw_boxes_on_image_with_labels_score(image, bboxes_with_labels_score)
        cv2.imshow("YOLO Output", image_with_boxes)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Extract boxes and class IDs for further processing
    boxes = [bbox for bbox, _, _ in bboxes_with_labels_score]
    class_ids = [class_id for _, class_id, _ in bboxes_with_labels_score]
    
    return boxes, class_ids

def evaluate_predictions(yolo, predictions, labels, iou_thres):
    """
    Evaluate the predictions against the ground truth labels.
    
    Args:
        yolo: The YOLO model instance.
        predictions: List of predicted bounding boxes and class IDs.
        labels: List of ground truth bounding boxes and class IDs.
        iou_thres: IoU threshold for evaluation.
    
    Returns:
        metrics: Evaluation metrics (e.g., mAP, precision, recall).
    """
    metrics = yolo.evaluate(predictions, labels, iou_thres)
    return metrics


def main(args):
    """
    Main function to process images and evaluate predictions.
    
    Args:
        args: Command-line arguments.
    """
    # Determine the YOLO model type and create the YOLO instance
    yolotype = YoloType.from_string(args.type)
    if args.anchors:
        yolo = YoloFactory.create(yolotype, args.model, args.num_class, anchors=args.anchors)
    else:
        yolo = YoloFactory.create(yolotype, args.model, args.num_class)
    
    # Get the list of image paths (single image or directory of images)
    if os.path.isdir(args.img):
        image_paths = [
            os.path.join(args.img, fname) 
            for fname in os.listdir(args.img) 
            if fname.lower().endswith(('.jpg', '.png', '.jpeg'))
        ]
    else:
        image_paths = [args.img]
    
    # Limit the number of images to process if max_images is specified
    if args.max_images > 0:
        image_paths = image_paths[:args.max_images]
    
    all_predictions = []  # Store all predictions (bounding boxes and class IDs)
    all_labels = []       # Store all ground truth labels (bounding boxes and class IDs)
    
    # Process each image
    for image_path in tqdm(image_paths, desc="Processing images"):
        try:
            # Load the image and get its dimensions
            image = BaseYolo.load_image(image_path, ColorMode.GRAY if yolo.get_number_image_channel() == 1 else ColorMode.RGB)
            img_height, img_width, _ = image.shape
            
            # Run the YOLO model on the image
            yoloOut = yolo.run(image, img_height, img_width, args.conf_thres, args.iou_thres)
            
            # Get bounding boxes, class IDs, and scores from get_boxes()
            boxes_with_info = yoloOut.get_boxes()  # Use CORNERS format for drawing
            boxes = [bbox for bbox, _, _ in boxes_with_info]  # Extract BBox objects
            class_ids = [class_id for _, class_id, _ in boxes_with_info]  # Extract class IDs
            scores = [score for _, _, score in boxes_with_info]  # Extract confidence scores
            
            # Store predictions
            all_predictions.append((boxes, class_ids))
            
            # Print processing times
            print(f"\nProcessing times for {image_path}:")
            print(f"  Pre-processing time: {yoloOut.get_preprocessing_time():.2f} ms")
            print(f"  Inference time: {yoloOut.get_inference_time():.2f} ms")
            print(f"  Post-processing time: {yoloOut.get_postprocessing_time():.2f} ms")
            print(f"  Total processing time: {yoloOut.get_total_processing_time():.2f} ms")
            
            # Print bounding boxes, format, class IDs, and scores
            print("\nBounding boxes, format, class IDs, and scores:")
            for i, (bbox, class_id, score) in enumerate(zip(boxes, class_ids, scores)):
                print(f"  BBox {i + 1}:")
                print(f"    Coordinates: {bbox.get_coordinates()}")  # Get coordinates in the current format
                print(f"    Format: {bbox.get_format().name}")  # Get format (CORNERS, CENTER, TOPLEFT)
                print(f"    Class ID: {class_id}")  # Print class ID
                print(f"    Confidence score: {score:.4f}")
            
            # If labels are provided, load the corresponding YOLO labels
            if args.label:
                label_path = os.path.join(
                    args.label, 
                    os.path.basename(image_path).replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt')
                )
                if os.path.exists(label_path):
                    yolo_bboxes, yolo_class_ids = BBox.from_yolo_file(label_path, img_width=img_width, img_height=img_height)
                    all_labels.append((yolo_bboxes, yolo_class_ids))
            
            # If GUI mode is enabled, draw bounding boxes, labels, and scores on the image
            if args.gui:
                image_with_boxes = draw_boxes_on_image_with_labels_score(image, yoloOut.get_boxes(format=BBoxFormat.CORNERS))
                cv2.imshow("YOLO Output", image_with_boxes)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            continue
    
    # If labels are provided, evaluate the predictions
    if args.label and all_labels:
        try:
            metrics = evaluate_predictions(yolo, all_predictions, all_labels, args.iou_thres)
            print("Evaluation Metrics:", metrics)
        except Exception as e:
            print(f"Error during evaluation: {e}")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Input NN model.")
    parser.add_argument("--type", type=str,  help="Input your YOLO model type.")
    parser.add_argument("--img", type=str, default="bus.jpg", help="Path to input image or folder.")
    parser.add_argument("--anchors", type=str, default="", help="Path to the anchors if needed.")
    parser.add_argument("--preprocess", type=str, default='true', help="Preprocess the image before entering the model.")
    parser.add_argument("--label", type=str, help="Path to image label or folder.")
    parser.add_argument("--gui", action='store_true', help="Open GUI to draw bounding boxes.")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="Confidence threshold.")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="NMS IoU threshold.")
    parser.add_argument("--cls_name", type=list, default=[], help="Name of the classes.")
    parser.add_argument("--num_class", type=int, help="Number of the classes.")
    parser.add_argument("--show_bbox", type=str, help="Print bounding box information.")
    parser.add_argument("--max_images", type=int, default=0, help="Maximum number of images to process.")

    args = parser.parse_args()
    main(args)