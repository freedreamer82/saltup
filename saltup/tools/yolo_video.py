#!/usr/bin/env python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable OneDNN optimizations
import time
import argparse
from typing import Tuple
from tqdm import tqdm
from pathlib import Path
import signal
import sys
import json

from saltup.ai.object_detection.yolo.yolo import BaseYolo, YoloOutput
from saltup.ai.object_detection.yolo.yolo_type import YoloType
from saltup.ai.object_detection.yolo.yolo_factory import YoloFactory
from saltup.ai.object_detection.utils.bbox import draw_boxes_on_image_with_labels_score, NotationFormat
from saltup.utils.data.image.image_utils import Image, generate_random_bgr_colors
from saltup.utils.data.video.video_utils import process_video, get_video_properties
from saltup.ai.object_detection.dataset.yolo_darknet import YoloDataset


def signal_handler(sig, frame):
    """Handle Ctrl+C signal to gracefully exit the program."""
    print("\nProgram interrupted with Ctrl + C!")
    sys.exit(0)  # Terminate the script
 
def process_frame(
    yolo: BaseYolo, 
    frame, 
    frame_number, 
    args, 
    class_colors_dict, 
    class_labels_dict
) -> Image:
    """
    Process a video frame using the YOLO model.
    
    Args:
        yolo: The YOLO model instance.
        frame: Input frame (as a Image).
        frame_number: Current frame number.
        args: Command-line arguments.
        class_colors_dict: Dictionary mapping class IDs to colors.
        class_labels_dict: Dictionary mapping class IDs to class names.
    
    Returns:
        frame_with_boxes: Frame with bounding boxes drawn.
    """
    # Run YOLO inference
    yoloOut = yolo.run(frame, args.conf_thres, args.iou_thres)
    boxes_with_info = yoloOut.get_boxes()
    
    boxes_list = [box[0] for box in boxes_with_info]

    # Draw bounding boxes on the frame
    frame_with_boxes = draw_boxes_on_image_with_labels_score(
        frame, 
        boxes_with_info,
        class_colors_bgr=class_colors_dict,
        class_labels=class_labels_dict
    ) 
    
    return frame_with_boxes

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Input NN model.")
    parser.add_argument("--type", type=str, required=True, help="Input your YOLO model type.")
    parser.add_argument("--input-video", type=str, required=True, help="Path to input video.")
    
    group0 = parser.add_mutually_exclusive_group(required=True)
    group0.add_argument("--output-video", type=str, help="Path to save the output video.")
    group0.add_argument("--output-dataset", type=str, help="Path to save the output dataset.")
    
    parser.add_argument("--anchors", type=str, default="", help="Path to the anchors if needed.")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="Confidence threshold.")
    parser.add_argument("--iou-thres", type=float, default=0.5, help="NMS IoU threshold.")
    
    group1 = parser.add_mutually_exclusive_group(required=True)
    group1.add_argument("--num-class", type=int, help="Number of the classes.")
    group1.add_argument("--cls-name", type=str, nargs="*", default=[], help="List of class names separated by spaces.")
    
    parser.add_argument("--fps", type=int, default=None, help="FPS of the output video (default: same as input).")
    
    parser.add_argument("--verbose", "-v", action="store_true", help="Increase output verbosity.")
    
    return parser.parse_args()

def main(args=None):
    """
    Main function to process a video using the YOLO model.
    
    Args:
        args: Command-line arguments.
    """
    
    # If args is not provided, use sys.argv[1:]
    if args is None:
        args = sys.argv[1:]

    args = get_args()
    
    if args.verbose:
        print(json.dumps(vars(args), indent=4))
    
    signal.signal(signal.SIGINT, signal_handler)

    # Load classes names
    class_labels = args.cls_name    
    num_classes = len(class_labels) if class_labels else args.num_class
    
    # Generate classes colors
    class_colors = generate_random_bgr_colors(num_classes)
    class_colors_dict = {i: color for i, color in enumerate(class_colors)}
    class_labels_dict = {i: label for i, label in enumerate(class_labels)} if class_labels else {i: f"class_{i}" for i in range(num_classes)}

    yolotype = YoloType.from_string(args.type)
    if args.anchors:
        yolo = YoloFactory.create(yolotype, args.model, args.num_class, anchors=args.anchors)
    else:
        yolo = YoloFactory.create(yolotype, args.model, args.num_class)

    # Get video properties using OpenCV (manual counting)
    input_fps, total_frames, w, h = get_video_properties(args.input_video)
    print(f"Input video FPS: {input_fps}")
    print(f"Total frames: {total_frames}")

    # Define the callback function for process_video
    if args.output_video:

        def callback(frame, frame_number, total_frames):
            start_time = time.time()
            result = process_frame(yolo, frame, frame_number, args, class_colors_dict, class_labels_dict)
            elapsed_time = time.time() - start_time
            
            # Update the progress bar
            pbar.set_postfix({
                "Frame": f"{frame_number + 1}/{total_frames}",  # Show current frame number (1-based)
                "Time per frame": f"{elapsed_time:.4f}s"
            })
            pbar.update(1)  # Increment the progress bar by 1

            return result

    elif args.output_dataset:
        # Create Yolo Dataset
        root_dir = Path(args.output_dataset)
        root_dir.mkdir(parents=True, exist_ok=True)
        dataset = YoloDataset(root_dir, root_dir, refresh_each=3)
        
        # Create classes names file
        with open(root_dir / "classes.names", 'w') as f:
            f.writelines('\n'.join(class_labels if class_labels else [f"class_{i}" for i in range(num_classes)]))
                
        def callback(frame, frame_number, total_frames):
            # Run YOLO inference
            start_time = time.time()
            yoloOut = yolo.run(frame, args.conf_thres, args.iou_thres)
            elapsed_time = time.time() - start_time
            
            # Update the progress bar
            pbar.set_postfix({
                "Frame": f"{frame_number + 1}/{total_frames}",  # Show current frame number (1-based)
                "Time per frame": f"{elapsed_time:.4f}s"
            })
            pbar.update(1)  # Increment the progress bar by 1
            
            dataset.save_image_annotations(
                image_id=f"{Path(args.input_video).stem}_{frame_number:05d}", 
                image=frame,
                annotations=[
                    (e[1], *e[0].get_coordinates(NotationFormat.YOLO))
                    for e in yoloOut.get_boxes()
                ],
                overwrite= True
            )
            
            return None
    else:
        def callback(frame, frame_number, total_frames):
            return frame  # No processing, return the original frame

    # Process the video using the integrated process_video function
    with tqdm(total=total_frames, desc="Processing video", unit="frame") as pbar:
        def callback_with_progress(frame, frame_number, total_frames):
            result = callback(frame, frame_number, total_frames)
            return result

        process_video(
            video_input=args.input_video,
            callback=callback_with_progress,
            video_output=args.output_video,
            fps=args.fps 
        )

    print(f"Video processing complete. Output saved to {args.output_video}")

if __name__ == "__main__":
    main()