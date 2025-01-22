#!/usr/bin/env python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable OneDNN optimizations
import time
import argparse
import numpy as np
from typing import List, Tuple, Dict
from tqdm import tqdm
from pathlib import Path

from saltup.ai.object_detection.yolo.yolo import BaseYolo, YoloOutput
from saltup.ai.object_detection.yolo.yolo_type import YoloType
from saltup.ai.object_detection.yolo.yolo_factory import YoloFactory
from saltup.ai.object_detection.utils.bbox import BBox, draw_boxes_on_image_with_labels_score, BBoxFormat , draw_boxes_on_image
from saltup.utils.data.image.image_utils import ColorMode, ColorsBGR, Image, ImageFormat
from saltup.ai.object_detection.utils.metrics import Metric
from saltup.utils.data.image.image_utils import generate_random_bgr_colors
from saltup.utils.data.video.video_utils import process_video,get_video_properties
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

def process_frame(yolo, frame, frame_number, args, class_colors_dict, class_labels_dict):
    """
    Process a video frame using the YOLO model.
    
    Args:
        yolo: The YOLO model instance.
        frame: Input frame (as a NumPy array).
        frame_number: Current frame number.
        args: Command-line arguments.
        class_colors_dict: Dictionary mapping class IDs to colors.
        class_labels_dict: Dictionary mapping class IDs to class names.
    
    Returns:
        frame_with_boxes: Frame with bounding boxes drawn.
    """
    # Get input information from the YOLO model
    model_color = yolo.get_input_model_color()
    channels = yolo.get_input_model_channel()
    
    # Convert frame to Image object using the input information
    image = Image(frame, color_mode=  model_color if channels > 1 else ColorMode.GRAY)
   
    # Run YOLO inference
    yoloOut = yolo.run(image, args.conf_thres, args.iou_thres)
    boxes_with_info = yoloOut.get_boxes()
    
    boxes_list = [box[0] for box in boxes_with_info]

    # im2 = draw_boxes_on_image(bboxes=boxes_list, image=image, color=ColorsBGR.RED.value, thickness=2)
    # im2.show()

    # Draw bounding boxes on the frame
    frame_with_boxes = draw_boxes_on_image_with_labels_score(
        image, 
        boxes_with_info,
        class_colors_bgr=class_colors_dict,
        class_labels=class_labels_dict
    )  # Convert back to NumPy array
    
    # frame_with_boxes.show()
    return frame_with_boxes.get_data()

def main(args=None):
    # Se args non è fornito, usa sys.argv[1:]
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Input NN model.")
    parser.add_argument("--type", type=str, help="Input your YOLO model type.")
    parser.add_argument("--input_video", type=str, required=True, help="Path to input video.")
    parser.add_argument("--output_video", type=str, help="Path to save the output video.")
    parser.add_argument("--anchors", type=str, default="", help="Path to the anchors if needed.")
    parser.add_argument("--label", type=str, help="Path to frame labels folder.")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="Confidence threshold.")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="NMS IoU threshold.")
    parser.add_argument("--num_class", type=int, help="Number of the classes.")
    parser.add_argument("--cls_name", type=str, default="", help="Comma-separated list of class names.")
    parser.add_argument("--fps", type=int, default=None, help="FPS of the output video (default: same as input).")

    # Analizza gli argomenti
    args = parser.parse_args(args)
    
    signal.signal(signal.SIGINT, signal_handler)



    """
    Main function to process a video using the YOLO model.
    
    Args:
        args: Command-line arguments.
    """
    # Load class names and colors
    if args.cls_name:
        class_labels = args.cls_name.split(',')
    else:
        class_labels = []
    
    num_classes = len(class_labels) if class_labels else args.num_class
    class_colors = generate_random_bgr_colors(num_classes)
    class_colors_dict = {i: color for i, color in enumerate(class_colors)}
    class_labels_dict = {i: label for i, label in enumerate(class_labels)} if class_labels else {i: f"class_{i}" for i in range(num_classes)}

    yolotype = YoloType.from_string(args.type)
    if args.anchors:
        yolo = YoloFactory.create(yolotype, args.model, args.num_class, anchors=args.anchors)
    else:
        yolo = YoloFactory.create(yolotype, args.model, args.num_class)

    # Get video properties using OpenCV (manual counting)
    input_fps, total_frames,w,h = get_video_properties(args.input_video)
    print(f"Input video FPS: {input_fps}")
    print(f"Total frames: {total_frames}")

    # Define the callback function for process_video
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