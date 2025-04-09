#!/usr/bin/env python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import time
import argparse
import random
from typing import Tuple, Optional, Dict, List
from tqdm import tqdm
from pathlib import Path
import signal
import sys
import json

from saltup.ai.object_detection.yolo.yolo_type import YoloType
from saltup.ai.object_detection.yolo.yolo_factory import YoloFactory
from saltup.ai.object_detection.utils.bbox import draw_boxes_on_image_with_labels_score, BBoxFormat
from saltup.utils.data.image.image_utils import Image, generate_random_bgr_colors
from saltup.utils.data.video.video_utils import process_video, get_video_properties
from saltup.ai.object_detection.dataset.yolo_darknet import YoloDataset

def signal_handler(sig, frame):
    print("\nProgram interrupted with Ctrl + C!")
    sys.exit(0)

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="YOLO model file")
    parser.add_argument("--type", type=str, required=True, help="YOLO model type")
    parser.add_argument("--input-video", type=str, required=True, help="Input video path")
    
    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument("--output-video", type=str, help="Output video path")
    output_group.add_argument("--output-dataset", type=str, help="Output dataset directory")
    
    parser.add_argument("--anchors", type=str, default="", help="Anchors config file")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.5, help="IoU threshold")
    
    class_group = parser.add_mutually_exclusive_group(required=True)
    class_group.add_argument("--num-class", type=int, help="Number of classes")
    class_group.add_argument("--cls-name", type=str, nargs="*", help="Class names list")
    
    parser.add_argument("--fps", type=int, help="Output FPS (default: input FPS)")
    parser.add_argument("--verbose", "-v", type=int, nargs='?', const=1, default=0, help="Verbose output")
    parser.add_argument("--frame-percent", type=float, default=1.0, help="Percentage of frames to process randomly (only with --output-dataset)")
    
    return parser.parse_args()

def main(args: Optional[argparse.Namespace] = None) -> None:
    args = args or get_args()
    
    if args.verbose > 1:
        print(json.dumps(vars(args), indent=4))
        
    signal.signal(signal.SIGINT, signal_handler)

    # Validate frame-percent parameter
    if args.frame_percent <= 0 or args.frame_percent > 1.0:
        print("Error: --frame-percent must be between 0 and 100")
        sys.exit(1)
    
    # Check that frame-percent is only used with output-dataset
    if args.frame_percent < 1.0 and not args.output_dataset:
        print("Warning: --frame-percent is only applicable with --output-dataset, ignoring")

    # Initialize class metadata
    class_labels = args.cls_name or [f"class_{i}" for i in range(args.num_class)]
    class_colors = {i: color for i, color in enumerate(generate_random_bgr_colors(len(class_labels)))}
    class_labels_dict = {i: lbl for i, lbl in enumerate(class_labels)}

    # Initialize YOLO model
    yolotype = YoloType.from_string(args.type)
    yolo_kwargs = {
        "yolo_type": yolotype,
        "modelpath": args.model,
        "number_class": len(class_labels),
        **({"anchors": args.anchors} if args.anchors else {})
    }        
    yolo = YoloFactory.create(**yolo_kwargs)

    # Get video properties
    input_fps, total_frames, width, height = get_video_properties(args.input_video)
    if args.verbose:
        print(f"Input video FPS: {input_fps}")
        print(f"Total frames: {total_frames}")
        if args.output_dataset and args.frame_percent < 1.0:
            print(f"Processing approximately {args.frame_percent*100}% of frames randomly")

    # Initialize dataset if needed
    dataset = None
    if args.output_dataset:
        output_dir = Path(args.output_dataset)
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset = YoloDataset(output_dir, output_dir)
        (output_dir / "classes.names").write_text('\n'.join(class_labels))

    # Main processing loop
    with tqdm(total=total_frames, desc="Processing video", unit="frame") as pbar:
        
        def process_callback(frame: Image, frame_number: int, _: int) -> Optional[Image]:
            start_time = time.time()
            
            # Determine if we should process this frame (for dataset mode)
            process_for_dataset = True
            if args.output_dataset and args.frame_percent < 1.0:
                # Use random sampling to decide whether to process this frame
                process_for_dataset = random.random() <= args.frame_percent
            
            processed_frame = None
            annotations = []
            
            # Run inference (always for video output, conditionally for dataset)
            if args.output_video or (args.output_dataset and process_for_dataset):
                yolo_output = yolo.run(frame, args.conf_thres, args.iou_thres)
                
                # Prepare output based on mode
                if args.output_video:
                    boxes_with_info = yolo_output.get_boxes()
                    processed_frame = draw_boxes_on_image_with_labels_score(
                        frame, 
                        boxes_with_info,
                        class_colors_bgr=class_colors,
                        class_labels=class_labels_dict
                    )
                
                if args.output_dataset and process_for_dataset:
                    annotations = [
                        (box_data[1], *box_data[0].get_coordinates(BBoxFormat.YOLO))
                        for box_data in yolo_output.get_boxes()
                    ]
            
            # Save to dataset if we're processing this frame
            if dataset and process_for_dataset and annotations:
                dataset.save_image_annotations(
                    image_id=f"{Path(args.input_video).stem}_{frame_number:05d}",
                    image=frame,  # Save original frame without boxes
                    annotations=annotations,
                    overwrite=True
                )
            
            # Update progress
            pbar.set_postfix({
                "Frame": f"{frame_number+1}/{total_frames}",
                "Time/frame": f"{time.time()-start_time:.3f}s"
            })
            pbar.update(1)
            
            return processed_frame if args.output_video else None

        process_video(
            video_input=args.input_video,
            callback=process_callback,
            video_output=args.output_video,
            fps=args.fps or input_fps
        )

    if args.output_video:
        print(f"Video saved: {args.output_video}")
    if args.output_dataset:
        print(f"Dataset saved: {args.output_dataset}")


if __name__ == "__main__":
    main()