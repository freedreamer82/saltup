#!/usr/bin/env python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import time
import argparse
from typing import Tuple, Optional, Dict, List
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
    
    return parser.parse_args()

def main(args: Optional[argparse.Namespace] = None) -> None:
    args = args or get_args()
    
    if args.verbose > 1:
        print(json.dumps(vars(args), indent=4))
        
    signal.signal(signal.SIGINT, signal_handler)

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
            
            # Run inference
            yolo_output = yolo.run(frame, args.conf_thres, args.iou_thres)
            
            processed_frame = None
            annotations = []
            
            # Prepare output based on mode
            if args.output_video:
                boxes_with_info = yolo_output.get_boxes()
                processed_frame = draw_boxes_on_image_with_labels_score(
                    frame, 
                    boxes_with_info,
                    class_colors_bgr=class_colors,
                    class_labels=class_labels_dict
                )
            
            if args.output_dataset:
                annotations = [
                    (box_data[1], *box_data[0].get_coordinates(NotationFormat.YOLO))
                    for box_data in yolo_output.get_boxes()
                ]
            
            # Save to dataset
            if dataset and annotations:
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