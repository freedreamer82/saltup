# Ultralytics YOLO 🚀, AGPL-3.0 license

import argparse

import cv2
import numpy as np
import onnxruntime as ort
import torch
import time
import signal
import sys
import logging
import os

from tqdm import tqdm
from saltup.ai.object_detection.yolo.yolo import BaseYolo, ColorMode
from saltup.ai.object_detection.yolo.yolo_type import YoloType
from saltup.ai.object_detection.yolo.yolo_factory import YoloFactory
from saltup.ai.object_detection.utils.bbox  import BBox,BBoxFormat


if __name__ == "__main__":
    # Create an argument parser to handle command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolov8n.onnx", help="Input your ONNX model.")
    parser.add_argument("--type", type=str, default="", help="Input your yolo model type.")
    parser.add_argument("--img", type=str, default=str("bus.jpg"), help="Path to input image or folder.")
    parser.add_argument("--anchors", type=str, default="", help="Path to the anchors if needed.")
    parser.add_argument("--preprocess", type=str, default='true', help="To preprocess or not the image before entering the model")
    parser.add_argument("--label", type=str, help="Path to image label or folder.")
    parser.add_argument("--gui",  action='store_true', help="open gui to draw bbox")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="NMS IoU threshold")
    
    parser.add_argument("--cls_name", type=list[str], default=[], help="name of the classes")
    parser.add_argument("--num_class", type=int, help="number of the classes")
    parser.add_argument("--show_bbox", type=str, help="to print bounding box")
    parser.add_argument("--max-images", type=int, default=0, help="Maximum number of images to process")

    args = parser.parse_args()

    # Registra la funzione di gestione del segnale
    #signal.signal(signal.SIGINT, signal_handler)
    
    yolotype = YoloType.from_string(args.type)
    
    yolo = YoloFactory.create(yolotype, args.model , args.num_class, anchors=args.anchors)
    
    image  = BaseYolo.load_image(args.img, ColorMode.GRAY) #if yolo.get_number_image_channel() == 1 else ColorMode.RGB)
    
    img_height, img_width, _ = image.shape
    
    yoloOut = yolo.run(image, img_height, img_width, args.conf_thres, args.iou_thres)
    
    print(yoloOut.get_boxes())
    
    #yolo_bboxes, yolo_class_ids = BBox.from_yolo_file(args.label, img_width=img_width, img_height=img_height)
    
    #print(yolo_bboxes)
    #metrics = yolo.evaluate([yoloOut.get_boxes(),yoloOut.get_class_ids()], yolo_bboxes, args.iou-thres)
    
    #print(metrics)
            
