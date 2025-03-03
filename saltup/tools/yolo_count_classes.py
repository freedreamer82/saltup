import os
import json
import argparse
from collections import Counter
from tqdm import tqdm

def get_agrs():
    parser = argparse.ArgumentParser(description="Count occurrences of each class in dataset annotations.")
    parser.add_argument('labels_dir', type=str, help='Path to the directory containing annotation label files.')
    parser.add_argument('--format', type=str, choices=['yolo', 'coco'], default='yolo', 
                        help="Format of annotation files (default: 'yolo').")
    parser.add_argument('--class-names', type=str, nargs='+', 
                        help='Optional list of class names for more readable output.')
    parser.add_argument('-v', '--verbose', default=False, action='store_true', 
                        help='Enable verbose mode to display progress during processing.')

    return parser.parse_args()


def main(args=None):
    """
    Main function to parse command-line arguments and count occurrences of classes in dataset annotations.
    """
    
    if not args:
        args = get_agrs()

    # Count label occurrences and annotated images based on the provided arguments
    if args.format == 'yolo':
        from saltup.ai.object_detection.dataset.yolo_darknet import count_objects
        class_counts, num_images = count_objects(
            labels_dir=args.labels_dir, 
            class_names=args.class_names,
            verbose=args.verbose
        )
    elif args.format == 'coco':
        from saltup.ai.object_detection.dataset.coco import count_annotations
        class_counts, num_images = count_annotations(
            args.labels_dir, 
            class_names=args.class_names,
            verbose=args.verbose
        )
    else:  
        raise ValueError(f"Unsupported annotation format: {args.format}")

    # Print the results
    print(f"Number of annotated images: {num_images}")
    print("Class occurrences:")
    print(json.dumps(class_counts, indent=3))


if __name__ == '__main__':
    args = get_agrs()
    main(args)
