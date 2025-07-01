#!/usr/bin/env python3
import argparse
import os

from saltup.ai.object_detection.dataset.yolo_darknet import split_and_organize_dataset


def get_args():
    parser = argparse.ArgumentParser(
        description="Split YOLO dataset into train/val/test directories."
    )

    parser.add_argument(
        "-ld", "--labels-dir", required=True, 
        help="Path to the directory containing YOLO label files (.txt)"
    )
    parser.add_argument(
        "-id", "--images-dir", required=True, 
        help="Path to the directory containing image files (.jpg/.jpeg)"
    )
    parser.add_argument(
        "-m", "--max-images-per-class", type=int, default=0, 
        help="Maximum number of images per class (default: 0, no limit)"
    )
    parser.add_argument(
        "-tr", "--train-ratio", type=float, default=0.7, 
        help="Proportion of training set (default: 0.7)"
    )
    parser.add_argument(
        "-vr", "--val-ratio", type=float, default=0.2, 
        help="Proportion of validation set (default: 0.2)"
    )
    parser.add_argument(
        "-te", "--test-ratio", type=float, default=0.1, 
        help="Proportion of test set (default: 0.1)"
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, default=None,
        help="Path to the output directory for the split dataset. If not specified, the current directory will be used."
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable detailed progress output."
    )

    return parser.parse_args()

def main():
    args = get_args()
    
    print(f"Train ratio: {args.train_ratio}, Validation ratio: {args.val_ratio}, Test ratio: {args.test_ratio}")
    # Validate ratios
    if not (0.0 <= args.train_ratio <= 1.0 and 0.0 <= args.val_ratio <= 1.0 and 0.0 <= args.test_ratio <= 1.0):
        raise ValueError("Train, validation, and test ratios must be between 0 and 1.")
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError(f"Train, validation, and test ratios must sum to 1. Current sum: {args.train_ratio + args.val_ratio + args.test_ratio}")

    # Check if directories exist
    if not os.path.exists(args.labels_dir):
        raise FileNotFoundError(f"Labels directory not found: {args.labels_dir}")
    if not os.path.exists(args.images_dir):
        raise FileNotFoundError(f"Images directory not found: {args.images_dir}")

    # Perform dataset split and organization
    split_and_organize_dataset(
        labels_dir=args.labels_dir,
        images_dir=args.images_dir,
        max_images_per_class=args.max_images_per_class,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        output_dir=args.output_dir,
        verbose=args.verbose
    )

    print("Dataset split and organization completed successfully.")

if __name__ == "__main__":
    main()
