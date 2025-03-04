#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys

from saltup.utils import configure_logging
from saltup.ai.object_detection.dataset.yolo_darknet import create_symlinks_by_class, _extract_unique_classes

def get_args():
    """
    Parse command-line arguments for the symlink creation script.

    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description='Create symbolic links for images and labels organized by class.'
    )
    parser.add_argument(
        'images_dir', type=Path, 
        help='Path to images directory'
    )
    parser.add_argument(
        '-l', '--labels-dir', type=Path, required=False, 
        help='Path to labels directory (defaults to images_dir if not specified)'
    )
    parser.add_argument(
        '-o', '--output-dir', type=Path, required=True, 
        help='Path to output directory'
    )
    parser.add_argument(
        '-c', '--classes', nargs='+', 
        help='Optional list of class names'
    )
    parser.add_argument(
        '--log-level', 
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Set the logging level (default: INFO)'
    )
    parser.add_argument(
        '--log-file', 
        help='Optional log file path'
    )
    
    return parser.parse_args()

def main(args=None):
    """
    Main entry point for the script.
    Processes arguments and calls create_symlinks_by_class.
    """
    if not args:
        args = get_args()
    
    # Configure logging
    logger = configure_logging.get_logger(__name__)
    
    # Ensure images_dir exists and contains images
    if not args.images_dir.exists():
        logger.error(f"Images directory does not exist: {args.images_dir}")
        sys.exit(1)
        
    image_dirs = set([
        e.parent.absolute() for e in args.images_dir.rglob('*.jpg')
    ])
    
    if not image_dirs:
        logger.error(f"No JPG images found in: {args.images_dir}")
        sys.exit(1)
    
    # Handle labels directory
    if not args.labels_dir:
        logger.info("Labels directory not specified, using images directory")
        args.labels_dir = args.images_dir
        label_dirs = image_dirs
    else:
        if not args.labels_dir.exists():
            logger.error(f"Labels directory does not exist: {args.labels_dir}")
            sys.exit(1)
            
        label_dirs = set([
            e.parent.absolute() for e in args.labels_dir.rglob('*.txt')
        ])
        
        if not label_dirs:
            logger.error(f"No TXT label files found in: {args.labels_dir}")
            sys.exit(1)
    
    # Validate the directory structure
    if len(image_dirs) != len(label_dirs):
        error_msg = f'Number of images and labels directories do not match: {len(image_dirs)} != {len(label_dirs)}'
        logger.error(error_msg)
        raise ValueError(error_msg)
        sys.exit(1)
    
    # Auto-detect classes if not specified
    if not args.classes:
        logger.info("Classes not specified, auto-detecting from label files")
        lbl_files = list(args.labels_dir.rglob('*.txt'))
        args.classes = _extract_unique_classes(lbl_files)
        logger.info(f"Detected classes: {args.classes}")
    
    # Create output directory if it doesn't exist
    if not args.output_dir.exists():
        logger.info(f"Creating output directory: {args.output_dir}")
        args.output_dir.mkdir(parents=True)
    
    # Process each directory pair
    for img_dir, lbl_dir in zip(image_dirs, label_dirs):
        logger.info(f"Processing images from {img_dir} and labels from {lbl_dir}")
        create_symlinks_by_class(
            imgs_dirpath=img_dir,
            lbls_dirpath=lbl_dir,
            dest_dir=args.output_dir,
            class_names=args.classes
        )
        
    logger.info("Symlink creation completed successfully")

if __name__ == "__main__":
    main()