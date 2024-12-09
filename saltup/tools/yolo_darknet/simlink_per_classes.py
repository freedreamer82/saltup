#!/usr/bin/env python3
import sys
import argparse

from saltup.utils import configure_logging
from saltup.ai.yolo.dataset.data_processing import create_symlinks_by_class

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the symlink creation script.

    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description='Create symbolic links for images and labels organized by class.'
    )
    parser.add_argument(
        '-i', '--images', 
        required=True, 
        help='Path to directory containing images'
    )
    parser.add_argument(
        '-l', '--labels', 
        required=True, 
        help='Path to directory containing label files'
    )
    parser.add_argument(
        '-d', '--destination', 
        required=True, 
        help='Destination directory for symlinks'
    )
    parser.add_argument(
        '-c', '--classes', 
        nargs='+', 
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

def main(args):
    """
    Main entry point for the script.
    Parses command-line arguments and calls create_symlinks_by_class.
    """

    # Get logger for this module
    logger = configure_logging.get_logger(__name__)

    try:
        # Call the main function with parsed arguments
        create_symlinks_by_class(
            imgs_dirpath=args.images,
            lbls_dirpath=args.labels,
            dest_dir=args.destination,
            class_names=args.classes
        )
        logger.info("Symlink creation completed successfully")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == '__main__':
    # Parse arguments
    args = parse_args()
    main(args)