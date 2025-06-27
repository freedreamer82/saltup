#!/usr/bin/env python3
import sys
import argparse

from saltup.utils import configure_logging
from saltup.utils.misc import unify_files


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Unify files from multiple directories into a single destination.'
    )
    
    parser.add_argument(
        '-s', '--sources',
        required=True,
        nargs='+',
        help='Source directories to process'
    )
    parser.add_argument(
        '-d', '--destination',
        required=True,
        help='Destination directory'
    )
    parser.add_argument(
        '-f', '--filters',
        nargs='+',
        help='File patterns to include (e.g., "*.txt" "*.pdf")'
    )
    parser.add_argument(
        '--divide-by-extension',
        action='store_true',
        help='Organize files into subdirectories by extension'
    )
    parser.add_argument(
        '--move',
        action='store_true',
        help='Move files instead of copying them'
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the script."""
    args = parse_arguments()
    print(f"Arguments received: {args}")
    
    try:
        
        logger = configure_logging.get_logger(__name__)
        
        # Execute file unification
        processed, failed = unify_files(
            source_dirs=args.sources,
            destination=args.destination,
            filters=args.filters,
            divide_by_extension=args.divide_by_extension,
            move_files=args.move
        )
        
        if failed > 0:
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()