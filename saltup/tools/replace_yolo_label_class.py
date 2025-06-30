#!/usr/bin/env python3
"""
CLI tool to replace a YOLO class ID in label files using replace_label_class().
"""
import argparse
import sys

from saltup.ai.object_detection.dataset.yolo_darknet import replace_label_class


def get_args():
    parser = argparse.ArgumentParser(
        description="Replace a YOLO class ID in label files with a new class ID. "
                    "You must specify either --label-dir or --file-list, but not both."
    )
    parser.add_argument(
        "old_class_id", type=int, 
        help="Original class ID to replace"
    )
    parser.add_argument(
        "new_class_id", type=int, 
        help="New class ID to assign"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--label-dir", type=str, 
        help="Directory containing YOLO label files"
    )
    group.add_argument(
        "--file-list", type=str, nargs='+', 
        help="List of label files to process"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", 
        help="Enable verbose output"
    )
    args = parser.parse_args()
    return args

def main():
    args = get_args()

    try:
        result = replace_label_class(
            old_class_id=args.old_class_id,
            new_class_id=args.new_class_id,
            label_dir=args.label_dir,
            filepath_list=args.file_list,
            verbose=args.verbose
        )
        print(f"Modified {result[0]} files.")
        if args.verbose and result[1]:
            print("Modified files:")
            for f in result[1]:
                print(f"  {f}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
