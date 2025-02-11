import os
import argparse
import tempfile
import logging
import shutil
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
from types import SimpleNamespace
from tqdm import tqdm

from saltup.utils.data.s3 import S3
from saltup.tools import yolo_video


class SignalHandler:
    def __init__(self, tmp_dirpath=None):
        """Initialize with optional temporary directory path."""
        self.tmp_dirpath = tmp_dirpath
        
    def cleanup(self, signum, frame):
        """Clean up temporary resources on script interruption."""
        if self.tmp_dirpath:
            shutil.rmtree(self.tmp_dirpath, ignore_errors=True)
            logging.warning("Script interrupted. Temporary files cleaned up.")
        sys.exit(0)


def setup_logging(log_level: str):
    """
    Configure logging to work with tqdm progress bars.
    
    Args:
        log_level (str): The logging level (e.g., "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").
    """
    class TqdmLoggingHandler(logging.Handler):
        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.write(msg)
            except Exception:
                self.handleError(record)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add tqdm-compatible handler
    handler = TqdmLoggingHandler()
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', 
                                datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


def get_args() -> argparse.Namespace:
    """Parse and return command line arguments."""
    parser = argparse.ArgumentParser(description="Process videos from S3 using YOLO model for inference")
    
    # S3 Configuration
    s3_group = parser.add_argument_group('S3 Configuration')
    s3_group.add_argument("--bucket", type=str, required=True,
                         help="The name of the S3 bucket to interact with")
    
    # AWS Authentication options
    auth_group = parser.add_argument_group('AWS Authentication')
    auth_exclusive = auth_group.add_mutually_exclusive_group(required=True)
    auth_exclusive.add_argument("--aws-credentials", type=str, nargs=2,
                            metavar=('ACCESS_KEY', 'SECRET_KEY'),
                            help="AWS access key ID and secret access key")
    auth_exclusive.add_argument("--aws-credential-filepath", type=str,
                            default="~/.aws/credentials",
                            help="Path to AWS credentials file (default: ~/.aws/credentials)")
    auth_group.add_argument("--section", type=str, default="default",
                         help="Section in credentials file to use (default: default)")
    
    s3_group.add_argument("--s3-folders", type=str, nargs='+', required=True,
                         help="List of S3 folder paths to search for videos")
    s3_group.add_argument("--file-pattern", type=str, default="*.mp4",
                         help="Pattern to filter video files (default: *.mp4)")
    s3_group.add_argument("--start-date", type=str,
                         help="Start date for file filtering (format: DD.MM.YYYY_HH.MM.SSZ)")
    s3_group.add_argument("--end-date", type=str,
                         help="End date for file filtering (format: DD.MM.YYYY_HH.MM.SSZ)")
    
    # YOLO Configuration
    yolo_group = parser.add_argument_group('YOLO Configuration')
    yolo_group.add_argument("--model", type=str, required=True,
                           help="YOLO model file")
    yolo_group.add_argument("--type", type=str, required=True,
                           help="YOLO model type")
    yolo_group.add_argument("--anchors", type=str, default="",
                           help="Anchors config file")
    yolo_group.add_argument("--conf-thres", type=float, default=0.5,
                           help="Confidence threshold")
    yolo_group.add_argument("--iou-thres", type=float, default=0.5,
                           help="IoU threshold")
    
    # Class Configuration
    class_group = parser.add_argument_group('Class Configuration')
    class_mutex = class_group.add_mutually_exclusive_group(required=True)
    class_mutex.add_argument("--num-class", type=int,
                           help="Number of classes")
    class_mutex.add_argument("--cls-name", type=str, nargs="*",
                           help="Class names list")
    
    # Output Configuration
    output_group = parser.add_argument_group('Output Configuration')
    output_group.add_argument("--output-dir", type=str, required=True,
                           help="Base directory for inference output")
    output_group.add_argument("--output-type", type=str, choices=['video', 'dataset'], default='dataset',
                           help="Type of output to generate (default: dataset)")
    
    # Logging Configuration
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                       help="Set logging level")
    
    return parser.parse_args()


def main(args: Optional[argparse.Namespace] = None) -> None:
    """Main execution function."""
    args = args or get_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logging.info("=== Starting YOLO S3 Inference ===")
    
    # Initialize SignalHandler
    try:
        # Initialize S3 client
        if args.aws_credentials:
            s3_client = S3(
                bucket_name=args.bucket,
                aws_access_key_id=args.aws_credentials[0],
                aws_secret_access_key=args.aws_credentials[1]
            )
        elif args.aws_credential_filepath:
            s3_client = S3(
                bucket_name=args.bucket,
                aws_credential_filepath=args.aws_credential_filepath,
                section=args.section
            )
        
        # Get list of videos matching criteria from all folders
        video_files = []
        for folder in tqdm(args.s3_folders, desc="Scanning folders", unit="folder"):
            folder_files = s3_client.list_files_by_date(
                s3_folder=folder,
                start_date=args.start_date,
                end_date=args.end_date,
                patterns=args.file_pattern
            )
            video_files.extend([os.path.join(folder, f) for f in folder_files])
            logging.info(f"Found {len(folder_files)} videos in {folder}")
        
        if not video_files:
            logging.warning("No video files found matching the specified criteria")
            return
        
        logging.info(f"Found {len(video_files)} total video files to process")
        
        # Create temporary directory for video processing
        with tempfile.TemporaryDirectory() as temp_dir:
            logging.debug(f"Temporary directory: {temp_dir}")
            
            # Setup signal handler for cleanup
            handler = SignalHandler(temp_dir)
            signal.signal(signal.SIGINT, handler.cleanup)
            
            # Process each video
            progress_bar = tqdm(
                video_files, 
                desc="Processing videos",
                unit="video",
                dynamic_ncols=True
            )
            for video_file in progress_bar:
                try:
                    progress_bar.set_description(f"Processing {os.path.basename(video_file)}")
                    
                    # Download video to temporary directory
                    local_video_path = os.path.join(temp_dir, os.path.basename(video_file))
                    logging.info(f"Downloading: {video_file}")
                    s3_client.download_file(
                        file_path=video_file,
                        destination_path=temp_dir
                    )
                    
                    # Prepare yolo_video arguments
                    video_output_dir = Path(args.output_dir) / Path(video_file).stem
                    video_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    yolo_args = SimpleNamespace(
                        model=args.model,
                        type=args.type,
                        input_video=local_video_path,
                        output_video=str(video_output_dir / "output.mp4") if args.output_type == 'video' else None,
                        output_dataset=str(video_output_dir) if args.output_type == 'dataset' else None,
                        anchors=args.anchors,
                        conf_thres=args.conf_thres,
                        iou_thres=args.iou_thres,
                        num_class=args.num_class,
                        cls_name=args.cls_name,
                        fps=None,
                        verbose=args.log_level == "DEBUG"
                    )
                    
                    # Process video using yolo_video
                    yolo_video.main(yolo_args)
                    logging.info(f"Completed processing: {video_file}")
                    
                    # Clean up downloaded video
                    os.remove(local_video_path)
                    
                except Exception as e:
                    logging.error(f"Error processing video {video_file}: {e}")
                    continue
        
        logging.info("=== Processing completed successfully ===")
        
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()