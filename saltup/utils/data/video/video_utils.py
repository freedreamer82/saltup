import os
import cv2
import numpy as np
from pathlib import Path
import subprocess
from typing import Callable, Union , List, Optional

def create_avi_from_jpg(folder: str, output_filename: str, fps: int = 4) -> None:
    """
    Creates an MJPEG video in an AVI container from JPEG images in a specified folder.

    Args:
        folder (str): Path to the folder containing the JPEG images.
        output_filename (str): Name of the output AVI video file.
        fps (int, optional): Frames per second for the output video. Defaults to 4.

    Returns:
        None
    """

    # Get a sorted list of JPEG files in the folder
    image_files: List[str] = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".jpg")])

    # Read the first image to get its dimensions
    first_image = cv2.imread(image_files[0])
    height, width, _ = first_image.shape

    # Create a VideoWriter object with the specified output filename and FPS
    fourcc= cv2.VideoWriter_fourcc(*'MJPG')
    video = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
   
    if not video.isOpened():
        print("Error opening VideoWriter")
        exit()

    # Iterate through the image files and write each one as a frame in the video
    for image_file in image_files:
        frame = cv2.imread(image_file)
        if frame is None:
            print(f"Problem during image handling: {image_file}")
            continue
        video.write(frame)

    # Release the VideoWriter object to close the output video file
    video.release()


def convert_ts_to_mp4(input_path: str, output_path: str, input_file_ts: str) -> None:
    '''
    Converts a .ts video file to .mp4 format using FFmpeg.

    Args:
        input_path (str): Directory path of the input .ts video file.
        output_path (str): Directory path where the output .mp4 video will be saved.
        input_file_ts (str): Name of the .ts file to be converted.

    Returns:
        None
    '''
    # Output name
    output_file_ts = input_file_ts.replace("ts", "mp4")
    # Conversion
    subprocess.call(['ffmpeg', '-i', os.path.join(input_path, input_file_ts), "-c", "copy", os.path.join(output_path, output_file_ts)])


def extract_jpg_frames_from_video(
    video_path: str,  
    frames_output_dir: str = "", 
    overwrite: bool = False, 
    start_frame: int = -1, 
    end_frame: int = -1, 
    frame_interval: int = 1, 
    filename_prefix:str=""
) -> int:
    '''Extracts JPG frames from a video file.

    Args:
        video_path (str): Path to the source video file.
        frames_output_dir (str, optional): Destination directory for saving extracted frames.
            If not specified, uses the current working directory.
        overwrite (bool, optional): If True, overwrites any existing files with the same name.
            If False, skips extraction for frames that already have a corresponding file. Default False.
        start_frame (int, optional): Frame number to start extraction from.
            A value of -1 indicates starting from the beginning of the video. Default -1.
        end_frame (int, optional): Frame number to end extraction at.
            A value of -1 indicates continuing until the end of the video. Default -1. 
        frame_interval (int, optional): Frame extraction interval.
            For example, 1 saves every frame, 2 saves one frame every two frames, etc. Default 1.
        filename_prefix (str, optional): Prefix to add to each saved frame filename.
            The final filename format will be: {prefix}{video_filename}_{frame_number}.jpg. Default "".

    Returns:
        int: Total number of successfully saved frames.

    Raises:
        AssertionError: If the specified video file does not exist.
    '''

    if frames_output_dir == "" :
        frames_output_dir = os.getcwd()

    # Get the video path and filename from the path
    video_dir, video_filename = os.path.split(video_path)  
    # Assert the video file exists
    assert os.path.exists(video_path)  

    # Open the video using OpenCV
    capture = cv2.VideoCapture(video_path)  

    # If start isn't specified lets assume 0
    if start_frame < 0:  
        start_frame = 0
    # if end isn't specified assume the end of the video
    if end_frame < 0:
        end_frame = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # Set the starting frame of the capture
    capture.set(1, start_frame)
    # Keep track of which frame we are up to, starting from start
    frame = start_frame
    # A safety counter to ensure we don't enter an infinite while loop (hopefully we won't need it)
    while_safety = 0
    # A count of how many frames we have saved
    saved_count = 0

    # Loop through the frames until the end
    while frame < end_frame:

        # Read an image from the capture
        _, image = capture.read()
        # Break the while if our safety maxs out at 500
        if while_safety > 500: 
            break

        # Skip in case of ''None' value read
        # Not saving in case of bad return
        if image is None:
            # Add 1
            while_safety += 1
            # skip
            continue

        # If this is a frame, write out based on the 'every' argument
        if frame % frame_interval == 0:
            # Reset the safety count
            while_safety = 0
            
            # variable 'path' creation
            path = os.path.join(frames_output_dir, Path(video_filename).stem)
            # Check whether the specified path exists or not
            if not os.path.exists(path):
               # Create a new directory because it does not exist
               os.makedirs(path)
    
            # Create the save path
            save_path = os.path.join(frames_output_dir,Path(video_filename).stem, f"{filename_prefix}{video_filename}_{frame:05d}.jpg")
            # If it doesn't exist or you want to overwrite anyways
            if not os.path.exists(save_path) or overwrite:
                # Save the extracted image
                cv2.imwrite(save_path, image)
                # Increment counter by one
                saved_count += 1

        # Increment frame count
        frame += 1  

    # After the while has finished close the capture
    capture.release()  

    # Return the count of the images we saved
    return saved_count
 
 

def get_video_properties(video_path: Union[str, Path]) -> tuple[float, int, int, int]:

    """
    Get video properties such as FPS, total frames, width, and height.
    - For .ts files, FPS is calculated manually using frame timestamps and rounded to the nearest integer.
    - For other formats, use OpenCV's default implementation.
    tuple: A tuple containing (fps, total_frames, width, height).
        float: The FPS (frames per second).
        int: The total number of frames.
        int: The width of the video.
        int: The height of the video.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # List of formats that require manual FPS calculation
    custom_formats = ['.ts']

    # Open the video
    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    # Get width and height (usually reliable)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # If the format is in the custom_formats list, manually calculate FPS and total_frames
    if video_path.suffix.lower() in custom_formats:
        print(f"Format {video_path.suffix} detected. Manually calculating FPS and total_frames...")
        total_frames = 0
        frame_timestamps = []  # Store frame timestamps to calculate FPS
        while True:
            ret, _ = video.read()
            if not ret:
                break
            total_frames += 1
            # Get the current frame's timestamp
            timestamp = video.get(cv2.CAP_PROP_POS_MSEC)  # Timestamp in milliseconds
            frame_timestamps.append(timestamp)

        # Manually calculate FPS using frame timestamps
        if len(frame_timestamps) > 1:
            time_diff = (frame_timestamps[-1] - frame_timestamps[0]) / 1000.0  # Time difference in seconds
            fps = total_frames / time_diff if time_diff > 0 else 0
        else:
            fps = 0

        # Round FPS to the nearest integer
        fps = round(fps)
    else:
        # Use OpenCV's default implementation for other formats
        print(f"Format {video_path.suffix} detected. Using video metadata...")
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)

    video.release()
    return fps, total_frames, width, height
 
# def get_video_properties(video_path: Union[str, Path]):
#     """
#     Get video properties such as FPS, total frames, width, and height.

#     Args:
#         video_path: Path to the video file.

#     Returns:
#         A tuple containing (fps, total_frames, width, height).
#     """
#     video = cv2.VideoCapture(str(video_path))
#     if not video.isOpened():
#         raise FileNotFoundError(f"Unable to open video: {video_path}")

#     fps = video.get(cv2.CAP_PROP_FPS)
#     total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
#     width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     video.release()
#     return fps, total_frames, width, height

def _infer_codec_from_filename(filename: Union[str, Path]) -> str:
    """
    Infer the video codec based on the file extension.

    Args:
        filename: Path to the output video file.

    Returns:
        A string representing the fourcc codec.
    """
    extension = Path(filename).suffix.lower()
    codec_mapping = {
        '.avi': 'XVID',
        '.mp4': 'mp4v',
        '.mov': 'avc1',
        '.mkv': 'X264',
        '.ts': 'MPEG',   
     }
    return codec_mapping.get(extension, 'XVID')   

def process_video(
    video_input: Union[str, Path],
    callback: Callable[[np.ndarray, int, int], np.ndarray],  # Updated callback signature
    video_output: Union[str, Path] = None,
    fps: int = None,
):
    """
    Process a video frame by frame, applying a callback to each frame.

    Args:
        video_input: Path to the input video.
        callback: Callback function that receives a frame (as a NumPy array), the frame number, and the total frame count.
        video_output: Path to the output video (optional).
        fps: FPS of the output video (if not specified, uses the same FPS as the input video).

    Returns:
        None
    """
    # Open the input video
    input_video = cv2.VideoCapture(str(video_input))
    if not input_video.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_input}")

    # Get video properties using the get_video_properties function
    input_fps, total_frames, width, height = get_video_properties(video_input)

    # Define the codec and create a VideoWriter object if an output video is specified
    if video_output:
        codec = _infer_codec_from_filename(video_output)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        output_fps = fps if fps is not None else input_fps
        out = cv2.VideoWriter(str(video_output), fourcc, output_fps, (width, height))
    else:
        out = None

    frame_number = 0
    while input_video.isOpened():
        ret, frame = input_video.read()
        if not ret:
            break

        # Apply the callback to the frame
        processed_frame = callback(frame, frame_number, total_frames)  # Pass total_frames to callback

        # If an output video is specified, write the processed frame
        if out is not None:
            out.write(processed_frame)

        frame_number += 1

    # Release everything when done
    input_video.release()
    if out is not None:
        out.release()