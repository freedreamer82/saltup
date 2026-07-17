import os
import queue
import threading
import cv2
import contextlib
import numpy as np
import re
import math
import struct
from pathlib import Path, PurePosixPath
import subprocess
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Union, List, Optional
from urllib.parse import urlparse
from saltup.utils.misc import is_url, extract_extension_from_url
from saltup.utils.data.image.image_utils import ColorMode, ColorsBGR, Image, ImageFormat


@contextlib.contextmanager
def _ffmpeg_capture_options(options: Optional[Dict[str, object]]):
    """Temporarily set OpenCV's FFmpeg demuxer options around a capture open.

    OpenCV's FFmpeg backend reads the ``OPENCV_FFMPEG_CAPTURE_OPTIONS`` env var
    when a ``VideoCapture`` is opened. We use it to pass ``ignore_editlist=1`` so
    the mov/mp4 demuxer ignores the edit list, mapping frame index N to the same
    picture a browser shows once the edit list is neutralized.

    The env var is process-global and only read at open time, so wrap **only**
    the ``cv2.VideoCapture(...)`` call. The previous value is restored on exit.
    """
    if not options:
        yield
        return
    key = "OPENCV_FFMPEG_CAPTURE_OPTIONS"
    previous = os.environ.get(key)
    os.environ[key] = "|".join(f"{k};{v}" for k, v in options.items())
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


@dataclass
class VideoReadOptions:
    """Decode-time options for reading a video (format-specific knobs).

    Kept separate from the core read API so the function signatures stay
    format-agnostic and new knobs can be added without churning callers.

    Attributes:
        ignore_edit_list: MP4/MOV only. Tell FFmpeg's mov demuxer to ignore the
            edit list (``ignore_editlist``) so frame index N maps to the same
            picture a browser shows once the edit list is neutralized.
    """
    ignore_edit_list: bool = False

    def to_ffmpeg_capture_options(self) -> Dict[str, object]:
        """Translate to FFmpeg demuxer options for OPENCV_FFMPEG_CAPTURE_OPTIONS."""
        opts: Dict[str, object] = {}
        if self.ignore_edit_list:
            opts["ignore_editlist"] = 1
        return opts


def _open_capture(source, options: Optional[VideoReadOptions]):
    """Open a ``cv2.VideoCapture`` honoring *options* (FFmpeg backend when set)."""
    ffmpeg_opts = options.to_ffmpeg_capture_options() if options else {}
    backend = cv2.CAP_FFMPEG if ffmpeg_opts else cv2.CAP_ANY
    with _ffmpeg_capture_options(ffmpeg_opts or None):
        return cv2.VideoCapture(source, backend)
    
@dataclass
class MotionDetectionOptions:
    """Tunable knobs for :func:`extract_quadrant_variance`.

    Every field defaults to the value previously hard-coded in the function,
    so passing ``None`` (or no config) reproduces the original behaviour.

    Attributes:
        pixel_k: Robust-sigma multiplier for the per-frame moving-pixel mask
            (median + ``pixel_k`` * MAD). Higher is stricter (fewer noise pixels).
        window_seconds: Temporal window (seconds) over which the per-pixel
            movement std is accumulated (EMA look-back).
        move_window_sec: Look-back (seconds) for the per-cell activity-centre
            shift that forms the movement signal.
        min_seconds: Minimum duration (seconds) an active run must last to be
            kept; shorter bursts are discarded.
        smooth_sec: Trailing-EMA smoothing window (seconds) for the movement
            signal (lower = less lag).
        move_threshold: Hysteresis ON threshold for the smoothed movement
            signal; the OFF threshold is ``0.5 * move_threshold``.
        start_s: Seek offset (seconds) into the video before processing.
        duration_s: Optional processing limit (seconds); ``None`` = whole video.
        resize_width: Width (px) the frame is downscaled to before analysis
            (height scaled to preserve aspect ratio).
        roi: Optional normalized ``(x_min, y_min, x_max, y_max)`` crop applied
            before resizing or any other frame preprocessing.
        store: If ``True``, keep per-frame RGB frames and std maps in memory.
        verbose: If ``True``, print periodic progress to stdout.
    """
    n_quadrants: int = 4
    pixel_k: float = 4.5
    window_seconds: float = 1.5
    move_window_sec: float = 1.0
    min_seconds: float = 0.0
    smooth_sec: float = 0.5
    move_threshold: float = 8.0
    start_s: float = 0.0
    duration_s: Optional[float] = None
    resize_width: int = 320
    store: bool = False
    verbose: bool = False
    fps_override: Optional[float] = None
    roi: Optional[Tuple[float, float, float, float]] = None

    # ------------------------------------------------------------------
    # High-level / GUI constructor
    # ------------------------------------------------------------------
    @classmethod
    def from_gui(cls, smoothing: float = 0.5, sensitivity: float = 0.5,
                 grid_quadrants: int = 4, *, start_s: float = 0.0,
                 duration_s: Optional[float] = None, resize_width: int = 320,
                 fps_override: Optional[float] = None, store: bool = False,
                 verbose: bool = False, roi: Optional[Tuple[float, float, float, float]] = None) -> "MotionDetectionOptions":
        """Build options from a few independent, high-level controls.

        The raw fields of :class:`MotionDetectionOptions` are coupled: the activity
        threshold is expressed as *% of the cell*, so it depends on the grid geometry,
        and the movement signal scales with the centroid-shift look-back, so it also
        depends on the temporal windows. This constructor hides those couplings behind
        three independent controls and derives every field consistently:

        * ``smoothing`` (0..1): temporal smoothing / lag. ``0`` = snappy & noisy,
          ``1`` = smooth & laggy. Drives ``window_seconds``, ``move_window_sec`` and
          ``smooth_sec``. Neutral ``0.5`` reproduces the class defaults.
        * ``sensitivity`` (0..1): ``0`` = strict (few detections), ``1`` = very sensitive
          (many detections). Drives the base threshold and the pixel-level gate (``pixel_k``).
        * ``grid_quadrants`` (int, 2..16): number of spatial cells; snapped to the
          range accepted by :func:`_compute_grid`.

        Absolute detection sensitivity is *preserved* across grid and smoothing
        changes: ``move_threshold`` is rescaled by the look-back window and by the cell
        size (a cell spans ``resize_width / cols`` of the image, where ``cols`` is the
        number of grid columns) so the same real-world motion still triggers.

        Calling with no arguments reproduces the default :class:`MotionDetectionOptions`
        exactly, i.e. ``MotionDetectionOptions.from_gui() == MotionDetectionOptions()``.
        """
        r = max(0.0, min(1.0, float(smoothing)))
        s = max(0.0, min(1.0, float(sensitivity)))
        nq = max(_MIN_QUADRANTS, min(_MAX_QUADRANTS, int(round(grid_quadrants))))

        # smoothing -> temporal windows (neutral 0.5 == class defaults).
        window_seconds = _lerp(0.5, 2.5, r)
        move_window_sec = _lerp(0.3, 1.7, r)
        smooth_sec = _lerp(0.2, 0.8, r)

        # Sensitivity -> base threshold + pixel gate (neutral 0.5 == class defaults).
        # Higher sensitivity = lower threshold / looser pixel gate = MORE detections.
        t0 = _lerp(14.0, 2.0, s)
        pixel_k = _lerp(7.0, 2.0, s)

        # Preserve absolute sensitivity: scale the threshold by the look-back window and
        # by the cell size so the same real motion triggers regardless of grid/smoothing.
        # The movement signal is built from the per-cell *normalised* centroid, so for a
        # fixed absolute image-space horizontal move it scales with the number of grid
        # columns (a cell spans 1/cols of the image width).  To keep the same real motion
        # triggering, the threshold must scale with cols (not sqrt(n_quadrants), which only
        # matches square grids 4/9/16 and under-scales every rectangular grid).
        _, cols = _compute_grid(nq)
        cols_4 = _compute_grid(4)[1]  # grid=4 reference used by the neutral defaults
        move_threshold = t0 * (move_window_sec / 1.0) \
            * (float(cols) / float(cols_4)) \
            * (320.0 / float(resize_width))

        return cls(
            n_quadrants=nq,
            pixel_k=round(pixel_k, 2),
            window_seconds=round(window_seconds, 3),
            move_window_sec=round(move_window_sec, 3),
            min_seconds=0.0,
            smooth_sec=round(smooth_sec, 3),
            move_threshold=round(move_threshold, 3),
            start_s=start_s,
            duration_s=duration_s,
            resize_width=int(resize_width),
            store=store,
            verbose=verbose,
            fps_override=fps_override,
            roi=roi
        )


def _lerp(a: float, b: float, t: float) -> float:
    """Linearly interpolate ``a`` -> ``b`` for ``t`` clamped to [0, 1]."""
    t = max(0.0, min(1.0, float(t)))
    return float(a + (b - a) * t)

# =============================================================================
# Module Constants
# =============================================================================

_MIN_QUADRANTS = 2
_MAX_QUADRANTS = 16


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
    if first_image is None:
        raise FileNotFoundError(f"Unable to read image: {image_files[0]}")
    height, width, _ = first_image.shape

    # Create a VideoWriter object with the specified output filename and FPS
    fourcc = cv2.VideoWriter.fourcc(*'MJPG')
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
 
 

@dataclass
class VideoProperties:
    fps: int
    total_frames: int
    width: int
    height: int

    def __iter__(self):
        """Support tuple unpacking: fps, total_frames, width, height = props."""
        yield self.fps
        yield self.total_frames
        yield self.width
        yield self.height


def get_video_properties(video_path: Union[str, Path], max_seconds: float = 15, *, options: Optional[VideoReadOptions] = None) -> VideoProperties:

    """
    Get video properties such as FPS, total frames, width, and height.
    Supports both local file paths and HTTP/HTTPS URLs (e.g. S3 presigned URLs).
    OpenCV uses FFmpeg internally, so URLs are opened directly without downloading.
    - For .ts files (local or remote), FPS is calculated manually using frame timestamps.
    - For other formats, use OpenCV's default implementation.

    Args:
        video_path: Local file path or HTTP/HTTPS URL (e.g. S3 presigned URL).
        max_seconds: Window (in seconds) used to sample frames when computing
            real FPS from PTS deltas (e.g. ``.ts`` format).  A value ``<= 0``
            uses a fixed 60-frame sample (default).  Positive values sample
            at most ``fps * max_seconds`` frames (minimum 60) so the full
            file is never downloaded.  Total frames are always estimated from
            container metadata (duration × real FPS), not by reading every frame.

    Returns:
        tuple: A tuple containing (fps, total_frames, width, height).
            float: The FPS (frames per second).
            int: The total number of frames (or frames counted within *max_seconds*
                when scanning is limited).
            int: The width of the video.
            int: The height of the video.
    """
    _is_url = is_url(video_path)

    if _is_url:
        video_source = video_path
    else:
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        video_source = str(video_path)

    # List of formats that require manual FPS calculation
    custom_formats = ['.ts']
    total_frames = 0

    # Open the video (OpenCV uses FFmpeg internally, supports both files and URLs).
    # options may force the FFmpeg backend with demuxer flags (e.g. ignore_editlist),
    # so total_frames/duration match the neutralized timeline.
    video = _open_capture(video_source, options)
    if not video.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    # Get width and height (usually reliable)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Determine the file suffix.
    # For presigned URLs, the path may be extension-less; in that case,
    # infer extension from response-content-disposition filename query param.
    if _is_url:
        suffix = extract_extension_from_url(str(video_path))
    else:
        suffix = Path(video_path).suffix.lower()

    # Millisecond budget for frame-scanning modes (None = unlimited)
    limit_ms = max_seconds * 1000.0 if max_seconds > 0 else None

    # If the format is in the custom_formats list, manually calculate FPS and total_frames
    if suffix in custom_formats:
        fps_container = video.get(cv2.CAP_PROP_FPS)
        fc_container  = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        if max_seconds <= 0:
            # Full scan: read every frame for an accurate count and precise FPS.
            total_frames = 0
            frame_timestamps = []
            while True:
                ret, _ = video.read()
                if not ret:
                    break
                total_frames += 1
                frame_timestamps.append(video.get(cv2.CAP_PROP_POS_MSEC))
        else:
            # Partial scan: sample at most fps_container * max_seconds frames
            # (minimum 60) to compute real FPS, then estimate total_frames from
            # container duration so the full file is never downloaded.
            fallback_fps = fps_container if fps_container > 0 else 25.0
            sample_count = max(60, int(fallback_fps * max_seconds))
            frame_timestamps = []
            for _ in range(sample_count):
                ret, _ = video.read()
                if not ret:
                    break
                frame_timestamps.append(video.get(cv2.CAP_PROP_POS_MSEC))

        # Compute real FPS from PTS deltas (common to both paths)
        fps = 0
        if len(frame_timestamps) > 1:
            deltas = [
                frame_timestamps[i + 1] - frame_timestamps[i]
                for i in range(len(frame_timestamps) - 1)
                if frame_timestamps[i + 1] > frame_timestamps[i]
            ]
            if deltas:
                avg_ms = sum(deltas) / len(deltas)
                fps = round(1000.0 / avg_ms) if avg_ms > 0 else 0

        # Fall back to container FPS if PTS-based calculation failed
        if fps == 0 and fps_container > 0:
            fps = round(fps_container)

        # For partial scans estimate total_frames from container metadata
        # so we report the real duration, not just the sampled window.
        if max_seconds > 0:
            if fc_container > 0 and fps_container > 0:
                duration = fc_container / fps_container
                total_frames = int(duration * fps) if fps > 0 else fc_container
            else:
                total_frames = len(frame_timestamps)
        # For full scans total_frames was already counted in the loop above.
    else:
        # Use OpenCV's default implementation for other formats
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = round(video.get(cv2.CAP_PROP_FPS))

        # Fallback for streams where the container reports no frame count
        if total_frames <= 0 and fps > 0 and limit_ms is not None:
            # Count frames up to max_seconds by reading
            frame_timestamps = []
            while True:
                ret, _ = video.read()
                if not ret:
                    break
                total_frames += 1
                timestamp = video.get(cv2.CAP_PROP_POS_MSEC)
                frame_timestamps.append(timestamp)
                if timestamp >= limit_ms:
                    break
            # Refine fps from real PTS deltas if we have enough samples
            if len(frame_timestamps) > 1:
                deltas = [
                    frame_timestamps[i + 1] - frame_timestamps[i]
                    for i in range(len(frame_timestamps) - 1)
                    if frame_timestamps[i + 1] > frame_timestamps[i]
                ]
                if deltas:
                    avg_ms = sum(deltas) / len(deltas)
                    fps = round(1000.0 / avg_ms) if avg_ms > 0 else fps

    video.release()
    return VideoProperties(fps=fps, total_frames=total_frames, width=width, height=height)

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
    callback: Optional[Callable[[Image, int, int, Optional[VideoProperties]], Image]] = None,
    video_output: Optional[Union[str, Path]] = None,
    metadata: Optional[VideoProperties] = None,
    frame_numbers: Optional[List[int]] = None,
    *,
    options: Optional[VideoReadOptions] = None,
) -> VideoProperties:
    """
    Process a video frame by frame, applying a callback to each frame.

    Args:
        video_input: Path to the input video.
        callback: Callback function that receives a frame (as Image), frame number, and total frame count.
        video_output: Path to the output video (optional).
        metadata: VideoProperties object containing video metadata (if not specified, it will be inferred).
        frame_numbers: List of specific frame numbers to process (e.g., [0, 10, 20, 150]).
                      If None, processes all frames sequentially.
        options: VideoReadOptions for decode-time knobs (e.g. ignore_edit_list).
                      When metadata is not supplied, the inferred properties are
                      read with the same options.

    Returns:
        VideoProperties
    """
    # Open the input video (options may force FFmpeg + ignore_editlist).
    input_video = _open_capture(str(video_input), options)
    if not input_video.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_input}")

    # Get video properties
    if metadata is None:
        metadata = get_video_properties(video_input, options=options)
    input_fps, total_frames, width, height = metadata.fps, metadata.total_frames, metadata.width, metadata.height
    
    # Setup output video if specified
    if video_output:
        codec = _infer_codec_from_filename(video_output)
        fourcc = cv2.VideoWriter.fourcc(*codec)
        output_fps = metadata.fps if metadata.fps is not None else input_fps
        out = cv2.VideoWriter(str(video_output), fourcc, output_fps, (width, height))
    else:
        out = None
 
    video_input_str = str(video_input)
    if is_url(video_input_str):
        input_suffix = extract_extension_from_url(video_input_str)
    else:
        input_suffix = Path(video_input_str).suffix.lower()
    is_ts_input = input_suffix == '.ts'
    
    if frame_numbers is not None:
        # Selective mode: use cheap frame grabs; only do coarse seeks on seek-friendly formats.
        frames_to_process = sorted(set(frame_numbers))
        current_frame = 0
        seek_gap_threshold = 300
        seek_backoff_frames = 30
 
        for frame_number in frames_to_process:
            if frame_number < 0:
                continue
            if frame_number >= total_frames:
                break
 
            gap = frame_number - current_frame
 
            # Hybrid strategy for large gaps: coarse seek near target, then grab forward.
            if not is_ts_input and gap > seek_gap_threshold:
                seek_to = max(0, frame_number - seek_backoff_frames)
                if input_video.set(cv2.CAP_PROP_POS_FRAMES, seek_to):
                    current_frame = seek_to
 
            # Advance to target by grabbing packets without decoding frame pixels.
            while current_frame < frame_number:
                if not input_video.grab():
                    break
                current_frame += 1
 
            # Stream ended while seeking forward.
            if current_frame != frame_number:
                break
 
            ret, frame = input_video.read()
            if not ret:
                break
 
            # We just consumed frame_number.
            current_frame += 1
 
            # Apply callback
            if callback:
                processed_frame = callback(Image(frame), frame_number, total_frames, metadata)
            else:
                processed_frame = Image(frame)

            # Write to output if specified
            if out is not None:
                out.write(processed_frame.get_data())
    else:
        # SEQUENTIAL MODE: process all frames (original behavior)
        frame_number = 0
        while input_video.isOpened():
            ret, frame = input_video.read()
            if not ret:
                break
            
            # Apply callback
            if callback:
                processed_frame = callback(Image(frame), frame_number, total_frames, metadata)
            else:
                processed_frame = Image(frame)

            # Write to output if specified
            if out is not None:
                out.write(processed_frame.get_data())
            
            frame_number += 1
    
    # Cleanup
    input_video.release()
    if out is not None:
        out.release()
    
    return metadata


# =============================================================================
# Frame Preprocessing
# =============================================================================

def preprocess_frame(
    frame: np.ndarray,
    resize: Optional[Tuple[int, int]] = None,
    gray: bool = True,
    blur: Optional[Tuple[int, int]] = None,
    normalize: bool = False,
    roi: Optional[Tuple[float, float, float, float]] = None,
) -> np.ndarray:
    """Apply a configurable preprocessing pipeline to a single video frame.

    Operations are applied in order: ROI crop → resize → grayscale → blur
    → normalize.
    Each step is optional and controlled by its corresponding argument.

    Args:
        frame: Input frame as a BGR NumPy array (H×W×3).
        resize: Target ``(width, height)`` for resizing.  ``None`` skips
            resizing.  Defaults to ``None``.
        gray: If ``True``, convert the frame to single-channel grayscale.
            Defaults to ``True``.
        blur: Gaussian blur kernel size as ``(kW, kH)`` (both must be odd).
            ``None`` skips blurring.  Defaults to ``None``.
        normalize: If ``True``, apply z-score normalization rescaled to
            the 0–255 uint8 range (mean ≈ 128, std ≈ 32).  Useful for
            reducing sensitivity to global illumination changes.
            Defaults to ``False``.
        roi: Normalized coordinates ``(x_min, y_min, x_max, y_max)`` of the region of interest.
            If specified, only this region will be processed. Defaults to ``None``.

    Returns:
        The preprocessed frame as a NumPy array.

    Examples:
        >>> import cv2, numpy as np
        >>> frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        >>> gray_small = preprocess_frame(frame, resize=(320, 240), gray=True)
        >>> gray_small.shape
        (240, 320)
    """
    # Crop first so every subsequent operation only processes the ROI.
    if roi is not None:
        x1, y1, x2, y2 = roi
        h, w = frame.shape[:2]

        x1 = int(round(x1 * w))
        x2 = int(round(x2 * w))
        y1 = int(round(y1 * h))
        y2 = int(round(y2 * h))

        x1 = max(0, min(x1, w - 1))
        x2 = max(x1 + 1, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(y1 + 1, min(y2, h))

        frame = frame[y1:y2, x1:x2]
        
    if resize is not None:
        frame = cv2.resize(frame, resize, interpolation=cv2.INTER_LINEAR)
    if gray:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if blur is not None:
        frame = cv2.GaussianBlur(frame, blur, 0)
    if normalize:
        frame_f = frame.astype(np.float32)
        mean = float(np.mean(frame_f))
        std = float(np.std(frame_f))
        if std > 1e-6:
            norm = (frame_f - mean) / std
            frame = np.clip(128.0 + 32.0 * norm, 0, 255).astype(np.uint8)
    return frame


# =============================================================================
# Quadrant Grid Helpers
# =============================================================================


def _compute_grid(n_quadrants: int) -> Tuple[int, int]:
    """Find the most square-like (rows, cols) grid for *n_quadrants*.

    The function returns the pair ``(rows, cols)`` where
    ``rows * cols == n_quadrants`` and the difference ``|rows - cols|``
    is minimised.

    Args:
        n_quadrants: Desired number of spatial regions (2–16).

    Returns:
        ``(rows, cols)`` tuple.

    Raises:
        ValueError: If *n_quadrants* is outside the 2–16 range.
    """
    if not (_MIN_QUADRANTS <= n_quadrants <= _MAX_QUADRANTS):
        raise ValueError(
            f"n_quadrants must be between {_MIN_QUADRANTS} and "
            f"{_MAX_QUADRANTS}, got {n_quadrants}."
        )
    best = (1, n_quadrants)
    for r in range(1, int(n_quadrants ** 0.5) + 1):
        if n_quadrants % r == 0:
            c = n_quadrants // r
            if abs(r - c) < abs(best[0] - best[1]):
                best = (r, c)
    return best


def _quadrant_names(n_quadrants: int) -> List[str]:
    """Return a list of canonical quadrant key names.

    Keys are ``'quadrant_1'`` … ``'quadrant_N'`` where *N* equals
    *n_quadrants*.  The ordering follows a **row-major, bottom-to-top**
    scan: the first quadrant is the bottom-right cell of the grid, and
    the last is the top-left cell.  For ``n_quadrants=4`` this matches
    the legacy numbering.
    """
    return [f"quadrant_{i + 1}" for i in range(n_quadrants)]

# =============================================================================
# Quadrant Intensity Analysis
# =============================================================================

def motion_detection(
    path: str,
    config: Optional[MotionDetectionOptions] = None,
    *,
    options: Optional[VideoReadOptions] = None,
) -> Tuple[Dict[int, Dict[str, float]], np.ndarray]:
    """
    Variant of extract_quadrant_variance that reads frames using process_video
    and a callback. This version restricts processing to config.start_s/duration_s
    by passing explicit frame_numbers to process_video and adds a producer/consumer
    worker so decoding (process_video) runs concurrently with the per-frame CPU work.
    """
    if config is None:
        config = MotionDetectionOptions()

    def cell_slices(H, W):
        ys = np.linspace(0, H, rows + 1).round().astype(int)
        xs = np.linspace(0, W, cols + 1).round().astype(int)
        return [(ys[r], ys[r + 1], xs[c], xs[c + 1])
                for r in range(rows) for c in range(cols)]

    def moving_mask(m, pixel_k):
        med = np.median(m); mad = np.median(np.abs(m - med)) * 1.4826 + 1e-6
        return m > (med + pixel_k * mad)

    n_quadrants = config.n_quadrants
    rows, cols = _compute_grid(n_quadrants)
    names = _quadrant_names(n_quadrants)
    pixel_k = config.pixel_k
    window_seconds = config.window_seconds
    move_window_sec = config.move_window_sec
    min_seconds = config.min_seconds
    smooth_sec = config.smooth_sec
    move_threshold = config.move_threshold
    start_s = config.start_s
    duration_s = config.duration_s
    resize_width = config.resize_width
    store = config.store
    verbose = config.verbose
    fps_override = config.fps_override
    roi = config.roi

    # Get metadata and compute frame range to process
    metadata = get_video_properties(path, options=options)
    fps = float(fps_override) if fps_override else float(metadata.fps or 25.0)
    total_frames = int(metadata.total_frames)
    N = max(2, int(round(window_seconds * fps)))
    alpha = 1.0 / N

    start_frame = int(round(start_s * fps)) if start_s and start_s > 0 else 0
    if start_frame < 0:
        start_frame = 0
    if duration_s and duration_s > 0:
        end_frame = min(total_frames, start_frame + int(round(duration_s * fps)))
    else:
        end_frame = total_frames

    # Guard the range
    if start_frame >= end_frame:
        # nothing to process
        return {}, np.array([])

    frames_to_process = list(range(start_frame, end_frame))

    # Shared mutable state consumed by the worker
    mean_acc = None
    sq_acc = None
    frames_rgb: List[np.ndarray] = []
    std_maps: List[np.ndarray] = []
    centroids: List[List[Tuple[float, float]]] = []
    ncent: List[List[Tuple[float, float]]] = []
    _openk = np.ones((3, 3), np.uint8)
    frame_counter = 0

    q = queue.Queue(maxsize=8)

    def worker():
        nonlocal mean_acc, sq_acc, frame_counter
        while True:
            item = q.get()
            if item is None:
                q.task_done()
                break
            frame, frame_number = item
            try:
                if roi is not None:
                    # if ROI is specified, crop the frame before resizing and processing
                    frame = preprocess_frame(frame, gray=False, roi=roi)

                h0, w0 = frame.shape[:2]
                h = int(round(h0 * resize_width / w0)) if w0 > 0 else resize_width
                small = cv2.resize(frame, (resize_width, h), interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)

                # brightness compensation per cell
                for (y0, y1, x0, x1) in cell_slices(*gray.shape):
                    gray[y0:y1, x0:x1] -= gray[y0:y1, x0:x1].mean()

                # temporal movement EMA
                if mean_acc is None:
                    mean_acc = gray.copy(); sq_acc = gray * gray
                else:
                    cv2.accumulateWeighted(gray, mean_acc, alpha)
                    cv2.accumulateWeighted(gray * gray, sq_acc, alpha)
                std_map = np.sqrt(np.clip(sq_acc - mean_acc * mean_acc, 0, None))

                mask = moving_mask(std_map, pixel_k)
                mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, _openk).astype(bool)

                cnorm = [(np.nan, np.nan)] * n_quadrants
                cframe = [(np.nan, np.nan)] * n_quadrants
                for qi, (y0, y1, x0, x1) in enumerate(cell_slices(*mask.shape)):
                    m = mask[y0:y1, x0:x1]; tot = float(m.sum()); qh, qw = y1 - y0, x1 - x0
                    if tot < max(15.0, 0.015 * qh * qw):
                        continue
                    ex = float((np.arange(qw) * m.sum(0)).sum() / tot)
                    ey = float((np.arange(qh) * m.sum(1)).sum() / tot)
                    cnorm[qi] = (ex / qw, ey / qh)
                    cframe[qi] = (x0 + ex, y0 + ey)

                ncent.append(cnorm)
                centroids.append(cframe)
                if store:
                    frames_rgb.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
                    std_maps.append(std_map.astype(np.float16))

                frame_counter += 1
                if verbose and frame_counter % 300 == 0:
                    print(f"  processed {frame_counter} frames...", end="\r")
            finally:
                q.task_done()

    # Start worker thread
    w = threading.Thread(target=worker)
    w.start()

    # Callback just enqueues raw frames so decoding can run in parallel with processing
    def _enqueue_callback(img: Image, frame_number: int, total_frames_cb: int, meta: Optional[VideoProperties]):
        arr = img.get_data()
        # put blocks when queue is full, providing backpressure to decoder
        q.put((arr, frame_number))
        return img

    # Run process_video in selective mode for the desired frame numbers
    process_video(path, callback=_enqueue_callback, metadata=metadata, frame_numbers=frames_to_process, options=options)

    # Signal worker end and wait for completion
    q.put(None)
    w.join()

    if verbose:
        print(f"Done. {frame_counter} frames @ {fps:.1f} fps  (window N = {N} = {N/fps:.2f}s)")

    # If no centres were recorded, return empty results
    if len(ncent) == 0:
        return {}, np.array([])

    # Post-process movement signals same as original
    nc = np.asarray(ncent, dtype=np.float32).reshape(-1, n_quadrants, 2)
    Wm = max(1, int(round(move_window_sec * fps)))
    signal = np.zeros((len(nc), n_quadrants), dtype=np.float32)
    for qi in range(n_quadrants):
        a = nc[:, qi]; b = np.roll(a, Wm, axis=0); b[:Wm] = np.nan
        d = 100.0 * np.hypot(a[:, 0] - b[:, 0], a[:, 1] - b[:, 1])
        d[np.isnan(d)] = 0.0
        signal[:, qi] = d

    T, Q = signal.shape
    ws = max(1, int(round(smooth_sec * fps))); a_s = 1.0 / ws
    min_len = int(round(min_seconds * fps))
    hi = float(move_threshold); lo = 0.5 * hi

    smooth = np.empty_like(signal)
    for qi in range(Q):
        e = float(signal[0, qi])
        for i in range(T):
            e += a_s * (float(signal[i, qi]) - e); smooth[i, qi] = e

    active = np.zeros((T, Q), dtype=bool)
    min_run = max(min_len, 1)
    for qi in range(Q):
        sm = smooth[:, qi]; on = False; raw = np.zeros(T, dtype=bool)
        for i in range(N, T):
            if on and sm[i] < lo:
                on = False
            elif (not on) and sm[i] > hi:
                on = True
            raw[i] = on
        i = N
        while i < T:
            if raw[i]:
                j = i
                while j < T and raw[j]:
                    j += 1
                if j - i >= min_run:
                    active[i:j, qi] = True
                i = j
            else:
                i += 1

    threshold = np.full(Q, hi)
    energies: Dict[int, Dict[str, float]] = {
        i: {
            name: float(smooth[i, ci])
            for ci, name in enumerate(names)
            if active[i, ci]
        }
        for i in range(T)
        if any(active[i])
    }
    return energies, threshold
 
# =============================================================================
# Windowed Segment Aggregation & Filtering
# =============================================================================

def compute_windowed_segment_stats(
    signal: np.ndarray,
    segments: List[Tuple[float, float]],
    fps: float,
    window_size_seconds: float = 10.0,
    agg_fn: Optional[Callable[[np.ndarray], float]] = None,
) -> List[np.ndarray]:
    """Downsample a per-frame signal within each time segment.

    For every ``(start, end)`` segment the function slices the
    corresponding frames from *signal*, splits them into
    non-overlapping windows of *window_size_seconds*, and reduces each
    window to a single scalar via *agg_fn*.

    This is a generic building block — it knows nothing about quadrants
    or variance; it simply aggregates any 1-D per-frame signal over
    time windows.

    Args:
        signal: 1-D NumPy array of per-frame values (length = total
            number of frames in the source video).
        segments: List of ``(start_seconds, end_seconds)`` tuples
            defining the time intervals to process.
        fps: Video frame rate (frames per second).
        window_size_seconds: Duration (seconds) of each non-overlapping
            aggregation window.  Defaults to 10.0.
        agg_fn: Callable that reduces a 1-D window array to a single
            float.  Receives an ``np.ndarray`` and must return a float.
            Defaults to ``np.mean`` when ``None``.

    Returns:
        A list of 1-D NumPy arrays (one per segment) containing the
        aggregated values.

    Examples:
        >>> import numpy as np
        >>> # 300 frames at 30 fps = 10 seconds of data
        >>> signal = np.random.rand(300)
        >>> segments = [(0.0, 10.0)]
        >>> result = compute_windowed_segment_stats(signal, segments, fps=30, window_size_seconds=5.0)
        >>> len(result)          # one segment
        1
        >>> result[0].shape[0]   # 10s / 5s = 2 windows
        2
    """
    if agg_fn is None:
        agg_fn = lambda w: float(np.mean(w))

    window_size_frames = max(1, int(window_size_seconds * fps))
    results: List[np.ndarray] = []

    for start, end in segments:
        start_frame = int(start * fps)
        end_frame = min(int(end * fps) + 1, len(signal))

        segment_data = signal[start_frame:end_frame]
        if len(segment_data) == 0:
            continue

        aggregated: List[float] = []
        for win_start in range(0, len(segment_data), window_size_frames):
            window = segment_data[win_start : win_start + window_size_frames]
            if len(window) > 0:
                aggregated.append(agg_fn(window))

        results.append(np.array(aggregated))

    return results


def filter_segments_by_std(
    segments: List[Tuple[float, float]],
    segment_signals: List[np.ndarray],
    std_threshold: float = 0.01,
) -> Tuple[List[Tuple[float, float]], List[np.ndarray]]:
    """Discard segments whose associated signal has low variability.

    Pairs of ``(segment, signal)`` are dropped when the standard
    deviation of *signal* falls below *std_threshold*.  This is useful
    for removing near-constant regions that are unlikely to contain
    meaningful activity.

    Args:
        segments: Time segments as ``(start, end)`` tuples.
        segment_signals: One 1-D NumPy array per segment (e.g. output
            of :func:`compute_windowed_segment_stats`).
        std_threshold: Minimum standard deviation to keep a segment.
            Defaults to 0.01.

    Returns:
        A 2-tuple ``(kept_segments, kept_signals)``.

    Examples:
        >>> import numpy as np
        >>> segs = [(0, 10), (20, 30)]
        >>> sigs = [np.array([0.5, 0.5, 0.5]), np.array([0.1, 0.9, 0.3])]
        >>> kept_segs, kept_sigs = filter_segments_by_std(segs, sigs, std_threshold=0.1)
        >>> len(kept_segs)  # first segment (constant) is dropped
        1
    """
    kept_segments: List[Tuple[float, float]] = []
    kept_signals: List[np.ndarray] = []

    for i, sig in enumerate(segment_signals):
        if np.std(sig) >= std_threshold:
            kept_segments.append(segments[i])
            kept_signals.append(sig)

    return kept_segments, kept_signals

# =============================================================================
# Header Parsing Helpers
# =============================================================================

from saltup.utils.data.image.image_utils import FileExtensionType

def get_header(path: Union[str, Path]) -> bytes:
    """
    Reads the first 256 KB of a file for header parsing.
    This is sufficient for formats like WAV, FLAC, and MP3 which have headers within this range.
    """
    file_path = Path(path)
    extension_name = file_path.suffix.lower().lstrip(".")

    read_sizes = {
        "micro": 64 * 1024,
        "small": 2000 * 1024,
        "medium": 5000 * 1024,
    }

    first_64kb_ext = {
        FileExtensionType.WMV,
        FileExtensionType.FLV,
        
    }

    first_2mb_ext = {
        FileExtensionType.AVI,
        FileExtensionType.MKV,
        FileExtensionType.WEBM,
    }
    first_5mb_ext = {
        FileExtensionType.MP4,
        FileExtensionType.MOV,
        FileExtensionType.GP,
        FileExtensionType.M3U8
    }

    try:
        extension = FileExtensionType(extension_name)
    except ValueError:
        extension = None

    if extension in first_64kb_ext:
        with open(file_path, "rb") as file:
            return file.read(read_sizes["micro"])

    if extension in first_2mb_ext:
        with open(file_path, "rb") as file:
            return file.read(read_sizes["small"])

    if extension in first_5mb_ext:
        with open(file_path, "rb") as file:
            return file.read(read_sizes["medium"])

    with open(file_path, "rb") as file:
        return file.read(4)


def get_tail(path: Union[str, Path]) -> bytes:
    """
    Reads the last 256 KB of a file for footer parsing.
    This is useful for formats that may have important metadata at the end of the file.
    """
    file_path = Path(path)
    extension_name = file_path.suffix.lower().lstrip(".")

    read_sizes = {
        "medium": 5000 * 1024,
    }
    last_5mb_ext = {
        FileExtensionType.MP4,
        FileExtensionType.MOV,
    }

    try:
        extension = FileExtensionType(extension_name)
    except ValueError:
        extension = None

    if extension in last_5mb_ext:
        with open(file_path, "rb") as file:
            file_size = file.seek(0, os.SEEK_END)
            file.seek(-min(read_sizes["medium"], file_size), os.SEEK_END)
            return file.read()

    with open(file_path, "rb") as file:
        file_size = file.seek(0, os.SEEK_END)
        file.seek(-min(4, file_size), os.SEEK_END)
        return file.read(4)

def parse_avi_header(header: bytes) -> dict:
    """
    Parses the header of an AVI file to extract metadata such as format, resolution, and duration.
    This is a simplified parser that looks for specific byte patterns in the header.
    """
    metadata = {
        "format": "AVI",
        "width": None,
        "height": None,
        "fps": None,
        "bit_depth": None,
    }

    # AVI is a RIFF container with 'AVI ' as form type
    if len(header) >= 12 and header[0:4] == b'RIFF' and header[8:12] == b'AVI ':
        metadata["format"] = "AVI"
        # try to find 'avih' chunk and extract dwWidth/dwHeight
        idx = header.find(b'avih')
        if idx != -1 and idx + 8 + 40 <= len(header):
            # avih: 4 bytes size after 'avih', then data; width at offset 32 within data
            data_start = idx + 8
            try:
                # dwMicroSecPerFrame (uSec per frame) at offset 0
                dwMicroSecPerFrame = int.from_bytes(header[data_start + 0:data_start + 4], 'little')
                if dwMicroSecPerFrame > 0:
                    metadata["fps"] = 1_000_000.0 / float(dwMicroSecPerFrame)

                # dwTotalFrames at offset 16 (total frames in file)
                try:
                    total_frames = int.from_bytes(header[data_start + 16:data_start + 20], 'little')
                    metadata["total_frames"] = total_frames
                except Exception:
                    pass

                width = int.from_bytes(header[data_start + 32:data_start + 36], 'little')
                height = int.from_bytes(header[data_start + 36:data_start + 40], 'little')
                metadata["width"] = width
                metadata["height"] = height
            except Exception:
                pass
        return metadata

    return {"format": "AVI", "error": "Invalid AVI/RIFF header"}


def parse_mp4_header(header: bytes) -> dict:
    """
    Parses the header of an MP4 file to extract metadata such as format, resolution, and duration.
    This is a simplified parser that looks for specific byte patterns in the header.
    """
    metadata = {
        "format": "MP4",
        "width": None,
        "height": None,
        "fps": None,
        "bit_depth": None,
        "duration": None,
    }

    # MP4 files start with 'ftyp' box within the first few bytes
    if b'ftyp' not in header:
        return {"format": "MP4", "error": "Invalid MP4 signature"}

    metadata["format"] = "MP4"

    # Try to find 'moov' box which contains metadata; it may not be in the header if it's at the end of the file
    if b'moov' in header:
        # This is still a naive approach; proper MP4 parsing requires box sizes and nesting
        moov_idx = header.find(b'moov')
        if moov_idx != -1:
            # Look for 'mvhd' box inside 'moov' which contains duration and timescale
            mvhd_idx = header.find(b'mvhd', moov_idx)
            if mvhd_idx != -1 and mvhd_idx + 8 <= len(header):
                try:
                    # mvhd version is at mvhd_idx + 4 (mvhd type + fullbox version)
                    version = header[mvhd_idx + 4]
                    if version == 0 and mvhd_idx + 24 <= len(header):
                        # For mvhd v0: timescale @ +16, duration @ +20 (from type offset)
                        timescale = int.from_bytes(header[mvhd_idx + 16:mvhd_idx + 20], 'big')
                        duration = int.from_bytes(header[mvhd_idx + 20:mvhd_idx + 24], 'big')
                    elif version == 1 and mvhd_idx + 40 <= len(header):
                        # For mvhd v1: timescale @ +28, duration @ +32 (64-bit)
                        timescale = int.from_bytes(header[mvhd_idx + 28:mvhd_idx + 32], 'big')
                        duration = int.from_bytes(header[mvhd_idx + 32:mvhd_idx + 40], 'big')
                    else:
                        timescale = None
                        duration = None

                    if timescale and duration is not None:
                        metadata["duration"] = float(duration) / float(timescale)
                except Exception:
                    pass

            # Look for 'trak' boxes which contain track info; we want the video track
            trak_idx = header.find(b'trak', moov_idx)
            while trak_idx != -1:
                next_trak_idx = header.find(b'trak', trak_idx + 4)
                trak_end = next_trak_idx if next_trak_idx != -1 else len(header)

                # Look for 'tkhd' box inside 'trak' which contains width/height
                tkhd_idx = header.find(b'tkhd', trak_idx)
                if tkhd_idx != -1 and tkhd_idx + 8 <= len(header):
                    try:
                        version = header[tkhd_idx + 4]
                        if version == 0 and tkhd_idx + 88 <= len(header):
                            # tkhd_idx points to box type ('tkhd'), so width/height are at +80/+84
                            width = int.from_bytes(header[tkhd_idx + 80:tkhd_idx + 84], 'big') >> 16
                            height = int.from_bytes(header[tkhd_idx + 84:tkhd_idx + 88], 'big') >> 16
                        elif version == 1 and tkhd_idx + 100 <= len(header):
                            # version 1 has larger timestamps; width/height shift by +12 bytes
                            width = int.from_bytes(header[tkhd_idx + 92:tkhd_idx + 96], 'big') >> 16
                            height = int.from_bytes(header[tkhd_idx + 96:tkhd_idx + 100], 'big') >> 16
                        else:
                            width = None
                            height = None

                        if width and height:
                            metadata["width"] = width
                            metadata["height"] = height
                    except Exception:
                        pass

                # Try to compute FPS from track timing (mdhd) and sample count (stsz)
                if metadata.get("fps") is None:
                    mdhd_idx = header.find(b'mdhd', trak_idx, trak_end)
                    stsz_idx = header.find(b'stsz', trak_idx, trak_end)
                    if mdhd_idx != -1 and stsz_idx != -1:
                        try:
                            track_timescale = None
                            track_duration = None

                            # mdhd version is at mdhd_idx + 4
                            mdhd_version = header[mdhd_idx + 4] if mdhd_idx + 5 <= len(header) else None
                            if mdhd_version == 0 and mdhd_idx + 24 <= len(header):
                                # mdhd v0: timescale @ +16, duration @ +20 (from type offset)
                                track_timescale = int.from_bytes(header[mdhd_idx + 16:mdhd_idx + 20], 'big')
                                track_duration = int.from_bytes(header[mdhd_idx + 20:mdhd_idx + 24], 'big')
                            elif mdhd_version == 1 and mdhd_idx + 40 <= len(header):
                                # mdhd v1: timescale @ +28, duration @ +32 (64-bit)
                                track_timescale = int.from_bytes(header[mdhd_idx + 28:mdhd_idx + 32], 'big')
                                track_duration = int.from_bytes(header[mdhd_idx + 32:mdhd_idx + 40], 'big')

                            # stsz: sample_count @ +12 (from type offset)
                            sample_count = None
                            if stsz_idx + 16 <= len(header):
                                sample_count = int.from_bytes(header[stsz_idx + 12:stsz_idx + 16], 'big')

                            if (
                                track_timescale is not None
                                and track_duration is not None
                                and track_timescale > 0
                                and track_duration > 0
                                and sample_count is not None
                                and sample_count > 0
                            ):
                                duration_seconds = float(track_duration) / float(track_timescale)
                                if duration_seconds > 0:
                                    metadata["fps"] = int(round((float(sample_count) / duration_seconds), 0))
                        except Exception:
                            pass

                # Look for next 'trak' box
                trak_idx = header.find(b'trak', trak_idx + 4)

    return metadata

def parse_mov_header(header: bytes) -> dict:
    """
    Parses the header of a MOV file to extract metadata such as format, resolution, and duration.
    This is a simplified parser that looks for specific byte patterns in the header.
    """
    # MOV files are structurally similar to MP4 (both are based on the ISO Base Media File Format)
    # We can reuse the MP4 parsing logic with minor adjustments if needed
    metadata = parse_mp4_header(header)
    if "error" in metadata:
        return {"format": "MOV", "error": "Invalid MOV/MP4 signature"}
    
    metadata["format"] = "MOV"
    return metadata

def parse_video_header(path:Union[str, Path]) -> dict:
    """
    Parses the video header to extract metadata such as format, resolution, and duration.
    This function dispatches to specific parsers based on the detected format.
    """
    p = Path(path)
    extension_name = p.suffix.lower().lstrip(".")

    try:
        extension = FileExtensionType(extension_name)
    except ValueError:
        return {"error": f"Unsupported extension: {extension_name or 'none'}"}
    try:
        data = get_header(p)
    except Exception as exc:
        return {"error": f"Cannot read file: {exc}"}
    if extension == FileExtensionType.AVI:
        return parse_avi_header(data)
    if extension == FileExtensionType.MP4:
        if b"moov" in data:
            return parse_mp4_header(data)
        try:
            tail = get_tail(p)
        except Exception:
            tail = b""
        return parse_mp4_header(data + tail)
    if extension == FileExtensionType.MOV:
        if b"moov" in data:
            return parse_mov_header(data)
        try:
            tail = get_tail(p)
        except Exception:
            tail = b""
        return parse_mov_header(data + tail)

    return {"error": "Unsupported or unknown video format"}
