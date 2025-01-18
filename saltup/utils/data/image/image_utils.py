import numpy as np
import cv2
from enum import IntEnum, auto
from pathlib import Path
from typing import Union

class ColorMode(IntEnum):
    RGB = auto()
    BGR = auto()
    GRAY = auto()

class ImageFormat(IntEnum):
    HWC = auto()  # Height, Width, Channels (default)
    CHW = auto()  # Channels, Height, Width

def convert_image_format(image: np.ndarray, target_format: ImageFormat) -> np.ndarray:
    """Convert an image between HWC and CHW formats.

    Args:
        image: Input image as a NumPy array.
        target_format: Target format ("HWC" or "CHW").

    Returns:
        np.ndarray: Image in the target format.

    Raises:
        ValueError: If the input image has invalid dimensions or the conversion fails.
    """
    if len(image.shape) not in {2, 3}:
        raise ValueError(f"Invalid image shape: {image.shape}. Expected 2D (H, W) or 3D (H, W, C).")

    # If the image is grayscale (2D), expand it to 3D (H, W, 1)
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)

    # Check if the image is already in the target format
    if target_format == ImageFormat.CHW:
        if len(image.shape) == 3 and image.shape[2] <= 4:  # HWC format
            # Convert HWC to CHW
            return np.transpose(image, (2, 0, 1))
        elif len(image.shape) == 3 and image.shape[0] <= 4:  # Already CHW
            return image  # No conversion needed
    elif target_format == ImageFormat.HWC:
        if len(image.shape) == 3 and image.shape[0] <= 4:  # CHW format
            # Convert CHW to HWC
            return np.transpose(image, (1, 2, 0))
        elif len(image.shape) == 3 and image.shape[2] <= 4:  # Already HWC
            return image  # No conversion needed
    else:
        raise ValueError(f"Unsupported target format: {target_format}")

    # If we reach here, the image shape is invalid for the target format
    raise ValueError(f"Cannot convert image with shape {image.shape} to {target_format} format.")



def load_image(
    image_path: Union[str, Path], 
    color_mode: ColorMode = ColorMode.BGR, 
    image_format: ImageFormat = ImageFormat.HWC
) -> np.ndarray:
    """Load and convert image to specified color mode and format.

    Args:
        image_path: Path to the image file (as string or Path object).
        color_mode: Target color mode ("RGB", "BGR", or "GRAY").
        image_format: Target image format ("HWC" or "CHW").

    Returns:
        np.ndarray: Image in specified color mode and format.

    Raises:
        FileNotFoundError: If image file does not exist or cannot be loaded.
        ValueError: If color conversion or format conversion fails.
    """
    # Convert image_path to Path object if it's a string
    image_path = Path(image_path) if isinstance(image_path, str) else image_path

    # Verify file exists
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Load image in BGR (OpenCV default)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")

    # Convert to desired color mode
    try:
        if color_mode == ColorMode.RGB:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif color_mode == ColorMode.GRAY:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = np.expand_dims(image, axis=-1)  # Ensure 3D shape (H, W, 1)
    except cv2.error as e:
        raise ValueError(f"Error converting image color mode: {e}")

    # Check if the image is already in the desired format
    if image_format == ImageFormat.HWC:
        if len(image.shape) == 3 and image.shape[2] <= 4:  # Already HWC
            return image  # No conversion needed
    elif image_format == ImageFormat.CHW:
        if len(image.shape) == 3 and image.shape[0] <= 4:  # Already CHW
            return image  # No conversion needed

    # Convert to desired image format if necessary
    try:
        return convert_image_format(image, image_format)
    except Exception as e:
        raise ValueError(f"Error converting image format: {e}")
    
    
def jpg_to_raw_array(input_file: str, grayscale: bool = False) -> np.ndarray:
    '''
    Conversion into raw format of given JPEG image as given from 'path'.
    Args:
        - path: str variable where the image of interested is located.
        - grayscale: bool, whether to convert to grayscale
    Return:
        - img_raw_pixels: np.ndarray file i.e. the raw file
    '''
    if grayscale:
        img_data = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
    else:
        # OpenCV reads in BGR, convert to RGB
        img_data = cv2.imread(input_file)
        img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)

    return img_data


def resize_image(image: np.ndarray, new_size: tuple) -> np.ndarray:
    """
    Resize image using OpenCV
    """
    # OpenCV resize expects (width, height) order
    resized = cv2.resize(image, new_size)
    return resized


def crop_image(image: np.ndarray, crop_window: dict) -> np.ndarray:
    """
    Crops the input image according to the specified window.

    Args:
        image (np.ndarray): The input image to be cropped
        crop_window (dict): A dictionary specifying the cropping window with the following keys:
            - 'x_min' (int): The starting x-coordinate of the crop window
            - 'y_min' (int): The starting y-coordinate of the crop window
            - 'x_max' (int): The ending x-coordinate of the crop window
            - 'y_max' (int): The ending y-coordinate of the crop window

    Returns:
        np.ndarray: The cropped image
    """
    return image[crop_window['y_min']:crop_window['y_max'],
                 crop_window['x_min']:crop_window['x_max']]


def save_raw_image(image: np.ndarray, dest_path: str):
    """
    Save raw image data to file
    """
    with open(dest_path, 'wb') as f:
        f.write(image.tobytes())


def save_jpg_image(image: np.ndarray, dest_path: str):
    """
    Save image as JPG using OpenCV
    """
    # If image is RGB, convert to BGR for OpenCV
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(dest_path, image)


def invert_pixel(img: np.ndarray) -> np.ndarray:
    """
    Invert the pixel of a uint8 image while preserving the shape.

    Args:
        img (np.ndarray): An array of uint8 image (2D or 3D).

    Returns:
        np.ndarray: The pixel-inverted array image with the same shape as input.
    """
    # Invert the pixel values
    inverted_img = cv2.bitwise_not(img)

    # If the input was 3D with a single channel, restore the shape
    if len(img.shape) == 3 and img.shape[2] == 1:
        inverted_img = np.expand_dims(inverted_img, axis=-1)

    return inverted_img
