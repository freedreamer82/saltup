import numpy as np
import cv2


def load_grayscale_image(image_path) -> np.ndarray:
    """
    Load an image in grayscale using OpenCV
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image


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
    Invert the pixel of a uint8 image

    Args:
        img (np.ndarray): an array of uint8 image

    Returns:
        np.ndarray: the pixel inverted array image
    """
    return cv2.bitwise_not(img)
